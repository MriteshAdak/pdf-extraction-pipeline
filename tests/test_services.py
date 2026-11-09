from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.pipeline.vlm_pipeline import VlmPipeline

from pdf_extraction_pipeline.config import HardwareConfig
from pdf_extraction_pipeline.services.batch import BatchService
from pdf_extraction_pipeline.services.cache import CacheService
from pdf_extraction_pipeline.services.converter import ConverterService
from pdf_extraction_pipeline.services.extraction import ExtractionService


# ============================================================================
# CacheService Tests (Fast: No Torch / Docling dependency needed)
# ============================================================================


def test_cache_service_get_cache_file(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = Path("/some/path/document.pdf")
    assert cache_svc.get_cache_file(pdf) == tmp_path.resolve() / "document.json"


def test_cache_service_should_parse_missing(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    assert cache_svc.should_parse(pdf) is True


def test_cache_service_should_parse_up_to_date(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    time.sleep(0.01)

    json_file = cache_svc.get_cache_file(pdf)
    json_file.write_text('{"status": "ok"}', encoding="utf-8")

    assert cache_svc.should_parse(pdf) is False


def test_cache_service_save_and_load_roundtrip(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    data = {"text": "hello world", "formulas": ["\\sum_i x_i"]}
    saved_path = cache_svc.save(pdf, data)
    assert saved_path.is_file()

    loaded = cache_svc.load(pdf)
    assert loaded == data


def test_cache_service_load_corrupt_returns_none(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "book.pdf"
    cache_file = cache_svc.get_cache_file(pdf)
    cache_file.write_text("NOT A VALID JSON", encoding="utf-8")

    assert cache_svc.load(pdf) is None


# ============================================================================
# ConverterService Tests
# ============================================================================


def test_converter_service_config_defaults() -> None:
    svc = ConverterService(device="cpu")
    assert svc.config.device == "cpu"
    assert svc.config.torch_dtype == "float32"
    assert svc.config.cuda_use_flash_attention2 is False
    assert svc.config.repo_id == "ibm-granite/granite-docling-258M"


def test_converter_service_build_and_get() -> None:
    svc = ConverterService(device="cpu")
    converter = svc.get_converter()
    assert isinstance(converter, DocumentConverter)

    # Verify Docling VLM pipeline configuration
    pdf_option = converter.format_to_options[InputFormat.PDF]
    assert pdf_option.pipeline_cls is VlmPipeline

    pipeline_opts = pdf_option.pipeline_options
    assert pipeline_opts is not None
    assert pipeline_opts.vlm_options.repo_id == "ibm-granite/granite-docling-258M"
    assert pipeline_opts.vlm_options.torch_dtype == "float32"
    assert pipeline_opts.accelerator_options.cuda_use_flash_attention2 is False

    # Calling get_converter again reuses the cached instance
    assert svc.get_converter() is converter

    # Clear releases it
    svc.clear()
    assert svc._converter is None


# ============================================================================
# ExtractionService Tests
# ============================================================================


def test_extraction_service_missing_file(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    svc = ExtractionService(cache_service=cache_svc)
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        svc.extract(tmp_path / "nonexistent.pdf")


def test_extraction_service_invalid_extension(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    svc = ExtractionService(cache_service=cache_svc)
    text_file = tmp_path / "test.txt"
    text_file.write_text("hello")
    with pytest.raises(ValueError, match="Expected a PDF file"):
        svc.extract(text_file)


def test_extraction_service_cache_hit(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "cached_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    cached_content = {"cached": True}
    cache_svc.save(pdf, cached_content)

    mock_converter = MagicMock()
    svc = ExtractionService(cache_service=cache_svc)

    result = svc.extract(pdf, converter=mock_converter)
    assert result == cached_content
    mock_converter.convert.assert_not_called()


def test_extraction_service_cuda_oom_handling(tmp_path: Path) -> None:
    cache_svc = CacheService(tmp_path)
    pdf = tmp_path / "oom_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_converter = MagicMock()
    mock_converter.convert.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

    svc = ExtractionService(cache_service=cache_svc)
    with pytest.raises(RuntimeError, match="CUDA Out Of Memory"):
        svc.extract(pdf, converter=mock_converter)


# ============================================================================
# BatchService Tests
# ============================================================================


def test_batch_service_discover_and_filter(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    pdf1 = docs_dir / "a_doc.pdf"
    pdf1.write_bytes(b"%PDF-1.4 a")
    pdf2 = docs_dir / "b_doc.PDF"
    pdf2.write_bytes(b"%PDF-1.4 b")
    (docs_dir / "not_pdf.txt").write_text("text")

    batch_svc = BatchService(cache_service=CacheService(parsed_dir))
    discovered = batch_svc.discover_pdfs(docs_dir)
    assert discovered == [pdf1.resolve(), pdf2.resolve()]

    # Make pdf1 already parsed
    time.sleep(0.01)
    (parsed_dir / "a_doc.json").write_text('{"cached": 1}', encoding="utf-8")

    pending = batch_svc.filter_pending(discovered)
    assert pending == [pdf2.resolve()]

    # If force=True, returns all
    assert batch_svc.filter_pending(discovered, force=True) == [pdf1.resolve(), pdf2.resolve()]
