from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.pipeline.vlm_pipeline import VlmPipeline

from src.extractor import VlmPdfPipeline, extract_pdf, get_converter


def test_vlm_pdf_pipeline_alias() -> None:
    """Verify that VlmPdfPipeline is available and aliases VlmPipeline."""
    assert VlmPdfPipeline is VlmPipeline


def test_get_converter_configuration() -> None:
    """Verify that get_converter configures VlmPipeline with Granite and float32."""
    converter = get_converter(device="cpu")
    assert isinstance(converter, DocumentConverter)

    pdf_option = converter.format_to_options[InputFormat.PDF]
    assert pdf_option.pipeline_cls is VlmPipeline

    pipeline_opts = pdf_option.pipeline_options
    assert pipeline_opts is not None

    vlm_opts = pipeline_opts.vlm_options
    assert vlm_opts.repo_id == "ibm-granite/granite-docling-258M"
    assert vlm_opts.torch_dtype == "float32"

    acc_opts = pipeline_opts.accelerator_options
    assert acc_opts.cuda_use_flash_attention2 is False


def test_extract_pdf_missing_file(tmp_path: Path) -> None:
    """Verify that a non-existent PDF raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.pdf"
    cache_dir = tmp_path / "cache"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf(non_existent, cache_dir)


def test_extract_pdf_invalid_extension(tmp_path: Path) -> None:
    """Verify that a non-PDF file raises ValueError."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Not a PDF")
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="Expected a PDF file"):
        extract_pdf(text_file, cache_dir)


def test_extract_pdf_cache_hit(tmp_path: Path) -> None:
    """Verify that extract_pdf returns cached data without calling converter."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    cached_file = cache_dir / "sample.json"
    cached_content = {"document": "pre-cached", "equations": ["E = mc^2"]}
    cached_file.write_text(json.dumps(cached_content), encoding="utf-8")

    # Pass a dummy converter that would fail if called
    mock_converter = MagicMock()

    result = extract_pdf(pdf_file, cache_dir, converter=mock_converter)
    assert result == cached_content
    mock_converter.convert.assert_not_called()


def test_extract_pdf_cache_miss_and_save(tmp_path: Path) -> None:
    """Verify that extract_pdf executes conversion on cache miss, caches, and returns."""
    pdf_file = tmp_path / "chapter1.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test pdf content")

    cache_dir = tmp_path / "cache"

    expected_dict = {
        "schema_name": "DoclingDocument",
        "body": {
            "text": "Riemann hypothesis and analytic number theory.",
            "tables": [],
        },
    }

    mock_doc = MagicMock()
    mock_doc.export_to_dict.return_value = expected_dict

    mock_conv_result = MagicMock()
    mock_conv_result.document = mock_doc

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_conv_result

    # 1. First call: Cache miss
    result = extract_pdf(pdf_file, cache_dir, converter=mock_converter)
    assert result == expected_dict
    mock_converter.convert.assert_called_once_with(pdf_file.resolve())

    # Verify JSON file written to cache
    cached_file = cache_dir / "chapter1.json"
    assert cached_file.exists()
    assert json.loads(cached_file.read_text(encoding="utf-8")) == expected_dict

    # 2. Second call: Should hit cache now
    mock_converter.reset_mock()
    second_result = extract_pdf(pdf_file, cache_dir, converter=mock_converter)
    assert second_result == expected_dict
    mock_converter.convert.assert_not_called()
