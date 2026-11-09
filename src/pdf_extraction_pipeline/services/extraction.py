from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from docling.document_converter import DocumentConverter

from pdf_extraction_pipeline.services.cache import CacheService
from pdf_extraction_pipeline.services.converter import ConverterService

logger = logging.getLogger(__name__)


class ExtractionService:
    """Coordinates PDF document extraction using Docling VLM, cache checking, and fault tolerance."""

    def __init__(
        self,
        converter_service: ConverterService | None = None,
        cache_service: CacheService | None = None,
    ) -> None:
        self.converter_service = converter_service or ConverterService()
        self.cache_service = cache_service or CacheService()

    def extract(
        self,
        pdf_path: Path | str,
        *,
        force: bool = False,
        converter: DocumentConverter | None = None,
    ) -> dict[str, Any]:
        """Extract structured content from a PDF document, caching the serialized JSON result.

        Args:
            pdf_path: Path to the target PDF file on disk.
            force: If True, forces re-extraction even if a valid output JSON already exists.
            converter: Optional pre-configured DocumentConverter. If None, uses converter_service.

        Returns:
            dict[str, Any]: Serializable dictionary representation of the extracted DoclingDocument.

        Raises:
            FileNotFoundError: If the provided pdf_path does not exist or is not a file.
            ValueError: If pdf_path is not a valid PDF file.
            RuntimeError: If document conversion fails due to malformed PDF structures or CUDA OOM.
        """
        path = Path(pdf_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, but received: {path.name}")

        cache_file = self.cache_service.get_cache_file(path)

        # 1. Check Cache
        if not force and cache_file.is_file() and not self.cache_service.should_parse(path):
            cached_data = self.cache_service.load(path)
            if cached_data is not None:
                logger.info(
                    "[CACHE HIT] Found up-to-date extraction cache for '%s' at: %s",
                    path.name,
                    cache_file,
                )
                return cached_data

        # 2. Cache Miss: Run Heavy VLM Conversion
        logger.info("[CACHE MISS] Running Granite Docling VLM extraction on: %s", path.name)

        active_converter = converter or self.converter_service.get_converter()

        try:
            conversion_result = active_converter.convert(path)
            doc = conversion_result.document
            extracted_dict: dict[str, Any] = doc.export_to_dict()

            # 3. Atomic Cache Write
            self.cache_service.save(path, extracted_dict)
            return extracted_dict

        except torch.cuda.OutOfMemoryError as oom_err:
            logger.error(
                "CUDA Out Of Memory encountered while converting '%s' on GTX 1650 (4GB VRAM). "
                "Clearing CUDA cache...",
                path.name,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"VLM conversion failed due to CUDA Out Of Memory on {path.name}. "
                f"Consider running with device='cpu' or reducing image scale."
            ) from oom_err

        except Exception as err:
            logger.error("Failed to extract PDF '%s': %s", path.name, err, exc_info=True)
            raise RuntimeError(f"Error during PDF extraction for {path.name}: {err}") from err
