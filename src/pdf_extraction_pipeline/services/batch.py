from __future__ import annotations

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter

from pdf_extraction_pipeline.config import DEFAULT_CACHE_DIR, DEFAULT_INPUT_DIR
from pdf_extraction_pipeline.services.cache import CacheService
from pdf_extraction_pipeline.services.converter import ConverterService
from pdf_extraction_pipeline.services.extraction import ExtractionService

logger = logging.getLogger(__name__)


class BatchService:
    """Manages batch discovery, filtering of up-to-date documents, and sequential extraction."""

    def __init__(
        self,
        extraction_service: ExtractionService | None = None,
        cache_service: CacheService | None = None,
        converter_service: ConverterService | None = None,
    ) -> None:
        self.cache_service = cache_service or CacheService()
        self.converter_service = converter_service or ConverterService()
        self.extraction_service = extraction_service or ExtractionService(
            converter_service=self.converter_service,
            cache_service=self.cache_service,
        )

    def discover_pdfs(self, input_dir: Path | str) -> list[Path]:
        """Scan input directory for PDF files (case-insensitive) sorted by name."""
        in_dir = Path(input_dir).resolve()

        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {in_dir}")
        if not in_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory, but found file: {in_dir}")

        return sorted(
            {p.resolve() for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"}
        )

    def filter_pending(
        self,
        pdf_files: list[Path],
        *,
        cache_service: CacheService | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Filter PDF files to only those that require parsing."""
        if force:
            return list(pdf_files)
        active_cache = cache_service or self.cache_service
        return [p for p in pdf_files if active_cache.should_parse(p)]

    def process_directory(
        self,
        input_dir: Path | str = DEFAULT_INPUT_DIR,
        output_dir: Path | str = DEFAULT_CACHE_DIR,
        *,
        force: bool = False,
        converter: DocumentConverter | None = None,
        device: str = "auto",
    ) -> list[Path]:
        """Extract all unparsed or modified PDFs in input_dir into output_dir.

        Scans input_dir for PDF documents, filters out files that already have an up-to-date
        JSON output in output_dir (unless force=True), initializes the DocumentConverter once,
        and converts each pending PDF sequentially in a single session.

        Args:
            input_dir: Directory containing input PDF documents (default: ./docs).
            output_dir: Directory to save parsed JSON outputs (default: ./parsed).
            force: If True, forces re-parsing of all PDFs regardless of existing outputs.
            converter: Optional DocumentConverter instance to reuse across batch extractions.
            device: Hardware device target ('auto', 'cuda', or 'cpu') for converter initialization.

        Returns:
            list[Path]: List of PDF file paths that were extracted.
        """
        in_dir = Path(input_dir).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cache_svc = CacheService(out_dir)
        pdf_files = self.discover_pdfs(in_dir)

        if not pdf_files:
            logger.info("No PDF files found in '%s'.", in_dir)
            return []

        pending = self.filter_pending(pdf_files, cache_service=cache_svc, force=force)

        if not pending:
            logger.info(
                "All %d document(s) in '%s' are up to date in '%s'. Nothing to parse.",
                len(pdf_files),
                in_dir.name,
                out_dir.name,
            )
            return []

        logger.info(
            "Found %d PDF(s) in '%s'; %d pending extraction into '%s'.",
            len(pdf_files),
            in_dir.name,
            len(pending),
            out_dir.name,
        )

        active_converter = converter
        if active_converter is None:
            conv_service = ConverterService(device=device)
            active_converter = conv_service.get_converter()

        extract_svc = ExtractionService(
            converter_service=self.converter_service,
            cache_service=cache_svc,
        )

        processed: list[Path] = []
        for idx, pdf in enumerate(pending, start=1):
            logger.info("Processing [%d/%d]: %s", idx, len(pending), pdf.name)
            extract_svc.extract(
                pdf,
                force=force,
                converter=active_converter,
            )
            processed.append(pdf)

        logger.info("Successfully extracted %d document(s) into '%s'.", len(processed), out_dir)
        return processed
