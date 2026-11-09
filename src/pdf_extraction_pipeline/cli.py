from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_extraction_pipeline.config import DEFAULT_CACHE_DIR, DEFAULT_INPUT_DIR
from pdf_extraction_pipeline.services.batch import BatchService
from pdf_extraction_pipeline.services.cache import CacheService
from pdf_extraction_pipeline.services.converter import ConverterService
from pdf_extraction_pipeline.services.extraction import ExtractionService


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for PDF extraction."""
    parser = argparse.ArgumentParser(
        prog="pdf-extraction-pipeline",
        description="Extract STEM textbook PDF content using Docling and Granite Docling VLM.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Path to input PDF file or directory containing PDFs (default: ./docs).",
    )
    parser.add_argument(
        "--output-dir",
        "--cache-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to store/load extraction JSONs (default: ./parsed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for VLM inference (default: auto).",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-parsing even if output JSON is already up to date.",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point to extract PDFs individually or in batch."""
    args = parse_args(argv)
    target_input = Path(args.input_path).resolve()
    target_output = Path(args.output_dir).resolve()

    try:
        if not target_input.exists():
            print(f"Error: Input path not found: {target_input}", file=sys.stderr)
            sys.exit(1)

        cache_svc = CacheService(target_output)

        if target_input.is_dir():
            batch_svc = BatchService()
            processed = batch_svc.process_directory(
                input_dir=target_input,
                output_dir=target_output,
                device=args.device,
                force=args.force,
            )
            print(f"Batch processing complete: {len(processed)} document(s) processed.")
        else:
            if target_input.suffix.lower() != ".pdf":
                print(
                    f"Error: Expected a PDF file, but received: {target_input.name}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not args.force and not cache_svc.should_parse(target_input):
                output_file = cache_svc.get_cache_file(target_input)
                print(
                    f"Document '{target_input.name}' is already parsed and up to date at: {output_file}"
                )
                print("Use --force (-f) to re-parse.")
                return

            conv_svc = ConverterService(device=args.device)
            converter = conv_svc.get_converter()
            extract_svc = ExtractionService(converter_service=conv_svc, cache_service=cache_svc)
            result = extract_svc.extract(
                target_input,
                force=args.force,
                converter=converter,
            )
            output_file = cache_svc.get_cache_file(target_input)
            print(
                f"Extraction successful: {len(result.get('body', {}))} top-level elements extracted."
            )
            print(f"Parsed output saved to: {output_file}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
