from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, VlmPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Alias for compatibility with specifications expecting `VlmPdfPipeline`
VlmPdfPipeline = VlmPipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Module-level cached converter instance to avoid re-allocating heavy model weights
_CACHED_CONVERTER: DocumentConverter | None = None


def get_converter(
    *,
    device: str = "auto",
    torch_dtype: str = "float32",
    repo_id: str = "ibm-granite/granite-docling-258M",
) -> DocumentConverter:
    """Configure and initialize Docling's DocumentConverter with the Granite Docling VLM pipeline.

    Hardware & Precision Tuning:
    - Target GPU: NVIDIA GeForce GTX 1650 (Turing architecture, CC 7.5, 4 GB VRAM).
    - Precision: Turing lacks native bfloat16 tensor core acceleration; torch_dtype is explicitly
      set to 'float32' to prevent numerical instability, corrupted tokens, or kernel crashes.
    - Memory & Optimizations: Flash Attention 2 is unsupported on CC 7.5 and is disabled.
      Quantization defaults in GRANITEDOCLING_TRANSFORMERS (load_in_8bit=True) are preserved
      to keep peak VRAM safely within the 4 GB hardware ceiling.

    Args:
        device: Hardware device target ('auto', 'cuda', or 'cpu'). Defaults to 'auto'.
        torch_dtype: PyTorch tensor datatype ('float32'). Defaults to 'float32'.
        repo_id: Hugging Face model repository identifier. Defaults to
            'ibm-granite/granite-docling-258M'.

    Returns:
        DocumentConverter: Configured Docling converter utilizing VlmPipeline for PDF documents.
    """
    logger.info(
        "Initializing DocumentConverter with Granite Docling VLM "
        "(model=%s, dtype=%s, device=%s)...",
        repo_id,
        torch_dtype,
        device,
    )

    # Configure VLM options using the pre-tuned Granite Docling transformer spec
    vlm_options = GRANITEDOCLING_TRANSFORMERS.model_copy(
        update={
            "repo_id": repo_id,
            "torch_dtype": torch_dtype,
        }
    )

    accelerator_options = AcceleratorOptions(
        device=device,
        cuda_use_flash_attention2=False,  # Unsupported on GTX 1650 (requires Ampere+ CC 8.0+)
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        accelerator_options=accelerator_options,
        generate_page_images=True,
    )

    format_options: dict[InputFormat, Any] = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }

    return DocumentConverter(format_options=format_options)


def extract_pdf(
    pdf_path: Path | str,
    cache_dir: Path | str,
    converter: DocumentConverter | None = None,
) -> dict[str, Any]:
    """Extract structured content from a PDF document, caching the serialized JSON result.

    This function first inspects the target cache directory for an existing JSON representation
    of the extracted PDF. If found (cache hit), the cached extraction is loaded directly from
    disk without engaging the GPU. On a cache miss, it triggers the heavy VLM pipeline,
    serializes the resulting DoclingDocument to a dictionary, writes it atomically to the cache,
    and returns the structured data.

    Hardware & Fault Tolerance:
    - On GTX 1650 (4 GB VRAM), heavy VLM conversion can occasionally encounter CUDA OOM
      on very dense or high-DPI pages. CUDA memory caches are cleared and informative
      hardware diagnostics are logged upon memory failure.
    - Cache writes use atomic temporary file replacement to avoid leaving partial or corrupt
      JSON files in case of interruption.

    Args:
        pdf_path: Path to the target PDF file on disk.
        cache_dir: Directory where serialized extraction JSON files are stored.
        converter: Optional pre-configured DocumentConverter. If None, a cached module-level
            converter is reused to avoid expensive model reloading.

    Returns:
        dict[str, Any]: Serializable dictionary representation of the extracted DoclingDocument.

    Raises:
        FileNotFoundError: If the provided pdf_path does not exist or is not a file.
        ValueError: If pdf_path is not a valid PDF file.
        RuntimeError: If document conversion fails due to malformed PDF structures or CUDA OOM.
    """
    global _CACHED_CONVERTER

    path = Path(pdf_path).resolve()
    target_cache_dir = Path(cache_dir).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, but received: {path.name}")

    target_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = target_cache_dir / f"{path.stem}.json"

    # 1. Check Cache
    if cache_file.is_file():
        logger.info("[CACHE HIT] Found existing extraction cache for '%s' at: %s", path.name, cache_file)
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as err:
            logger.warning(
                "[CACHE CORRUPT] Failed reading cached JSON for '%s' (%s). Re-running extraction...",
                path.name,
                err,
            )

    # 2. Cache Miss: Run Heavy VLM Conversion
    logger.info("[CACHE MISS] Running Granite Docling VLM extraction on: %s", path.name)

    if converter is None:
        if _CACHED_CONVERTER is None:
            _CACHED_CONVERTER = get_converter()
        converter = _CACHED_CONVERTER

    try:
        conversion_result = converter.convert(path)
        doc = conversion_result.document
        extracted_dict: dict[str, Any] = doc.export_to_dict()

        # 3. Atomic Cache Write
        temp_cache_file = cache_file.with_suffix(".tmp")
        with open(temp_cache_file, "w", encoding="utf-8") as f:
            json.dump(extracted_dict, f, indent=2, ensure_ascii=False)
        temp_cache_file.replace(cache_file)

        logger.info(
            "[CACHE SAVED] Successfully extracted and cached '%s' (%d bytes) to: %s",
            path.name,
            cache_file.stat().st_size,
            cache_file,
        )
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


def main() -> None:
    """CLI entry point to extract and cache a PDF using Docling VLM."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract STEM textbook PDF content using Docling and Granite Docling VLM."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Directory to store/load extraction cache (default: ./cache).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for VLM inference (default: auto).",
    )
    args = parser.parse_args()

    try:
        converter = get_converter(device=args.device)
        result = extract_pdf(args.pdf_path, args.cache_dir, converter=converter)
        print(f"Extraction successful: {len(result.get('body', {}))} top-level elements extracted.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
