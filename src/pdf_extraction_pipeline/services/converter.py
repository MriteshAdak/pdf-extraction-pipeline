from __future__ import annotations

import logging
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, VlmPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from pdf_extraction_pipeline.config import (
    DEFAULT_DEVICE,
    DEFAULT_REPO_ID,
    DEFAULT_TORCH_DTYPE,
    HardwareConfig,
)

logger = logging.getLogger(__name__)


class ConverterService:
    """Manages Docling DocumentConverter configuration, VLM pipeline options, and instance lifecycle.

    Hardware & Precision Tuning:
    - Target GPU: NVIDIA GeForce GTX 1650 (Turing architecture, CC 7.5, 4 GB VRAM).
    - Precision: Turing lacks native bfloat16 tensor core acceleration; torch_dtype is explicitly
      set to 'float32' to prevent numerical instability, corrupted tokens, or kernel crashes.
    - Memory & Optimizations: Flash Attention 2 is unsupported on CC 7.5 and is disabled.
      Quantization defaults in GRANITEDOCLING_TRANSFORMERS (load_in_8bit=True) are preserved
      to keep peak VRAM safely within the 4 GB hardware ceiling.
    """

    def __init__(
        self,
        config: HardwareConfig | None = None,
        *,
        device: str = DEFAULT_DEVICE,
        torch_dtype: str = DEFAULT_TORCH_DTYPE,
        repo_id: str = DEFAULT_REPO_ID,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = HardwareConfig(
                device=device,
                torch_dtype=torch_dtype,
                repo_id=repo_id,
            )
        self._converter: DocumentConverter | None = None

    def build_converter(self) -> DocumentConverter:
        """Create a new DocumentConverter instance configured with Granite Docling VLM."""
        logger.info(
            "Initializing DocumentConverter with Granite Docling VLM "
            "(model=%s, dtype=%s, device=%s)...",
            self.config.repo_id,
            self.config.torch_dtype,
            self.config.device,
        )

        vlm_options = GRANITEDOCLING_TRANSFORMERS.model_copy(
            update={
                "repo_id": self.config.repo_id,
                "torch_dtype": self.config.torch_dtype,
            }
        )

        accelerator_options = AcceleratorOptions(
            device=self.config.device,
            cuda_use_flash_attention2=self.config.cuda_use_flash_attention2,
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            accelerator_options=accelerator_options,
            generate_page_images=self.config.generate_page_images,
        )

        format_options: dict[InputFormat, Any] = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }

        return DocumentConverter(format_options=format_options)

    def get_converter(self) -> DocumentConverter:
        """Return the cached DocumentConverter instance, initializing it lazily on first access."""
        if self._converter is None:
            self._converter = self.build_converter()
        return self._converter

    def set_converter(self, converter: DocumentConverter) -> None:
        """Explicitly set the underlying DocumentConverter instance."""
        self._converter = converter

    def clear(self) -> None:
        """Release the cached converter reference."""
        self._converter = None
