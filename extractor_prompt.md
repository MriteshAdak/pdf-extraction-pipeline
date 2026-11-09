# Prompt for Antigravity: Implement `src/extractor.py`

You are implementing the core PDF extraction module (`src/extractor.py`) for a RAG application designed to parse complex STEM textbooks containing formulas (both inline and block LaTeX), tables, diagrams, and multi-column layouts. 

No "vibe-coded slop" or placeholder comments are allowed. Write clean, production-grade Python code adhering strictly to the specifications below.

## Environment & Hardware Constraints
- **Hardware:** WSL Debian, 16 GB system RAM, 4 GB VRAM GTX 1650.
- **Precision Requirement:** Turing GPUs (GTX 1650) lack native `bfloat16` support. You must explicitly configure the VLM pipeline to use `float32` precision to prevent pipeline crashes or garbled outputs.
- **Dependencies:** Use `docling` with `docling[vlm]` utilizing the `ibm-granite/granite-docling-258M` model via PyTorch. Do **not** use Ollama.

## Implementation Requirements

1. **Module Structure (`src/extractor.py`):**
   - Implement a function `get_converter()` that initializes Docling's `DocumentConverter` with a `VlmPdfPipeline`.
   - Configure `VlmPipelineOptions` to use `ibm-granite/granite-docling-258M` and set `torch_dtype="float32"`.
   - Implement a function `extract_pdf(pdf_path: Path, cache_dir: Path) -> dict` that checks for an existing cached JSON extraction. If it doesn't exist, run the conversion, export the resulting `DoclingDocument` to a serializable dictionary, cache it to disk, and return it.

2. **Error Handling & Logging:**
   - Include clear `print` or `logging` statements to indicate when cache hits occur versus when the heavy VLM conversion is actively running.
   - Handle missing files or malformed PDF exceptions gracefully.

3. **Type Hinting & Docstrings:**
   - Fully type-hint all functions using `from __future__ import annotations`.
   - Provide concise docstrings detailing the purpose of each function and its handling of hardware constraints.

## Target Code Outline
```python
from __future__ import annotations

import json
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, VlmPdfPipeline

def get_converter() -> DocumentConverter:
    """Configure Docling with Granite Docling VLM, tuned for GTX 1650 (4GB VRAM, float32)."""
    # TODO: Implement pipeline options, set torch_dtype="float32", model_name, and return DocumentConverter
    pass

def extract_pdf(pdf_path: Path, cache_dir: Path) -> dict:
    """Extract a PDF into a structured document, caching the result to disk."""
    # TODO: Check cache, run extraction if missing, serialize to JSON, and return dict
    pass
```