# PDF Extraction Pipeline

A PDF extraction pipeline for complex STEM textbooks containing inline and block LaTeX formulas, tables, diagrams, and multi-column layouts, powered by **Docling** and IBM's **Granite Docling VLM** (`ibm-granite/granite-docling-258M`).

## Environment & Hardware Tuning

- **Hardware:** WSL Debian, 16 GB RAM, NVIDIA GeForce GTX 1650 (4 GB VRAM, Turing CC 7.5).
- **Precision:** Explicitly configured to `float32` (Turing architecture lacks native `bfloat16` acceleration).
- **Memory Optimization:** 8-bit quantization is enabled and Flash Attention 2 is disabled (`cuda_use_flash_attention2=False`, as Turing CC 7.5 requires Ampere+ for Flash Attention 2) to maintain a safe memory footprint under 4 GB VRAM.

## Project Structure

```
├── pyproject.toml
├── src/
│   ├── extractor.py                    # Core extraction and converter implementation
│   └── pdf_extraction_pipeline/        # Package exports
│       ├── __init__.py
│       └── extractor.py
└── tests/
    └── test_extractor.py               # Unit and integration test suite
```

## Quick Start

### 1. Run Tests

Execute the test suite using `uv`:

```bash
uv run pytest
```

### 2. Usage Example

```python
from pathlib import Path
from src.extractor import extract_pdf, get_converter

# Initialize or customize converter (optional, defaults to Granite float32 VLM)
converter = get_converter()

# Extract and cache PDF results
data = extract_pdf(
    pdf_path=Path("sample.pdf"),
    cache_dir=Path("./cache"),
    converter=converter,
)

print(data.keys())
```
