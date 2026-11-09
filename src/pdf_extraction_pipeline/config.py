from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEVICE: str = "auto"
DEFAULT_TORCH_DTYPE: str = "float32"
DEFAULT_REPO_ID: str = "ibm-granite/granite-docling-258M"
DEFAULT_INPUT_DIR: Path = Path("./docs")
DEFAULT_CACHE_DIR: Path = Path("./parsed")


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware and pipeline configuration tuned for GTX 1650 (Turing CC 7.5, 4 GB VRAM)."""

    device: str = DEFAULT_DEVICE
    torch_dtype: str = DEFAULT_TORCH_DTYPE
    cuda_use_flash_attention2: bool = False  # Flash Attention 2 is unsupported on CC 7.5
    generate_page_images: bool = True
    repo_id: str = DEFAULT_REPO_ID
