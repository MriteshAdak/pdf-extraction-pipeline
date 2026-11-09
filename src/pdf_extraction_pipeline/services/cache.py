from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pdf_extraction_pipeline.config import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


class CacheService:
    """Manages disk caching, staleness detection, and atomic persistence for parsed JSON documents."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir).resolve()

    def get_cache_file(self, pdf_path: Path | str) -> Path:
        """Return the target cached JSON file path for a given PDF."""
        pdf = Path(pdf_path).resolve()
        return self.cache_dir / f"{pdf.stem}.json"

    def should_parse(self, pdf_path: Path | str) -> bool:
        """Determine whether a PDF needs extraction based on output presence and timestamps.

        Returns True if:
        - The output JSON file does not exist in cache_dir.
        - The output JSON file is empty (0 bytes).
        - The source PDF has been modified more recently than the output JSON.

        Returns False if an up-to-date, non-empty JSON output already exists.
        """
        pdf = Path(pdf_path).resolve()
        target_json = self.get_cache_file(pdf)

        if not target_json.is_file():
            return True

        try:
            if target_json.stat().st_size == 0:
                return True
            return pdf.stat().st_mtime > target_json.stat().st_mtime
        except OSError:
            return True

    def load(self, pdf_path: Path | str) -> dict[str, Any] | None:
        """Load cached JSON data for a PDF if it exists and is readable.

        Returns None if the cache file is absent or corrupted.
        """
        pdf = Path(pdf_path).resolve()
        cache_file = self.get_cache_file(pdf)

        if not cache_file.is_file():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as err:
            logger.warning(
                "[CACHE CORRUPT] Failed reading cached JSON for '%s' (%s). Re-running extraction...",
                pdf.name,
                err,
            )
            return None

    def save(self, pdf_path: Path | str, data: dict[str, Any]) -> Path:
        """Atomically persist parsed dictionary to the cache directory as formatted JSON.

        Uses a temporary file with replacement to prevent corrupt files on interruption.
        """
        pdf = Path(pdf_path).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.get_cache_file(pdf)

        temp_cache_file = cache_file.with_suffix(".tmp")
        with open(temp_cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_cache_file.replace(cache_file)

        logger.info(
            "[CACHE SAVED] Successfully extracted and cached '%s' (%d bytes) to: %s",
            pdf.name,
            cache_file.stat().st_size,
            cache_file,
        )
        return cache_file
