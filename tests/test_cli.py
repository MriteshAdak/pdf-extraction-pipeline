from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_extraction_pipeline.cli import main, parse_args


def test_parse_args_defaults() -> None:
    """Verify CLI argument defaults are docs/ and parsed/."""
    args = parse_args([])
    assert args.input_path == Path("./docs")
    assert args.output_dir == Path("./parsed")
    assert args.device == "auto"
    assert args.force is False


def test_parse_args_custom() -> None:
    """Verify CLI argument custom values and aliases."""
    args = parse_args(["my_doc.pdf", "--cache-dir", "custom_cache", "--device", "cpu", "-f"])
    assert args.input_path == Path("my_doc.pdf")
    assert args.output_dir == Path("custom_cache")
    assert args.device == "cpu"
    assert args.force is True


def test_cli_main_missing_input(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI exits with code 1 when input path does not exist."""
    missing = tmp_path / "missing.pdf"
    with pytest.raises(SystemExit) as exc_info:
        main([str(missing)])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Input path not found" in captured.err


def test_cli_main_invalid_extension(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI exits with code 1 when file is not a PDF."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not a pdf")
    with pytest.raises(SystemExit) as exc_info:
        main([str(txt_file)])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Expected a PDF file" in captured.err


def test_cli_main_batch_directory(tmp_path: Path) -> None:
    """Verify CLI invokes BatchService when input is a directory."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    parsed_dir = tmp_path / "parsed"

    with patch("pdf_extraction_pipeline.cli.BatchService") as mock_batch_cls:
        mock_batch = MagicMock()
        mock_batch.process_directory.return_value = [docs_dir / "sample.pdf"]
        mock_batch_cls.return_value = mock_batch

        main([str(docs_dir), "--output-dir", str(parsed_dir)])
        mock_batch.process_directory.assert_called_once_with(
            input_dir=docs_dir.resolve(),
            output_dir=parsed_dir.resolve(),
            device="auto",
            force=False,
        )
