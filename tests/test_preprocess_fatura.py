# tests/test_preprocess_fatura.py
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from src.stages.preprocess_fatura import extract_ocr_from_images


def test_preprocess_creates_csv(tmp_path, monkeypatch):
    # --- Arrange ---
    # Create a mock image folder and a valid dummy image
    raw_dir = tmp_path / "FATURA"
    raw_dir.mkdir()
    dummy_img = raw_dir / "dummy.jpg"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(dummy_img)

    # Monkeypatch Tesseract to return constant text
    # Must accept *args and **kwargs because your code passes config=...
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "TEST_TEXT"
    )

    # Redirect pipeline paths to temporary locations
    monkeypatch.setattr("src.stages.preprocess_fatura.RAW_DIR", raw_dir)

    # Monkeypatch OUT_FILE to temporary path
    out_file = tmp_path / "fatura_ocr.csv"
    monkeypatch.setattr("src.stages.preprocess_fatura.OUT_FILE", out_file)

    # Monkeypatch CACHE_FILE to prevent saving to real data/processed/
    monkeypatch.setattr(
        "src.stages.preprocess_fatura.CACHE_FILE",
        tmp_path / "fatura_ocr_cache.csv"
    )

    # --- Act ---
    extract_ocr_from_images()

    # --- Assert ---
    assert out_file.exists(), "Output CSV was not created"
    df = pd.read_csv(out_file)
    assert "ocr_text" in df.columns, "Missing OCR text column"
    assert df.shape[0] == 1, "Expected exactly one row of OCR output"
