# tests/test_validate_fatura.py
import pandas as pd
from src.stages.validate_fatura import validate_fatura, INPUT_FILE, REPORT_FILE
from pathlib import Path
import json

def test_validate_reports(tmp_path, monkeypatch):
    # create fake OCR CSV
    test_csv = tmp_path / "fatura_ocr.csv"
    df = pd.DataFrame({"file_name": ["a.jpg", "b.jpg"], "ocr_text": ["ok", ""]})
    df.to_csv(test_csv, index=False)

    monkeypatch.setattr("src.stages.validate_fatura.INPUT_FILE", test_csv)
    report_path = tmp_path / "validation_summary.json"
    monkeypatch.setattr("src.stages.validate_fatura.REPORT_FILE", report_path)

    validate_fatura()

    assert report_path.exists()
    summary = json.loads(report_path.read_text())
    assert "total_records" in summary
