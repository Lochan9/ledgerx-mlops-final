import pandas as pd
from loguru import logger
from pathlib import Path
import json

INPUT_FILE = Path("data/processed/fatura_ocr.csv")
REPORT_FILE = Path("reports/validation_summary.json")

def validate_fatura():
    logger.info(f"Loading OCR output from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    total = len(df)
    missing_text = df["ocr_text"].isna().sum()
    empty_text = (df["ocr_text"].str.strip() == "").sum()
    duplicate_files = df["file_name"].duplicated().sum()

    summary = {
        "total_records": int(total),
        "missing_text": int(missing_text),
        "empty_text": int(empty_text),
        "duplicate_files": int(duplicate_files),
        "valid_records": int(total - (missing_text + empty_text))
    }

    logger.info(f"Validation summary: {summary}")

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    if missing_text or empty_text or duplicate_files:
        logger.warning("⚠️  Some data quality issues found.")
    else:
        logger.success("✅  All records validated successfully.")

if __name__ == "__main__":
    validate_fatura()
