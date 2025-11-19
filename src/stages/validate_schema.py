import pandas as pd
from pathlib import Path
from loguru import logger
import sys

INPUT_FILE = Path("/opt/airflow/data/processed/fatura_cleaned.csv")
REPORT_FILE = Path("/opt/airflow/reports/schema_check.txt")

REQUIRED_COLS = [
    "invoice_number",
    "invoice_date",
    "total_amount",
    "vendor_name",
    "currency",
]

def main():
    logger.info("🔍 Validating strict invoice schema...")

    if not INPUT_FILE.exists():
        msg = f"❌ File not found: {INPUT_FILE}"
        REPORT_FILE.write_text(msg)
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    # Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        msg = f"❌ Missing columns: {missing}"
        REPORT_FILE.write_text(msg)
        sys.exit(1)

    # Date format strict yyyy-mm-dd
    try:
        pd.to_datetime(df["invoice_date"], format="%Y-%m-%d", errors="raise")
    except:
        msg = "❌ invoice_date has invalid format (expected YYYY-MM-DD)"
        REPORT_FILE.write_text(msg)
        sys.exit(1)

    if not pd.api.types.is_numeric_dtype(df["total_amount"]):
        msg = "❌ total_amount is not numeric"
        REPORT_FILE.write_text(msg)
        sys.exit(1)

    success = "✔ Schema validation passed (strict mode)"
    REPORT_FILE.write_text(success)
    logger.info(success)
    print(success)

if __name__ == "__main__":
    main()
