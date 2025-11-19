import pandas as pd
from pathlib import Path
from loguru import logger

# Cleaned structured file
INPUT_FILE = Path("/opt/airflow/data/processed/fatura_cleaned.csv")
OUTPUT_FILE = Path("/opt/airflow/reports/schema_check.txt")

def check_schema():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        msg = f"❌ File not found: {INPUT_FILE}"
        OUTPUT_FILE.write_text(msg)
        print(msg)
        return

    df = pd.read_csv(INPUT_FILE)

    # ✔ Expected FINAL structured schema
    expected_columns = [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "currency",
        "total_amount",
    ]

    results = []

    # Check required columns
    for col in expected_columns:
        if col in df.columns:
            results.append(f"✔ Column present: {col}")
        else:
            results.append(f"❌ MISSING column: {col}")

    # Row count
    results.append(f"Total rows: {len(df)}")

    # Save report
    OUTPUT_FILE.write_text("\n".join(results))

    logger.info(f"Schema check complete → {OUTPUT_FILE}")
    print("Schema check complete.")

if __name__ == "__main__":
    check_schema()
