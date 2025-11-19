from loguru import logger
import great_expectations as ge
from pathlib import Path

# ABSOLUTE PATH for Airflow container
DATA_PATH = Path("/opt/airflow/data/processed/fatura_ocr.csv")

def main():
    if not DATA_PATH.exists():
        logger.error(f"❌ Missing file: {DATA_PATH}")
        return

    df = ge.read_csv(str(DATA_PATH))
    logger.info(f"📦 Loaded {len(df)} rows for validation")

    results = {
        "columns": df.expect_table_columns_to_match_ordered_list(["file_name", "ocr_text"]),
        "nulls": df.expect_column_values_to_not_be_null("ocr_text"),
        "types": df.expect_column_values_to_be_of_type("file_name", "str"),
    }

    failed = [k for k, r in results.items() if not r.success]
    if failed:
        logger.error(f"❌ Schema checks failed: {failed}")
        exit(1)
    else:
        logger.success("✅ All schema checks passed")

if __name__ == "__main__":
    main()
