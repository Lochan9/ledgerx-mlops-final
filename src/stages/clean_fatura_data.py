import pandas as pd
from pathlib import Path
from loguru import logger

IN_FILE = Path("/opt/airflow/data/processed/fatura_structured.csv")
OUT_FILE = Path("/opt/airflow/data/processed/fatura_cleaned.csv")

def main():
    logger.info("🧹 Starting cleaning process...")

    if not IN_FILE.exists():
        raise FileNotFoundError(f"Structured file not found: {IN_FILE}")

    df = pd.read_csv(IN_FILE)

    # Fill missing vendor
    df["vendor_name"] = df["vendor_name"].fillna("UNKNOWN_VENDOR")

    # Fill missing currency
    df["currency"] = df["currency"].fillna("UNK")

    # Remove malformed dates
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["invoice_date"] = df["invoice_date"].fillna(pd.Timestamp("2000-01-01"))
    df["invoice_date"] = df["invoice_date"].dt.strftime("%Y-%m-%d")

    # Amount must be numeric
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)

    df.to_csv(OUT_FILE, index=False)
    logger.info(f"✅ Cleaned file saved → {OUT_FILE}")

if __name__ == "__main__":
    main()
