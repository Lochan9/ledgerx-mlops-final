import pandas as pd
from pathlib import Path
from loguru import logger
import re

RAW_FILE = Path("/opt/airflow/data/processed/fatura_ocr.csv")
OUT_FILE = Path("/opt/airflow/data/processed/fatura_structured.csv")

def extract_invoice_number(text):
    match = re.search(r"(INV\w+|\d{6,})", str(text))
    return match.group(0) if match else None

def extract_invoice_date(text):
    # Normalize formats → YYYY-MM-DD
    match = re.search(r"(\d{2})[-/](\w{3})[-/](\d{4})", str(text))
    if match:
        day, mon, year = match.groups()
        try:
            return pd.to_datetime(f"{day}-{mon}-{year}").strftime("%Y-%m-%d")
        except Exception:
            return None
    return None

def extract_total_amount(text):
    match = re.search(r"(\d+[\.,]\d{2})", str(text))
    return float(match.group(1).replace(",", ".")) if match else None

def extract_vendor(text):
    lines = str(text).splitlines()
    return lines[0] if lines else None

def extract_currency(text):
    if "USD" in text or "$" in text:
        return "USD"
    if "EUR" in text or "€" in text:
        return "EUR"
    return "UNK"

def main():
    logger.info("🔧 Transforming OCR → structured schema...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw OCR file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    df_struct = pd.DataFrame()
    df_struct["invoice_number"] = df["ocr_text"].apply(extract_invoice_number)
    df_struct["invoice_date"] = df["ocr_text"].apply(extract_invoice_date)
    df_struct["total_amount"] = df["ocr_text"].apply(extract_total_amount)
    df_struct["vendor_name"] = df["ocr_text"].apply(extract_vendor)
    df_struct["currency"] = df["ocr_text"].apply(extract_currency)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_struct.to_csv(OUT_FILE, index=False)

    logger.info(f"✅ Structured schema saved → {OUT_FILE}")

if __name__ == "__main__":
    main()
