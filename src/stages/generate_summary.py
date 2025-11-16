from pathlib import Path
import pandas as pd

FILE = Path("/opt/airflow/data/processed/fatura_cleaned.csv")
OUT = Path("/opt/airflow/reports/summary_report.txt")

def main():
    if not FILE.exists():
        OUT.write_text("❌ No cleaned file found.")
        return

    df = pd.read_csv(FILE)

    summary = [
        "=== LedgerX Fatura Summary Report ===",
        f"Rows: {len(df)}",
        f"Vendors: {df['vendor_name'].nunique()}",
        f"Total Amount Sum: {df['total_amount'].sum():.2f}",
        "=====================================",
    ]

    OUT.write_text("\n".join(summary))

if __name__ == "__main__":
    main()
