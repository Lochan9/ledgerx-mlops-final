import pandas as pd
from pathlib import Path
from loguru import logger

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
        "",
    ]

    # ------------------------------------------------------
    # 🚨 ANOMALY DETECTION (REQUIRED FOR IE7305 RUBRIC)
    # Rule: If any critical field has >20% missing → ALERT
    # ------------------------------------------------------
    critical_cols = ["invoice_number", "invoice_date", "total_amount", "vendor_name", "currency"]

    # Missing ratio for each critical column
    missing_ratios = df[critical_cols].isna().mean()

    anomalies = []
    for col, ratio in missing_ratios.items():
        if ratio > 0.20:
            anomalies.append(f"⚠️ ALERT: Column '{col}' has {ratio*100:.1f}% missing values")

    if anomalies:
        summary.append("=== ⚠️ Anomaly Detection ===")
        summary.extend(anomalies)
    else:
        summary.append("No anomalies detected.")

    # ------------------------------------------------------

    OUT.write_text("\n".join(summary))
    logger.info(f"✅ Summary report generated → {OUT}")

if __name__ == "__main__":
    main()
