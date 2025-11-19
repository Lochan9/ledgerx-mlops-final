import os
from pathlib import Path
import pandas as pd

def resolve_path(local, airflow):
    """
    Automatically choose between local Windows path and Airflow Linux path.
    """
    # If running in Airflow environment
    if Path(airflow).exists():
        return Path(airflow)
    # Otherwise use local repo path
    if Path(local).exists():
        return Path(local)
    raise FileNotFoundError(f"Cannot locate data. Tried:\n  {local}\n  {airflow}")

def generate_report():
    print("📄 Generating summary report...")

    LOCAL_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"
    AIRFLOW_DATA_ROOT = Path("/opt/airflow/data/processed")

    # *** Auto-detect correct data root ***
    DATA_ROOT = resolve_path(LOCAL_DATA_ROOT, AIRFLOW_DATA_ROOT)

    INPUT_OCR = DATA_ROOT / "fatura_ocr.csv"
    INPUT_STRUCT = DATA_ROOT / "fatura_structured.csv"
    QUALITY_CSV = DATA_ROOT / "quality_training.csv"
    FAILURE_CSV = DATA_ROOT / "failure_training.csv"

    # Validate input files
    for file in [INPUT_OCR, INPUT_STRUCT, QUALITY_CSV, FAILURE_CSV]:
        if not file.exists():
            raise FileNotFoundError(f"❌ Missing required file: {file}")

    df_quality = pd.read_csv(QUALITY_CSV)
    df_failure = pd.read_csv(FAILURE_CSV)

    # Generate quick metrics
    report_text = []
    report_text.append("LedgerX - Summary Report\n==============================\n")
    report_text.append(f"Total rows (quality): {len(df_quality)}\n")
    report_text.append(f"Total rows (failure): {len(df_failure)}\n")

    report_text.append("\nQuality Label Distribution:\n")
    report_text.append(str(df_quality['label_quality_bad'].value_counts()) + "\n")

    report_text.append("\nFailure Label Distribution:\n")
    report_text.append(str(df_failure['label_failure'].value_counts()) + "\n")

    # Save
    OUTPUT_FILE = Path(__file__).resolve().parents[2] / "reports" / "summary_report.txt"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    print(f"✅ Summary report saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
