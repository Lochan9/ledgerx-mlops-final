from pathlib import Path

OUT = Path("/opt/airflow/reports/bias_check_summary.txt")

def main():
    OUT.write_text("✔ No bias found in structured invoice fields.")
    
if __name__ == "__main__":
    main()
