"""
Data Acquisition Stage for LedgerX FATURA
-----------------------------------------
Simulates pulling raw invoice images or data from a local directory
or remote storage. Creates data/raw/FATURA if not exists.
"""

import os
from loguru import logger

RAW_PATH = "/opt/airflow/data/raw/FATURA"

def main():
    os.makedirs(RAW_PATH, exist_ok=True)
    logger.info(f"✅ FATURA raw data directory ready at: {RAW_PATH}")
    # Simulate pulling data (you can replace this with actual fetch code)
    dummy_file = os.path.join(RAW_PATH, "sample_placeholder.txt")
    with open(dummy_file, "w") as f:
        f.write("This simulates data acquisition.")
    logger.info("✅ Data acquisition stage complete.")

if __name__ == "__main__":
    main()
