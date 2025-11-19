import shutil
from pathlib import Path
from loguru import logger

def main():
    logger.info("📥 Starting Fatura data acquisition...")

    source_path = Path("/opt/airflow/data/source/fatura_ocr.csv")
    dest_dir = Path("/opt/airflow/data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / "fatura_ocr.csv"

    if not source_path.exists():
        logger.error(f"❌ Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")

    shutil.copy(source_path, dest_path)
    logger.info(f"✅ Copied OCR file → {dest_path}")

if __name__ == "__main__":
    main()
