import os
import platform
import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tqdm import tqdm  # for progress bar

# ✅ Detect OS and set correct Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# --- Paths ---
RAW_DIR = Path("data/raw/FATURA")
OUT_FILE = Path("data/processed/fatura_ocr.csv")
CACHE_FILE = Path("data/processed/fatura_ocr_cache.csv")

# ⚙️ Tesseract config for performance
TESSERACT_CONFIG = "--psm 6 --oem 3 -l eng"  # PSM=6 uniform block; OEM=3 LSTM engine

# 🧩 Cache helpers --------------------------------------------------------------
def load_cache():
    """Load previously saved OCR cache to avoid reprocessing."""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE)
        return {row["file_name"]: row["ocr_text"] for _, row in df.iterrows()}
    return {}

def save_cache(data):
    """Save updated cache dictionary to disk."""
    pd.DataFrame(data.items(), columns=["file_name", "ocr_text"]).to_csv(CACHE_FILE, index=False)

def compute_md5(file_path):
    """Compute hash of file to detect changes."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# 🧠 OCR function ---------------------------------------------------------------
def ocr_single_image(img_path: Path):
    """Perform OCR on a single image."""
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        return img_path.name, text
    except Exception as e:
        logger.error(f"OCR failed for {img_path.name}: {e}")
        return img_path.name, ""

# 🚀 Main OCR extraction pipeline ---------------------------------------------
def extract_ocr_from_images():
    logger.info(f"🔍 Scanning {RAW_DIR}")

    if not RAW_DIR.exists():
        logger.error(f"❌ Directory {RAW_DIR} not found.")
        return

    img_files = list(RAW_DIR.rglob("*.jpg"))
    if not img_files:
        logger.warning("⚠️ No image files found.")
        return

    cache = load_cache()
    results = {}

    logger.info(f"🧠 Found {len(img_files)} images. Starting OCR with {os.cpu_count() or 4} threads...")

    # Multithreaded OCR with progress bar
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(ocr_single_image, img): img for img in img_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔠 OCR Progress", unit="img"):
            img_name, text = future.result()
            results[img_name] = text

    # Merge with existing cache
    cache.update(results)
    save_cache(cache)

    # Save final results
    df = pd.DataFrame(cache.items(), columns=["file_name", "ocr_text"])
    df.to_csv(OUT_FILE, index=False)
    logger.success(f"🚀 OCR completed and saved to {OUT_FILE}")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    extract_ocr_from_images()
