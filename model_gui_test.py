import joblib
import cv2
import pytesseract
import numpy as np
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog
import re
import os
import json

# -----------------------------------------------------
# TESSERACT PATH FIX (Windows)
# -----------------------------------------------------
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------------------------------
# CONFIG – Failure prediction model
# -----------------------------------------------------
MODEL_PATH = "models/failure_model.pkl"


# -----------------------------------------------------
# SAVE RAW OCR TEXT
# -----------------------------------------------------
def save_ocr_text(image_path: str, ocr_text: str):
    out_dir = Path("extracted_text")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / (Path(image_path).stem + ".txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    print(f"[INFO] OCR text saved to: {save_path}")


# -----------------------------------------------------
# STRUCTURED JSON SAVE
# -----------------------------------------------------
def save_structured_json(image_path: str, structured_data: dict):
    out_dir = Path("structured_output")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / (Path(image_path).stem + ".json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4)

    print(f"[INFO] Structured JSON saved to: {save_path}")


# -----------------------------------------------------
# SAFE OCR PREPROCESSING
# -----------------------------------------------------
def enhance_image_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance > 3000:
        return gray  # Clean digital invoices → preserve clarity

    # Light enhancement only for noisy scans
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, light_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return light_thresh


# -----------------------------------------------------
# LAST TOTAL EXTRACTION
# -----------------------------------------------------
def extract_total_amount(ocr_text: str) -> float:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    last_amount = None

    inline_pat = re.compile(r"total[\s:$]*([0-9]+[.,][0-9]+)", re.IGNORECASE)

    for line in lines:
        m = inline_pat.search(line)
        if m:
            try:
                last_amount = float(m.group(1).replace(",", ""))
            except:
                pass

    for i, line in enumerate(lines):
        if re.match(r"^total$", line) and i + 1 < len(lines):
            m2 = re.search(r"([0-9]+[.,][0-9]+)", lines[i + 1])
            if m2:
                try:
                    last_amount = float(m2.group(1).replace(",", ""))
                except:
                    pass

    return last_amount if last_amount is not None else 0.0


# -----------------------------------------------------
# IMPROVED VENDOR NAME EXTRACTION
# -----------------------------------------------------
def extract_vendor_name(ocr_text: str) -> str:
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]

    address_noise = [
        "street", "road", "rd", "avenue", "city",
        "state", "zip", "email", "@", ".com", ".net",
        ".org", "phone", "tel", "www", "http"
    ]

    candidates = []

    for line in lines[:10]:
        lower = line.lower()
        if any(w in lower for w in address_noise):
            continue
        if any(c.isdigit() for c in line):
            continue

        uppercase_ratio = sum(1 for c in line if c.isupper()) / len(line)
        if uppercase_ratio > 0.4:
            candidates.append(line)

    if candidates:
        return candidates[0]

    return lines[0] if lines else ""


# -----------------------------------------------------
# IMPROVED INVOICE NUMBER EXTRACTION
# -----------------------------------------------------
def extract_invoice_number(ocr_text: str) -> str:
    patterns = [
        r"invoice[^0-9a-z]+([a-z0-9\-\/]+)",
        r"invoice no[:\s]*([a-z0-9\-\/]+)",
        r"invoice #[:\s]*([a-z0-9\-\/]+)",
        r"inv no[:\s]*([a-z0-9\-\/]+)",
        r"inv #[:\s]*([a-z0-9\-\/]+)",
        r"bill no[:\s]*([a-z0-9\-\/]+)"
    ]

    for pat in patterns:
        m = re.search(pat, ocr_text, re.IGNORECASE)
        if m:
            return m.group(1)

    # fallback → number near "invoice"
    for line in ocr_text.splitlines():
        if "invoice" in line:
            tokens = re.findall(r"[a-z0-9\-]+", line)
            if len(tokens) > 1:
                return tokens[1]

    return ""


# -----------------------------------------------------
# STRUCTURED FIELD EXTRACTION (JSON)
# -----------------------------------------------------
def extract_structured_fields(ocr_text: str) -> dict:
    data = {}
    text = ocr_text.lower()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    data["vendor_name"] = extract_vendor_name(text)
    data["invoice_number"] = extract_invoice_number(text)

    # Invoice date
    date_patterns = [
        r"invoice date[^0-9]*([0-9]{1,2}[-/][a-z]{3}[-/][0-9]{2,4})",
        r"date[:\s]*([0-9]{1,2}[-/][a-z]{3}[-/][0-9]{2,4})",
        r"invoice date[:\s]*([0-9/.\-]+)"
    ]
    data["invoice_date"] = ""

    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            data["invoice_date"] = m.group(1)
            break

    # Due Date
    due_patterns = [
        r"due date[:\s]*([0-9/.\-a-z]+)",
        r"payment due[:\s]*([0-9/.\-a-z]+)"
    ]
    data["due_date"] = ""
    for pat in due_patterns:
        m = re.search(pat, text)
        if m:
            data["due_date"] = m.group(1)
            break

    # Subtotal
    sub = re.search(r"sub[_\s\-]*total[^0-9]*([0-9.,]+)", text)
    data["subtotal"] = float(sub.group(1).replace(",", "")) if sub else None

    # Tax
    tax = re.search(r"tax[^0-9]*([0-9.,]+)", text)
    data["tax"] = float(tax.group(1).replace(",", "")) if tax else None

    # Total
    data["total"] = extract_total_amount(text)

    # Currency
    if "eur" in text:
        data["currency"] = "EUR"
    elif "$" in text:
        data["currency"] = "USD"
    else:
        data["currency"] = "UNKNOWN"

    # Bill To block
    billto = []
    capturing = False
    for line in lines:
        if "bill to" in line:
            capturing = True
            continue
        if capturing:
            if any(x in line for x in ["ship", "gst", "tax", "invoice"]):
                break
            billto.append(line)

    data["bill_to"] = " ".join(billto)

    # Items — simple extractor
    data["items"] = []
    capture_items = False

    for line in lines:
        if "qty" in line or "quantity" in line:
            capture_items = True
            continue

        if capture_items:
            m = re.findall(r"([a-z\s]+)\s([0-9.]+)\s*\$?([0-9.]+)", line)
            if m:
                for desc, qty, price in m:
                    data["items"].append({
                        "description": desc.strip(),
                        "qty": float(qty),
                        "price": float(price)
                    })

    data["full_text"] = text

    return data


# -----------------------------------------------------
# NUMERICAL BUCKETING
# -----------------------------------------------------
def bucket_amount(total_amount: float) -> int:
    if total_amount <= 0:
        return 0
    if total_amount < 1000:
        return 1
    if total_amount < 10000:
        return 2
    return 3


# -----------------------------------------------------
# FEATURE EXTRACTION PIPELINE
# -----------------------------------------------------
def extract_features(image_path: str) -> pd.DataFrame | None:
    print(f"[INFO] Reading image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Cannot load image.")
        return None

    enhanced = enhance_image_for_ocr(img)

    try:
        data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if c != -1]
        ocr_confidence = float(np.mean(confs)) if confs else 0.0

        ocr_text = pytesseract.image_to_string(enhanced)
        ocr_text = ocr_text.lower()

        print("\n===== OCR DEBUG TEXT =====")
        print(ocr_text)
        print("==========================\n")

        save_ocr_text(image_path, ocr_text)

        structured = extract_structured_fields(ocr_text)
        save_structured_json(image_path, structured)

    except:
        ocr_confidence = 0.0
        ocr_text = ""

    total_amount = extract_total_amount(ocr_text)
    amount_bucket = bucket_amount(total_amount)

    vendor_name = extract_vendor_name(ocr_text)
    vendor_name_length = len(vendor_name)

    invoice_num = extract_invoice_number(ocr_text)
    invoice_number_present = 1 if invoice_num else 0

    file_size_kb = Path(image_path).stat().st_size / 1024

    features = pd.DataFrame([{
        "blur_score": float(cv2.Laplacian(enhanced, cv2.CV_64F).var()),
        "contrast_score": float(enhanced.std()),
        "ocr_confidence": ocr_confidence,
        "num_missing_fields": 0,
        "has_critical_missing": 0,
        "vendor_freq": 1,
        "file_size_kb": file_size_kb,
        "num_pages": 1,
        "total_amount": total_amount,
        "amount_bucket": amount_bucket,
        "vendor_name_length": vendor_name_length,
        "invoice_number_present": invoice_number_present,
    }])

    print("\n[DEBUG] Features passed to model:")
    print(features.to_string(index=False))

    return features


# -----------------------------------------------------
# MODEL PREDICTION
# -----------------------------------------------------
def predict_image(image_path: str) -> None:
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    features = extract_features(image_path)
    if features is None:
        return

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    print("\n========== LEDGERX FAILURE MODEL RESULT ==========")
    print(f"Prediction : {pred}  (0 = OK, 1 = FAIL)")
    print(f"Confidence : {prob:.4f}")
    print("==================================================")


# -----------------------------------------------------
# MAIN – File Picker
# -----------------------------------------------------
if __name__ == "__main__":
    Tk().withdraw()
    print("[INFO] Select an invoice image...")

    file_path = filedialog.askopenfilename(
        title="Select Invoice",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("PDF Files", "*.pdf")]
    )

    if file_path:
        print(f"[INFO] Selected: {file_path}")
        predict_image(file_path)
    else:
        print("[INFO] No file selected.")
