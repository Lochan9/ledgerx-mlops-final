"""
===============================================================
 LEDGERX - ENTERPRISE INVOICE PREPROCESSING ENGINE (v3.1)
===============================================================

This module simulates a REAL enterprise-grade invoice preprocessing
pipeline including:

 • OCR distortions (motion blur, perspective warp, JPEG compression)
 • Image corruption (brightness shifts, occlusions)
 • Fraud-like text modifications (amount tampering, vendor spoofing)
 • Business-rule validation (total mismatch, currency issues, date checks)
 • Multi-page simulation
 • Quality scoring (weighted)
 • Failure labeling (accounting risk)

Designed to represent:
  - A 10-year ML Engineer’s preprocessing
  - A 15-year Finance/Business Leader’s rules

===============================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# IMAGE AUGMENTATIONS — ENTERPRISE-LEVEL OCR HARD CASES
# ================================================================

def motion_blur(img):
    size = random.choice([5, 9, 13])
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)


def jpeg_artifacts(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(10, 35)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


def perspective_warp(img):
    h, w = img.shape[:2]
    delta = random.randint(10, 50)
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [random.randint(0,delta), random.randint(0,delta)],
        [w-random.randint(0,delta), random.randint(0,delta)],
        [random.randint(0,delta), h-random.randint(0,delta)],
        [w-random.randint(0,delta), h-random.randint(0,delta)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))


def brightness_shift(img):
    factor = random.uniform(0.4, 1.6)
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    return np.array(enhancer.enhance(factor))


def random_occlusion(img):
    h, w = img.shape[:2]
    x = random.randint(0, w//2)
    y = random.randint(0, h//2)
    w_occ = random.randint(50, 200)
    h_occ = random.randint(30, 100)
    img[y:y+h_occ, x:x+w_occ] = 255
    return img


def apply_ocr_hardcases(img):
    funcs = [motion_blur, jpeg_artifacts, perspective_warp,
             brightness_shift, random_occlusion]
    count = random.randint(1, 3)
    for f in random.sample(funcs, count):
        img = f(img)
    return img


# ================================================================
# TEXT / DATA CORRUPTION — FRAUD & OCR-LIKE MODIFICATIONS
# ================================================================

def corrupt_amount(val):
    """Simulate amount fraud patterns with safe fallback."""
    try:
        v = float(val)
    except:
        return val  # if not numeric, skip

    options = []

    # Inflate / deflate / tamper
    options.append(v * random.uniform(1.2, 2.5))
    options.append(v * random.uniform(0.1, 0.5))
    options.append(v + random.randint(10, 200))

    # Digit drop corruption — SAFELY handled
    dropped = str(v).replace("0", "")
    try:
        dropped_float = float(dropped)
        options.append(dropped_float)
    except:
        pass  # ignore invalid corruption

    out = random.choice(options)

    try:
        return round(float(out), 2)
    except:
        return v


def corrupt_vendor_name(v):
    if not isinstance(v, str):
        return v
    if len(v) < 5:
        return v

    corruptions = [
        v.replace("o", "0").replace("l", "1").replace("a", "4"),
        v[:len(v)//2],
        v + random.choice([" LLC", " LTD", " & Sons", " INC"]),
        "".join(sorted(v)),
    ]
    return random.choice(corruptions)


def corrupt_invoice_id(inv):
    if not isinstance(inv, str):
        return inv

    corruptions = [
        inv[:-2],
        inv + str(random.randint(0,99)),
        inv.replace("1","I").replace("0","O"),
        "",
    ]
    return random.choice(corruptions)


def corrupt_currency(cur):
    options = ["USD", "TRY", "EUR", "GBP", "", "US$", "₺", "$$"]
    return random.choice(options)


def simulate_fraud_row(row):
    """Applies fraud-like corruption patterns."""
    row = row.copy()

    if random.random() < 0.50:
        row["total_amount"] = corrupt_amount(row.get("total_amount"))

    if random.random() < 0.40:
        row["vendor_name"] = corrupt_vendor_name(row.get("vendor_name"))

    if random.random() < 0.40:
        row["invoice_number"] = corrupt_invoice_id(row.get("invoice_number"))

    if random.random() < 0.25:
        row["currency"] = corrupt_currency(row.get("currency"))

    return row


# ================================================================
# BUSINESS RULE VALIDATION — REAL ACCOUNTING LOGIC
# ================================================================

def compute_business_failures(row):
    failures = 0

    # 1) subtotal + tax != total
    try:
        subtotal = float(row.get("subtotal", 0))
        tax = float(row.get("tax", 0))
        total = float(row.get("total_amount", 0))
        if abs((subtotal + tax) - total) > 1e-3:
            failures += 1
    except:
        failures += 1

    # 2) suspicious / missing currency
    if row.get("currency", "") not in ["USD", "EUR", "TRY", "GBP"]:
        failures += 1

    # 3) invalid / missing date
    try:
        date = pd.to_datetime(row["invoice_date"], errors="coerce")
        if date is pd.NaT:
            failures += 1
    except:
        failures += 1

    # 4) vendor anomaly
    if len(str(row.get("vendor_name", ""))) < 3:
        failures += 1

    return 1 if failures > 0 else 0


# ================================================================
# QUALITY SCORING — ENTERPRISE WEIGHTED FORMULA
# ================================================================

def compute_quality_score(row):
    normalized_blur = min(row["blur_score"] / 100.0, 1)
    normalized_ocr = row["ocr_confidence"]
    contrast = min(row["contrast_score"] / 50.0, 1)

    score = (
        0.40 * normalized_blur +
        0.40 * normalized_ocr +
        0.10 * contrast +
        0.10 * (1 - (row["num_pages_fake"] - 1) * 0.20)
    )
    return score


def compute_label_quality(score):
    return 1 if score < 0.45 else 0


# ================================================================
# MAIN PREPROCESSING ENGINE
# ================================================================

def preprocess_enterprise(df, image_root):
    processed = []

    for _, row in df.iterrows():
        r = row.copy()

        img_path = image_root / r["file_name"]

        # -----------------------------------------------
        # LOAD + ENTERPRISE OCR CORRUPTION
        # -----------------------------------------------
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if random.random() < 0.40:
                img = apply_ocr_hardcases(img)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            r["blur_score"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            r["contrast_score"] = gray.std()
            r["ocr_confidence"] = random.uniform(0.4, 0.95)

        except:
            r["blur_score"] = 0.0
            r["contrast_score"] = 0.0
            r["ocr_confidence"] = 0.0

        # -----------------------------------------------
        # MULTI-PAGE SIMULATION
        # -----------------------------------------------
        r["num_pages_fake"] = random.choice([1,1,1,2,3])

        # -----------------------------------------------
        # FRAUD ROW SIMULATION (20%)
        # -----------------------------------------------
        if random.random() < 0.20:
            r = simulate_fraud_row(r)

        # -----------------------------------------------
        # BUSINESS FAILURE LABEL
        # -----------------------------------------------
        r["label_failure"] = compute_business_failures(r)

        # -----------------------------------------------
        # QUALITY SCORING (WEIGHTED)
        # -----------------------------------------------
        qscore = compute_quality_score(r)
        r["quality_score"] = qscore
        r["label_quality_bad"] = compute_label_quality(qscore)

        processed.append(r)

    return pd.DataFrame(processed)
