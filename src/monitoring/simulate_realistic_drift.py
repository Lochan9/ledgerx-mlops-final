"""
Simulate Realistic Data Drift for Demo
Creates production data with realistic distribution shifts
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load training data
train = pd.read_csv("data/processed/quality_training.csv")

# Simulate 6 months later - realistic changes:
# 1. Better scanners → improved blur_score
# 2. OCR upgrades → higher ocr_confidence  
# 3. Different vendors → changed amounts

prod_simulated = train.sample(n=500, random_state=42).copy()

# Simulate realistic drift (6% improvement in image quality)
if 'blur_score' in prod_simulated.columns:
    prod_simulated['blur_score'] = prod_simulated['blur_score'] * 0.94  # 6% better (less blur)

if 'ocr_confidence' in prod_simulated.columns:
    prod_simulated['ocr_confidence'] = np.clip(prod_simulated['ocr_confidence'] * 1.08, 0, 1)  # 8% better

# Save
prod_simulated.to_csv("data/production/simulated_drift.csv", index=False)

print("✓ Simulated production data with realistic drift")
print("\nExpected changes:")
print("  • Blur score: 6% improvement (better scanners)")
print("  • OCR confidence: 8% improvement (software upgrades)")
print("  • This represents 6 months of system improvements")

# Test drift
from src.monitoring.drift_threshold_checker import DriftThresholdChecker

checker = DriftThresholdChecker(
    reference_data_path="data/processed/quality_training.csv",
    production_data_path="data/production/simulated_drift.csv"
)

result = checker.detect_drift()

print("\n" + "="*70)
print("DRIFT DETECTION RESULT:")
print("="*70)
import json
print(json.dumps(result, indent=2))