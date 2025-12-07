"""
Test Script: Simulate Real Data Drift
======================================
This script creates ACTUAL drifted data to verify drift detection works.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("Testing Drift Detection with SIMULATED DRIFT")
print("="*70)

# Load training baseline
baseline_path = Path("data/processed/fatura_enterprise_preprocessed.csv")
if not baseline_path.exists():
    print(f"❌ Baseline data not found at {baseline_path}")
    print("Please run from project root: D:\\vsCOde\\ledgerx-mlops-final")
    exit(1)

baseline = pd.read_csv(baseline_path)
print(f"\n✅ Loaded baseline: {len(baseline)} rows, {len(baseline.columns)} columns")

# Create DRIFTED production data
print("\n🔄 Simulating data drift...")
production = baseline.sample(500, random_state=42).copy()

# Apply realistic drift scenarios
print("\nApplying drift:")

# 1. Blur score increased (worse quality scans)
if 'blur_score' in production.columns:
    production['blur_score'] = production['blur_score'] * 1.5 + 20
    print("  ✓ Increased blur_score by 50% + 20 (worse image quality)")

# 2. OCR confidence decreased (worse OCR performance)
if 'ocr_confidence' in production.columns:
    production['ocr_confidence'] = production['ocr_confidence'] * 0.7
    print("  ✓ Decreased ocr_confidence by 30% (worse OCR)")

# 3. Total amount shifted (different invoice amounts)
if 'total_amount' in production.columns:
    production['total_amount'] = production['total_amount'] * 2.5 + 500
    print("  ✓ Increased total_amount by 150% + 500 (inflation/different clients)")

# 4. Resolution changed
if 'image_width' in production.columns:
    production['image_width'] = production['image_width'] * 0.8
    print("  ✓ Decreased image_width by 20% (lower resolution images)")

# 5. More missing values
missing_cols = ['vendor_name', 'line_items']
for col in missing_cols:
    if col in production.columns:
        mask = np.random.random(len(production)) < 0.3
        production.loc[mask, col] = np.nan
        print(f"  ✓ Added 30% missing values to {col}")

# Save drifted production data
output_dir = Path("data/production")
output_dir.mkdir(exist_ok=True, parents=True)
output_path = output_dir / "recent_features.csv"
production.to_csv(output_path, index=False)
print(f"\n✅ Saved drifted data to: {output_path}")

# Now run drift detection
print("\n" + "="*70)
print("Running Drift Detection on DRIFTED Data")
print("="*70)

import sys
sys.path.insert(0, str(Path.cwd() / "src"))

from monitoring.drift_threshold_checker import DriftThresholdChecker

checker = DriftThresholdChecker()
result = checker.detect_drift()

print("\n📊 DRIFT DETECTION RESULT:")
print(json.dumps(result, indent=2))

if result['should_retrain']:
    print("\n🚨 SUCCESS! Drift was DETECTED as expected!")
    print(f"   - Drift Score: {result['drift_score']:.1%}")
    print(f"   - Drifted Features: {result['num_drifted_features']}/{result['total_features']}")
    print(f"   - Top Drifted Features: {result['drifted_features'][:5]}")
    print("\n✅ The drift detection system is working correctly!")
else:
    print("\n⚠️ Drift not detected - threshold might need adjustment")
    print(f"   Current drift score: {result['drift_score']:.1%}")
    print(f"   Threshold: 15%")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
