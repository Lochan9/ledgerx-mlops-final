"""
Evidently AI - Data Drift Detection for LedgerX
Requirement: PDF Section 2 - Use Evidently AI for drift detection
Compares production data vs training baseline
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *

# Paths
DATA_DIR = Path("data/processed")
PRODUCTION_DIR = Path("data/production")
REPORTS_DIR = Path("reports/evidently")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("Evidently AI - Data Drift Detection")
print("=" * 70)

# Load reference data (training baseline)
print("\n1. Loading reference data (training baseline)...")
reference_data = pd.read_csv(DATA_DIR / "quality_training.csv")
reference_features = reference_data.drop(columns=['label_quality_bad', 'file_name'], errors='ignore')
print(f"✓ Reference data: {len(reference_features)} samples, {len(reference_features.columns)} features")

# Load production data (recent predictions)
print("\n2. Loading production data...")
# Simulate production data by sampling from test set
production_data = pd.read_csv(DATA_DIR / "quality_test.csv").sample(n=500, random_state=42)
production_features = production_data.drop(columns=['label_quality_bad', 'file_name'], errors='ignore')
print(f"✓ Production data: {len(production_features)} samples")

# Align columns (production may have different features)
common_cols = list(set(reference_features.columns) & set(production_features.columns))
reference_aligned = reference_features[common_cols]
production_aligned = production_features[common_cols]
print(f"✓ Aligned on {len(common_cols)} common features")

# Create Evidently Report
print("\n3. Generating Evidently AI drift report...")
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ColumnDriftMetric(column_name='blur_score'),
    ColumnDriftMetric(column_name='ocr_confidence'),
    ColumnDriftMetric(column_name='total_amount', stattest='ks'),
])

report.run(
    reference_data=reference_aligned.head(1000),  # Baseline
    current_data=production_aligned.head(500)      # Production
)

# Save HTML report
html_path = REPORTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
report.save_html(str(html_path))
print(f"✓ HTML report saved: {html_path}")

# Save JSON report
json_path = REPORTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
report.save_json(str(json_path))
print(f"✓ JSON report saved: {json_path}")

# Extract drift metrics
drift_results = report.as_dict()

# Parse drift summary
dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
n_drifted_features = drift_results['metrics'][0]['result']['number_of_drifted_columns']
drift_share = drift_results['metrics'][0]['result']['share_of_drifted_columns']

print("\n" + "=" * 70)
print("DRIFT DETECTION SUMMARY")
print("=" * 70)
print(f"Dataset Drift Detected: {dataset_drift}")
print(f"Number of Drifted Features: {n_drifted_features}")
print(f"Share of Drifted Features: {drift_share:.2%}")
print(f"\nThreshold: 50% of features")
print(f"Status: {'⚠️ DRIFT DETECTED - Retrain needed!' if dataset_drift else '✅ NO DRIFT - Model is stable'}")
print("=" * 70)

# Retraining decision
DRIFT_THRESHOLD = 0.5  # 50% of features

if drift_share > DRIFT_THRESHOLD or dataset_drift:
    print(f"\n🔄 RETRAINING TRIGGERED!")
    print(f"   Reason: {drift_share:.1%} of features show drift (threshold: {DRIFT_THRESHOLD:.1%})")
    print(f"   Next steps:")
    print(f"   1. Pull latest data")
    print(f"   2. Run training pipeline")
    print(f"   3. Validate new model")
    print(f"   4. Deploy if F1 > current model")
    
    # Log retraining event
    retraining_log = REPORTS_DIR / "retraining_events.json"
    import json
    events = []
    if retraining_log.exists():
        with open(retraining_log) as f:
            events = json.load(f)
    
    events.append({
        'timestamp': datetime.now().isoformat(),
        'trigger': 'evidently_drift_detection',
        'drift_share': float(drift_share),
        'drifted_features': n_drifted_features,
        'status': 'triggered'
    })
    
    with open(retraining_log, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"\n✓ Retraining event logged to: {retraining_log}")

else:
    print(f"\n✅ Model is stable - no retraining needed")
    print(f"   Drift: {drift_share:.1%} (threshold: {DRIFT_THRESHOLD:.1%})")

print(f"\n📊 View detailed report: {html_path}")
print(f"   Open in browser to see feature-by-feature drift analysis")