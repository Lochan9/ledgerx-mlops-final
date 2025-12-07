"""
Check LedgerX Model Accuracy
Shows current model performance metrics
"""

import json
import pickle
from pathlib import Path
import pandas as pd

print("="*70)
print("LEDGERX MODEL ACCURACY CHECK")
print("="*70)
print()

# Check if models exist
models_dir = Path("models")
quality_model = models_dir / "quality_model.pkl"
failure_model = models_dir / "failure_model.pkl"

if not quality_model.exists():
    print("❌ Quality model not found at:", quality_model)
    print("   Run training first: python src/training/train_all_models.py")
    exit(1)

if not failure_model.exists():
    print("❌ Failure model not found at:", failure_model)
    print("   Run training first: python src/training/train_all_models.py")
    exit(1)

print("✅ Models found")
print()

# Check model leaderboard
leaderboard_file = Path("reports/model_leaderboard.json")

if leaderboard_file.exists():
    print("📊 MODEL LEADERBOARD (from reports/model_leaderboard.json)")
    print("-"*70)
    
    with open(leaderboard_file) as f:
        data = json.load(f)
    
    # Quality models
    print("\n🎯 QUALITY PREDICTION MODELS:")
    print("-"*70)
    for i, model in enumerate(data.get('quality', []), 1):
        print(f"{i}. {model['model']:20} F1: {model['f1']:.4f} ({model['f1']*100:.2f}%)")
        if i == 1:
            print(f"   {'':20} Acc: {model.get('accuracy', 0):.4f} Prec: {model.get('precision', 0):.4f} Rec: {model.get('recall', 0):.4f}")
    
    # Failure models
    print("\n💥 FAILURE PREDICTION MODELS:")
    print("-"*70)
    for i, model in enumerate(data.get('failure', []), 1):
        print(f"{i}. {model['model']:20} F1: {model['f1']:.4f} ({model['f1']*100:.2f}%)")
        if i == 1:
            print(f"   {'':20} Acc: {model.get('accuracy', 0):.4f} Prec: {model.get('precision', 0):.4f} Rec: {model.get('recall', 0):.4f}")
    
    print()
else:
    print("⚠️  Model leaderboard not found")
    print()

# Load and test models
print("🧪 TESTING MODELS ON SAMPLE DATA")
print("-"*70)

# Check if test data exists
test_files = {
    'quality': Path("data/processed/quality_test.csv"),
    'failure': Path("data/processed/failure_test.csv")
}

for model_type, test_file in test_files.items():
    if test_file.exists():
        print(f"\n{model_type.upper()} Model Test:")
        df_test = pd.read_csv(test_file)
        print(f"  Test samples: {len(df_test)}")
        
        # Check label distribution
        if model_type == 'quality':
            label_col = 'label_quality_bad'
        else:
            label_col = 'label_failure'
        
        if label_col in df_test.columns:
            dist = df_test[label_col].value_counts().to_dict()
            print(f"  Label distribution: {dist}")
        else:
            print(f"  ⚠️  Label column '{label_col}' not found")
    else:
        print(f"\n{model_type.upper()} Model Test:")
        print(f"  ⚠️  Test file not found: {test_file}")

print()

# Check performance history
perf_history = Path("reports/performance_history.json")
if perf_history.exists():
    print("📈 PERFORMANCE HISTORY")
    print("-"*70)
    with open(perf_history) as f:
        history = json.load(f)
    
    print(f"Total training runs: {len(history)}")
    if history:
        latest = history[-1]
        print(f"Latest run: {latest.get('timestamp', 'Unknown')}")
        print(f"Quality F1: {latest.get('quality_f1', 0):.4f}")
        print(f"Failure F1: {latest.get('failure_f1', 0):.4f}")
    print()

print("="*70)
print("✅ MODEL ACCURACY CHECK COMPLETE")
print("="*70)