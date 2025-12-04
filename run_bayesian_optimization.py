"""
LedgerX - Run Bayesian Optimization on Production Models
Improves 77.1% / 70.9% → Target 79-82% / 73-76%
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.hyperparameter_tuning_ADVANCED import (
    tune_catboost_quality
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("BAYESIAN OPTIMIZATION - Improving Production Models")
    logger.info("Current: Quality 77.1% F1, Failure 70.9% F1")
    logger.info("Target:  Quality 79-82% F1, Failure 73-76% F1")
    logger.info("="*80)
    
    # Tune Quality Model
    logger.info("\n🎯 Starting Bayesian optimization for Quality Model...")
    logger.info("Trials: 100 | Method: TPE (Tree-structured Parzen Estimator)")
    logger.info("Estimated time: 15-20 minutes\n")
    
    best_quality_params, best_quality_f1 = tune_catboost_quality(n_trials=100)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Quality Model Optimization Complete!")
    logger.info(f"Best F1: {best_quality_f1:.4f} (Current: 0.7710)")
    logger.info(f"Improvement: {(best_quality_f1 - 0.7710)*100:+.2f}%")
    logger.info("="*80)
    logger.info(f"\nBest Parameters:")
    for param, value in best_quality_params.items():
        logger.info(f"  {param:20s} = {value}")
    
    # Note about failure model
    logger.info("\n" + "="*80)
    logger.info("📝 NOTE: Failure model tuning function not available")
    logger.info("You can add tune_catboost_failure() to hyperparameter_tuning_ADVANCED.py")
    logger.info("Or use the optimized quality parameters as a starting point")
    logger.info("="*80)
    
    # Save best parameters
    import json
    best_params = {
        'quality': {
            'params': best_quality_params,
            'f1_score': best_quality_f1,
            'improvement': (best_quality_f1 - 0.7710) * 100
        },
        'failure': {
            'note': 'Use similar parameters to quality model',
            'suggested_params': best_quality_params  # Use same as starting point
        }
    }
    
    output_file = Path('reports/bayesian_optimization_results.json')
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    
    # Generate update instructions
    logger.info("\n" + "="*80)
    logger.info("📝 NEXT STEPS - Update train_all_models.py")
    logger.info("="*80)
    logger.info("\n1. Open src/training/train_all_models.py")
    logger.info(f"\n2. Find line ~283 (Quality CatBoost) and update to:")
    logger.info("   CatBoostClassifier(")
    for param, value in best_quality_params.items():
        logger.info(f"       {param}={value},")
    logger.info("       auto_class_weights='Balanced',")
    logger.info("       random_seed=42,")
    logger.info("       verbose=0")
    logger.info("   )")
    
    logger.info(f"\n4. Retrain models with optimized parameters:")
    logger.info("   dvc repro --force")
    
    logger.info(f"\n5. Expected new performance:")
    logger.info(f"   Quality F1: {best_quality_f1:.1%} (currently 77.1%)")
    logger.info(f"   Failure F1: Use similar params and retrain")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 Bayesian Optimization Complete!")
    logger.info("="*80)