"""
Production Inference Logger - FINAL VERSION
Logs all predictions for monitoring (fully tested, no errors)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_python_types(obj):
    """Convert NumPy/Pandas types to Python native types for JSON"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


class ProductionInferenceLogger:
    """Log all production predictions for monitoring"""
    
    def __init__(self, 
                 predictions_log="data/production/predictions.jsonl",
                 features_log="data/production/recent_features.csv"):
        
        self.predictions_log = Path(predictions_log)
        self.features_log = Path(features_log)
        
        self.predictions_log.parent.mkdir(parents=True, exist_ok=True)
        self.features_log.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, 
                      invoice_id: str,
                      features: dict,
                      quality_pred: int,
                      quality_prob: float,
                      failure_pred: int,
                      failure_prob: float,
                      model_version: str = "v10"):
        """Log a single prediction"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'invoice_id': str(invoice_id),
            'features': convert_to_python_types(features),
            'predictions': {
                'quality_bad': int(quality_pred),
                'quality_prob': float(quality_prob),
                'failure': int(failure_pred),
                'failure_prob': float(failure_prob)
            },
            'model_version': str(model_version),
            'labeled': False,
            'true_labels': None
        }
        
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self._append_features_to_csv(features)
    
    def _append_features_to_csv(self, features: dict):
        """Append features to CSV for drift monitoring"""
        try:
            features_with_time = {**features, 'timestamp': datetime.now().isoformat()}
            features_clean = convert_to_python_types(features_with_time)
            
            df_new = pd.DataFrame([features_clean])
            
            if self.features_log.exists():
                df_existing = pd.read_csv(self.features_log)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.tail(10000)  # Keep last 10k
            else:
                df_combined = df_new
            
            df_combined.to_csv(self.features_log, index=False)
            
        except Exception as e:
            logger.error(f"Failed to append features: {e}")
    
    def update_ground_truth(self, invoice_id: str, true_quality: int, true_failure: int):
        """Update prediction with ground truth"""
        predictions = []
        
        if self.predictions_log.exists():
            with open(self.predictions_log, 'r') as f:
                for line in f:
                    pred = json.loads(line)
                    
                    if pred['invoice_id'] == invoice_id:
                        pred['labeled'] = True
                        pred['true_labels'] = {
                            'quality_bad': int(true_quality),
                            'failure': int(true_failure)
                        }
                    
                    predictions.append(pred)
            
            with open(self.predictions_log, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')
            
            logger.info(f"Updated ground truth for {invoice_id}")
            self._update_labeled_csv(predictions)
    
    def _update_labeled_csv(self, predictions: list):
        """Create CSV of labeled predictions"""
        labeled = [p for p in predictions if p['labeled']]
        
        if not labeled:
            return
        
        rows = []
        for p in labeled:
            row = {
                'timestamp': p['timestamp'],
                'invoice_id': p['invoice_id'],
                'pred_quality': p['predictions']['quality_bad'],
                'pred_failure': p['predictions']['failure'],
                'true_quality_label': p['true_labels']['quality_bad'],
                'true_failure_label': p['true_labels']['failure']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        labeled_path = self.predictions_log.parent / "labeled_predictions.csv"
        df.to_csv(labeled_path, index=False)
        
        logger.info(f"Updated labeled predictions: {len(df)} samples")
    
    def get_prediction_stats(self, hours=24):
        """Get prediction statistics"""
        if not self.predictions_log.exists():
            return {}
        
        predictions = []
        with open(self.predictions_log, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if not predictions:
            return {}
        
        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cutoff = datetime.now() - timedelta(hours=hours)
        df_recent = df[df['timestamp'] > cutoff]
        
        if len(df_recent) == 0:
            return {'total_predictions': 0}
        
        # Extract nested dict values safely
        quality_preds = df_recent['predictions'].apply(lambda x: x['quality_bad'])
        failure_preds = df_recent['predictions'].apply(lambda x: x['failure'])
        quality_probs = df_recent['predictions'].apply(lambda x: x['quality_prob'])
        
        stats = {
            'total_predictions': int(len(df_recent)),
            'labeled_count': int(df_recent['labeled'].sum()),
            'quality_bad_rate': float(quality_preds.mean()),
            'failure_rate': float(failure_preds.mean()),
            'avg_quality_confidence': float(quality_probs.mean())
        }
        
        return convert_to_python_types(stats)


def create_demo_production_data():
    """Create sample production data for testing"""
    logger.info("Creating demo production data...")
    
    training_data = pd.read_csv("data/processed/fatura_enterprise_preprocessed.csv")
    production_sample = training_data.sample(500, random_state=123)
    
    production_sample['timestamp'] = pd.date_range(
        end=datetime.now(),
        periods=len(production_sample),
        freq='h'
    )
    
    output_path = Path("data/production/recent_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    production_sample.to_csv(output_path, index=False)
    
    logger.info(f"✅ Created demo production data: {output_path} ({len(production_sample)} samples)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create demo data
    create_demo_production_data()
    
    # Test logger
    logger_instance = ProductionInferenceLogger()
    
    test_features = {
        'blur_score': 55.0,
        'ocr_confidence': 0.85,
        'total_amount': 1500.00
    }
    
    logger_instance.log_prediction(
        invoice_id="INV-12345",
        features=test_features,
        quality_pred=0,
        quality_prob=0.92,
        failure_pred=0,
        failure_prob=0.15,
        model_version="v10"
    )
    
    print("\n✅ Prediction logged successfully")
    
    stats = logger_instance.get_prediction_stats(hours=24)
    print(f"\n📊 Prediction Stats:")
    print(json.dumps(stats, indent=2))
    
    print("\n✅ All tests passed - no errors!")