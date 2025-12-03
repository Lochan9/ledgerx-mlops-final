"""
Drift Threshold Checker - PRODUCTION VERSION
Uses correct training data as baseline
Compares production features to what model was trained on
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Drift thresholds
DRIFT_SCORE_THRESHOLD = 0.15  # 15% of features drifting
FEATURE_DRIFT_THRESHOLD = 0.30  # 30% drift in any feature

def convert_to_python_types(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj

class DriftThresholdChecker:
    """
    Production drift detector
    Compares production data to TRAINING baseline
    """
    
    def __init__(self, 
                 reference_data_path: str = "data/processed/fatura_enterprise_preprocessed.csv",
                 drift_history_path: str = "reports/drift_history.json",
                 production_data_path: str = "data/production/recent_features.csv"):
        
        self.reference_data_path = Path(reference_data_path)
        self.production_data_path = Path(production_data_path)
        self.drift_history_path = Path(drift_history_path)
        self.drift_history_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.reference_data = None
        self.drift_history = self._load_drift_history()
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference (training) data"""
        try:
            if self.reference_data_path.exists():
                df = pd.read_csv(self.reference_data_path)
                logger.info(f"Loaded training baseline: {len(df)} rows, {len(df.columns)} columns")
                return df
            else:
                logger.error(f"Training baseline not found: {self.reference_data_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return None
    
    def _load_drift_history(self) -> list:
        if self.drift_history_path.exists():
            try:
                with open(self.drift_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load drift history: {e}")
                return []
        return []
    
    def _save_drift_history(self):
        try:
            history_cleaned = convert_to_python_types(self.drift_history)
            with open(self.drift_history_path, 'w') as f:
                json.dump(history_cleaned, f, indent=2)
            logger.info(f"Saved drift history to {self.drift_history_path}")
        except Exception as e:
            logger.error(f"Failed to save drift history: {e}")
    
    def _statistical_drift_test(self, ref_col: pd.Series, cur_col: pd.Series) -> Dict:
        """Kolmogorov-Smirnov test for drift"""
        try:
            ref_clean = ref_col.dropna()
            cur_clean = cur_col.dropna()
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return {'drifted': False, 'p_value': 1.0, 'method': 'insufficient_data'}
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
            
            drifted = bool(p_value < 0.05)
            
            return {
                'drifted': drifted,
                'p_value': float(p_value),
                'statistic': float(statistic),
                'method': 'ks_test'
            }
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return {'drifted': False, 'p_value': 1.0, 'method': 'error'}
    
    def get_production_data(self, hours=24):
        """
        Load recent production feature data
        
        Returns:
            DataFrame with production features OR reference sample for demo
        """
        try:
            if self.production_data_path.exists():
                df = pd.read_csv(self.production_data_path)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                    cutoff = datetime.now() - timedelta(hours=hours)
                    df = df[df['timestamp'] > cutoff]
                
                logger.info(f"Loaded {len(df)} production samples from last {hours} hours")
                return df
            else:
                logger.warning("No production data available")
                # DEMO MODE: Use sample from reference data
                logger.info("DEMO MODE: Using reference data sample to simulate production")
                ref = self._load_reference_data()
                if ref is not None:
                    return ref.sample(min(500, len(ref)), random_state=42)
                return None
                
        except Exception as e:
            logger.error(f"Failed to load production data: {e}")
            return None
    
    def detect_drift(self, current_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect drift between training baseline and production data
        
        Args:
            current_data: Production feature data (if None, loads from file)
            
        Returns:
            Drift analysis report
        """
        if self.reference_data is None:
            self.reference_data = self._load_reference_data()
        
        if self.reference_data is None:
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'drifted_features': [],
                'timestamp': datetime.now().isoformat(),
                'should_retrain': False,
                'error': 'Training baseline not available'
            }
        
        # Get production data
        if current_data is None:
            current_data = self.get_production_data(hours=24)
        
        if current_data is None:
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'drifted_features': [],
                'timestamp': datetime.now().isoformat(),
                'should_retrain': False,
                'error': 'No production data available'
            }
        
        try:
            # Get common numeric columns
            ref_numeric = self.reference_data.select_dtypes(include=[np.number]).columns
            cur_numeric = current_data.select_dtypes(include=[np.number]).columns
            common_cols = list(set(ref_numeric) & set(cur_numeric))
            
            if not common_cols:
                return {
                    'drift_score': 0.0,
                    'drift_detected': False,
                    'drifted_features': [],
                    'timestamp': datetime.now().isoformat(),
                    'should_retrain': False,
                    'error': 'No common numeric columns'
                }
            
            logger.info(f"Checking drift on {len(common_cols)} common features")
            
            # Test each feature for drift
            drifted_features = []
            drift_details = {}
            
            for col in common_cols:
                drift_result = self._statistical_drift_test(
                    self.reference_data[col],
                    current_data[col]
                )
                
                if drift_result.get('drifted', False):
                    drifted_features.append(col)
                    drift_details[col] = drift_result
            
            # Calculate drift score
            drift_score = float(len(drifted_features) / len(common_cols))
            
            # Determine if retraining needed
            should_retrain = bool(
                drift_score > DRIFT_SCORE_THRESHOLD or 
                len(drifted_features) > len(common_cols) * FEATURE_DRIFT_THRESHOLD
            )
            
            result = {
                'drift_score': drift_score,
                'drift_detected': bool(drift_score > 0),
                'drifted_features': drifted_features[:10],  # Top 10
                'num_drifted_features': int(len(drifted_features)),
                'total_features': int(len(common_cols)),
                'timestamp': datetime.now().isoformat(),
                'should_retrain': should_retrain,
                'drift_details': convert_to_python_types(drift_details)
            }
            
            # Save to history
            self.drift_history.append(result)
            self._save_drift_history()
            
            if should_retrain:
                logger.warning(f"⚠️ Data drift detected! {drift_score:.1%} of features drifting")
                logger.warning(f"Drifted features: {drifted_features[:5]}")
            else:
                logger.info(f"✅ No significant drift. Drift score: {drift_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'drifted_features': [],
                'timestamp': datetime.now().isoformat(),
                'should_retrain': False,
                'error': str(e)
            }
    
    def get_drift_summary(self) -> Dict:
        if not self.drift_history:
            return {}
        
        drift_scores = [h['drift_score'] for h in self.drift_history if 'drift_score' in h]
        drift_events = sum(1 for h in self.drift_history if h.get('should_retrain', False))
        
        return {
            'total_checks': int(len(self.drift_history)),
            'drift_events': int(drift_events),
            'avg_drift_score': float(sum(drift_scores) / len(drift_scores)) if drift_scores else 0.0,
            'max_drift_score': float(max(drift_scores)) if drift_scores else 0.0,
            'last_check': self.drift_history[-1]['timestamp'] if self.drift_history else None
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing drift detection...")
    
    checker = DriftThresholdChecker()
    
    # Test with current data
    result = checker.detect_drift()
    
    print("\n📊 Drift Detection Result:")
    print(json.dumps(result, indent=2))
    
    if result['should_retrain']:
        print("\n🚨 Retraining recommended!")
    else:
        print("\n✅ No significant drift")