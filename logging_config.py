import logging
import logging.handlers
from pathlib import Path

def setup_logging(name: str, log_file: str = None, level: int = logging.INFO):
    """
    Setup logging configuration
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

---

#!/usr/bin/env python3
"""
File 2: src/utils/config.py
Configuration loader
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path: str = "config/config.yaml"):
    """Load YAML configuration"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_env():
    """Load environment variables from .env"""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
    return dict(os.environ)

def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default"""
    return os.getenv(key, default)

---

#!/usr/bin/env python3
"""
File 3: src/utils/__init__.py
Utilities module
"""

from .logging_config import setup_logging
from .config import load_config, load_env, get_env

__all__ = ['setup_logging', 'load_config', 'load_env', 'get_env']

---

#!/usr/bin/env python3
"""
File 4: src/inference/__init__.py
Inference module
"""

---

#!/usr/bin/env python3
"""
File 5: src/inference/predictor.py
Prediction service
"""

import pickle
import pandas as pd
import logging
from pathlib import Path
import catboost

logger = logging.getLogger(__name__)

class InvoicePredictor:
    """Handles quality and failure predictions"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.quality_model = None
        self.failure_model = None
        self.encoders = None
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            quality_path = self.models_dir / "quality_catboost.cbm"
            if quality_path.exists():
                self.quality_model = catboost.CatBoostClassifier()
                self.quality_model.load_model(str(quality_path))
                logger.info("✅ Quality model loaded")
            
            failure_path = self.models_dir / "failure_catboost.cbm"
            if failure_path.exists():
                self.failure_model = catboost.CatBoostClassifier()
                self.failure_model.load_model(str(failure_path))
                logger.info("✅ Failure model loaded")
            
            encoders_path = self.models_dir / "encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                logger.info("✅ Encoders loaded")
        
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def prepare_features(self, data: dict) -> pd.DataFrame:
        """Prepare features for prediction"""
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        if self.encoders:
            for col, le in self.encoders.items():
                if col in df.columns:
                    try:
                        df[col] = le.transform(df[col].astype(str))
                    except Exception as e:
                        logger.warning(f"Could not encode {col}: {e}")
        
        return df
    
    def predict_quality(self, data: dict) -> dict:
        """Predict invoice quality"""
        if not self.quality_model:
            return {'error': 'Model not loaded'}
        
        features = self.prepare_features(data)
        prediction = self.quality_model.predict(features)[0]
        probability = self.quality_model.predict_proba(features)[0]
        
        return {
            'prediction': "good" if prediction == 1 else "fair",
            'confidence': float(max(probability)),
            'probability': float(max(probability))
        }
    
    def predict_failure(self, data: dict) -> dict:
        """Predict payment failure risk"""
        if not self.failure_model:
            return {'error': 'Model not loaded'}
        
        features = self.prepare_features(data)
        prediction = self.failure_model.predict(features)[0]
        probability = self.failure_model.predict_proba(features)[0]
        
        return {
            'prediction': "at_risk" if prediction == 1 else "safe",
            'confidence': float(max(probability)),
            'probability': float(probability[1])
        }

---

#!/usr/bin/env python3
"""
File 6: src/training/__init__.py
Training module
"""