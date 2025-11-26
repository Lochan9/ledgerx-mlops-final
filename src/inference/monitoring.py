"""
LedgerX - Model Performance Monitoring Module
==============================================

Implements comprehensive monitoring including:
- Prometheus metrics
- Performance tracking
- Drift detection
- Alerting
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger("ledgerx_monitoring")

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Prediction Counters
prediction_total = Counter(
    'ledgerx_predictions_total',
    'Total number of predictions made',
    ['model', 'prediction_class', 'user']
)

prediction_errors = Counter(
    'ledgerx_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Latency Metrics
prediction_latency = Histogram(
    'ledgerx_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Model Performance Metrics
model_quality_probability = Gauge(
    'ledgerx_quality_probability',
    'Average quality probability (recent predictions)',
)

model_failure_probability = Gauge(
    'ledgerx_failure_probability',
    'Average failure probability (recent predictions)',
)

quality_predictions_bad = Counter(
    'ledgerx_quality_bad_total',
    'Total number of quality=bad predictions'
)

quality_predictions_good = Counter(
    'ledgerx_quality_good_total',
    'Total number of quality=good predictions'
)

failure_predictions_risk = Counter(
    'ledgerx_failure_risk_total',
    'Total number of failure=risk predictions'
)

failure_predictions_safe = Counter(
    'ledgerx_failure_safe_total',
    'Total number of failure=safe predictions'
)

# Input Feature Distributions
feature_blur_score = Histogram(
    'ledgerx_feature_blur_score',
    'Distribution of blur score values',
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

feature_ocr_confidence = Histogram(
    'ledgerx_feature_ocr_confidence',
    'Distribution of OCR confidence values',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

feature_total_amount = Histogram(
    'ledgerx_feature_total_amount',
    'Distribution of invoice amounts',
    buckets=[0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000]
)

# System Info
model_info = Info(
    'ledgerx_model_info',
    'Information about deployed models'
)

# Active Users
active_users = Gauge(
    'ledgerx_active_users',
    'Number of unique users in last 5 minutes'
)

# ============================================================================
# MONITORING MANAGER
# ============================================================================

class ModelMonitor:
    """
    Manages model performance monitoring and drift detection
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent predictions to keep for analysis
        """
        self.window_size = window_size
        
        # Circular buffers for recent predictions
        self.recent_quality_probs = deque(maxlen=window_size)
        self.recent_failure_probs = deque(maxlen=window_size)
        self.recent_features = deque(maxlen=window_size)
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_users = deque(maxlen=100)  # Last 100 requests
        
        # Baseline statistics (set during initialization)
        self.baseline_stats = None
        
        # Drift detection thresholds
        self.drift_threshold = 0.1  # 10% change triggers alert
        self.performance_threshold = 0.05  # 5% degradation triggers alert
        
        logger.info("[MONITOR] Model monitor initialized")
    
    def log_prediction(
        self,
        features: Dict,
        quality_result: Dict,
        user: str,
        latency: float
    ):
        """
        Log a prediction for monitoring
        
        Args:
            features: Input features
            quality_result: Prediction results
            user: Username who made the request
            latency: Prediction latency in seconds
        """
        # Update Prometheus metrics
        prediction_total.labels(
            model='quality',
            prediction_class=str(quality_result['quality_bad']),
            user=user
        ).inc()
        
        prediction_total.labels(
            model='failure',
            prediction_class=str(quality_result['failure_risk']),
            user=user
        ).inc()
        
        # Track latency
        prediction_latency.labels(model='quality').observe(latency)
        
        # Track predictions
        if quality_result['quality_bad'] == 1:
            quality_predictions_bad.inc()
        else:
            quality_predictions_good.inc()
        
        if quality_result['failure_risk'] == 1:
            failure_predictions_risk.inc()
        else:
            failure_predictions_safe.inc()
        
        # Track feature distributions
        if 'blur_score' in features:
            feature_blur_score.observe(features['blur_score'])
        if 'ocr_confidence' in features:
            feature_ocr_confidence.observe(features['ocr_confidence'])
        if 'total_amount' in features:
            feature_total_amount.observe(features['total_amount'])
        
        # Store in circular buffers
        self.recent_quality_probs.append(quality_result['quality_probability'])
        self.recent_failure_probs.append(quality_result['failure_probability'])
        self.recent_features.append(features)
        self.recent_predictions.append(quality_result)
        self.recent_users.append((user, datetime.utcnow()))
        
        # Update gauges
        if len(self.recent_quality_probs) > 0:
            model_quality_probability.set(np.mean(self.recent_quality_probs))
            model_failure_probability.set(np.mean(self.recent_failure_probs))
        
        # Update active users count
        self._update_active_users()
    
    def _update_active_users(self):
        """Update count of active users in last 5 minutes"""
        now = datetime.utcnow()
        recent = [u for u, t in self.recent_users if (now - t).seconds < 300]
        unique_users = len(set(recent))
        active_users.set(unique_users)
    
    def log_error(self, error_type: str):
        """Log a prediction error"""
        prediction_errors.labels(error_type=error_type).inc()
    
    def get_recent_stats(self) -> Dict:
        """
        Get statistics from recent predictions
        
        Returns:
            Dictionary with recent statistics
        """
        if len(self.recent_quality_probs) == 0:
            return {
                "status": "no_data",
                "message": "No recent predictions available"
            }
        
        stats = {
            "window_size": len(self.recent_quality_probs),
            "quality": {
                "mean_probability": float(np.mean(self.recent_quality_probs)),
                "std_probability": float(np.std(self.recent_quality_probs)),
                "min_probability": float(np.min(self.recent_quality_probs)),
                "max_probability": float(np.max(self.recent_quality_probs)),
            },
            "failure": {
                "mean_probability": float(np.mean(self.recent_failure_probs)),
                "std_probability": float(np.std(self.recent_failure_probs)),
                "min_probability": float(np.min(self.recent_failure_probs)),
                "max_probability": float(np.max(self.recent_failure_probs)),
            },
            "predictions": {
                "quality_bad_rate": sum(1 for p in self.recent_predictions if p['quality_bad'] == 1) / len(self.recent_predictions),
                "failure_risk_rate": sum(1 for p in self.recent_predictions if p['failure_risk'] == 1) / len(self.recent_predictions),
            }
        }
        
        return stats
    
    def set_baseline(self, stats: Optional[Dict] = None):
        """
        Set baseline statistics for drift detection
        
        Args:
            stats: Baseline statistics (if None, use current stats)
        """
        if stats is None:
            stats = self.get_recent_stats()
        
        self.baseline_stats = stats
        logger.info(f"[MONITOR] Baseline set: {stats}")
    
    def detect_drift(self) -> Dict:
        """
        Detect data drift by comparing recent stats to baseline
        
        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_stats is None:
            return {
                "status": "no_baseline",
                "message": "Baseline not set. Call set_baseline() first."
            }
        
        if len(self.recent_quality_probs) < 10:
            return {
                "status": "insufficient_data",
                "message": "Need at least 10 predictions for drift detection"
            }
        
        current_stats = self.get_recent_stats()
        
        # Calculate drift for quality model
        quality_mean_drift = abs(
            current_stats['quality']['mean_probability'] - 
            self.baseline_stats['quality']['mean_probability']
        )
        
        # Calculate drift for failure model
        failure_mean_drift = abs(
            current_stats['failure']['mean_probability'] - 
            self.baseline_stats['failure']['mean_probability']
        )
        
        # Determine if drift detected
        quality_drift_detected = quality_mean_drift > self.drift_threshold
        failure_drift_detected = failure_mean_drift > self.drift_threshold
        
        drift_result = {
            "status": "checked",
            "timestamp": datetime.utcnow().isoformat(),
            "quality_model": {
                "drift_detected": quality_drift_detected,
                "drift_magnitude": float(quality_mean_drift),
                "threshold": self.drift_threshold,
                "baseline_mean": self.baseline_stats['quality']['mean_probability'],
                "current_mean": current_stats['quality']['mean_probability'],
            },
            "failure_model": {
                "drift_detected": failure_drift_detected,
                "drift_magnitude": float(failure_mean_drift),
                "threshold": self.drift_threshold,
                "baseline_mean": self.baseline_stats['failure']['mean_probability'],
                "current_mean": current_stats['failure']['mean_probability'],
            },
            "overall_drift_detected": quality_drift_detected or failure_drift_detected
        }
        
        if drift_result['overall_drift_detected']:
            logger.warning(f"[MONITOR] DRIFT DETECTED: {drift_result}")
        
        return drift_result
    
    def get_health_status(self) -> Dict:
        """
        Get overall health status of the monitoring system
        
        Returns:
            Dictionary with health status
        """
        recent_stats = self.get_recent_stats()
        drift_status = self.detect_drift() if self.baseline_stats else {"status": "no_baseline"}
        
        # Determine overall health
        is_healthy = True
        issues = []
        
        if drift_status.get('overall_drift_detected', False):
            is_healthy = False
            issues.append("Drift detected in model predictions")
        
        if recent_stats.get('status') == 'no_data':
            is_healthy = False
            issues.append("No recent predictions")
        
        return {
            "healthy": is_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "recent_predictions": len(self.recent_quality_probs),
            "baseline_set": self.baseline_stats is not None,
            "drift_status": drift_status,
            "recent_stats": recent_stats
        }


# Global monitor instance
monitor = ModelMonitor(window_size=100)

# Set model info
model_info.info({
    'quality_model': 'CatBoost',
    'failure_model': 'RandomForest',
    'version': '2.0.0',
    'deployment_date': datetime.utcnow().isoformat()
})