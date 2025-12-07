"""
LedgerX - Google Cloud Logging Integration
===========================================

Centralized logging infrastructure integrating:
- Structured logging to GCP Cloud Logging
- Local development fallback
- Automatic context enrichment
- Log level management
- Error tracking and alerting
"""

import os
import sys
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Environment detection
IS_CLOUD_RUN = os.getenv("K_SERVICE") is not None
IS_GCP = os.getenv("GOOGLE_CLOUD_PROJECT") is not None or os.getenv("GCP_PROJECT") is not None
ENABLE_CLOUD_LOGGING = os.getenv("ENABLE_CLOUD_LOGGING", "true").lower() == "true"

# Try to import Google Cloud Logging
try:
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    from google.cloud.logging_v2.handlers import setup_logging
    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False
    print("[WARNING] google-cloud-logging not installed. Install with: pip install google-cloud-logging")


class CloudLogger:
    """
    Unified logging interface that automatically routes to:
    - GCP Cloud Logging when running on Cloud Run or GCP
    - Local structured logging for development
    """
    
    def __init__(
        self,
        name: str = "ledgerx",
        project_id: Optional[str] = None,
        log_level: str = "INFO",
        enable_cloud_logging: Optional[bool] = None
    ):
        """
        Initialize Cloud Logger
        
        Args:
            name: Logger name (used as log name in Cloud Logging)
            project_id: GCP project ID (auto-detected if not provided)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_cloud_logging: Force enable/disable Cloud Logging (auto-detect if None)
        """
        self.name = name
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Determine if Cloud Logging should be used
        if enable_cloud_logging is None:
            self.use_cloud_logging = (
                CLOUD_LOGGING_AVAILABLE and 
                ENABLE_CLOUD_LOGGING and 
                (IS_CLOUD_RUN or IS_GCP) and
                self.project_id is not None
            )
        else:
            self.use_cloud_logging = enable_cloud_logging and CLOUD_LOGGING_AVAILABLE
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Log initialization
        self._log_initialization()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging handler based on environment"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        if self.use_cloud_logging:
            return self._setup_cloud_logging(logger)
        else:
            return self._setup_local_logging(logger)
    
    def _setup_cloud_logging(self, logger: logging.Logger) -> logging.Logger:
        """Setup Google Cloud Logging"""
        try:
            # Initialize Cloud Logging client
            client = google.cloud.logging.Client(project=self.project_id)
            
            # Create Cloud Logging handler
            handler = CloudLoggingHandler(client, name=self.name)
            handler.setLevel(self.log_level)
            
            # Add handler to logger
            logger.addHandler(handler)
            
            # Also setup automatic logging integration for stdlib logging
            # This captures logs from all Python loggers
            setup_logging(handler, log_level=self.log_level)
            
            return logger
            
        except Exception as e:
            print(f"[ERROR] Failed to setup Cloud Logging: {e}")
            print("[INFO] Falling back to local logging")
            return self._setup_local_logging(logger)
    
    def _setup_local_logging(self, logger: logging.Logger) -> logging.Logger:
        """Setup local structured logging for development"""
        # Create formatter with structured JSON-like output
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional, for local persistence)
        if os.getenv("LOG_TO_FILE", "false").lower() == "true":
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"{self.name}.log",
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _log_initialization(self):
        """Log initialization details"""
        self.logger.info(
            f"Logging initialized",
            extra={
                'environment': 'cloud_run' if IS_CLOUD_RUN else 'local',
                'cloud_logging_enabled': self.use_cloud_logging,
                'project_id': self.project_id,
                'log_level': logging.getLevelName(self.log_level)
            }
        )
    
    # Convenience methods that match standard logging interface
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
    
    # Structured logging methods
    def log_prediction(
        self,
        user_id: str,
        invoice_id: str,
        quality_prediction: str,
        failure_prediction: str,
        latency_ms: float,
        **metadata
    ):
        """Log ML prediction with structured data"""
        self.info(
            "ML prediction completed",
            prediction_type="invoice_prediction",
            user_id=user_id,
            invoice_id=invoice_id,
            quality_prediction=quality_prediction,
            failure_prediction=failure_prediction,
            latency_ms=latency_ms,
            **metadata
        )
    
    def log_api_request(
        self,
        method: str,
        endpoint: str,
        user_id: Optional[str],
        status_code: int,
        latency_ms: float,
        **metadata
    ):
        """Log API request with structured data"""
        level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        
        getattr(self, level)(
            f"API {method} {endpoint}",
            request_type="api_request",
            method=method,
            endpoint=endpoint,
            user_id=user_id,
            status_code=status_code,
            latency_ms=latency_ms,
            **metadata
        )
    
    def log_model_performance(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        **metadata
    ):
        """Log model performance metrics"""
        self.info(
            f"Model performance: {model_name}",
            event_type="model_performance",
            model_name=model_name,
            metric_name=metric_name,
            metric_value=metric_value,
            **metadata
        )
    
    def log_data_drift(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float,
        drift_detected: bool,
        **metadata
    ):
        """Log data drift detection"""
        level = "warning" if drift_detected else "info"
        
        getattr(self, level)(
            f"Data drift check: {feature_name}",
            event_type="data_drift",
            feature_name=feature_name,
            drift_score=drift_score,
            threshold=threshold,
            drift_detected=drift_detected,
            **metadata
        )
    
    def log_retraining_trigger(
        self,
        trigger_reason: str,
        model_name: str,
        **metadata
    ):
        """Log model retraining trigger"""
        self.warning(
            f"Model retraining triggered: {trigger_reason}",
            event_type="retraining_trigger",
            trigger_reason=trigger_reason,
            model_name=model_name,
            **metadata
        )
    
    def log_ocr_processing(
        self,
        invoice_id: str,
        ocr_engine: str,
        confidence: float,
        processing_time_ms: float,
        **metadata
    ):
        """Log OCR processing"""
        self.info(
            f"OCR processing completed",
            event_type="ocr_processing",
            invoice_id=invoice_id,
            ocr_engine=ocr_engine,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            **metadata
        )
    
    def log_cost_event(
        self,
        service: str,
        operation: str,
        cost_usd: Optional[float] = None,
        units_consumed: Optional[int] = None,
        **metadata
    ):
        """Log cost-related events (Document AI, Cloud SQL, etc.)"""
        self.info(
            f"Cost event: {service} - {operation}",
            event_type="cost_tracking",
            service=service,
            operation=operation,
            cost_usd=cost_usd,
            units_consumed=units_consumed,
            **metadata
        )


# Singleton instance for easy access
_global_logger: Optional[CloudLogger] = None


def get_logger(
    name: str = "ledgerx",
    project_id: Optional[str] = None,
    log_level: Optional[str] = None
) -> CloudLogger:
    """
    Get or create global CloudLogger instance
    
    Args:
        name: Logger name
        project_id: GCP project ID (optional)
        log_level: Log level (optional)
    
    Returns:
        CloudLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = CloudLogger(
            name=name,
            project_id=project_id,
            log_level=log_level or os.getenv("LOG_LEVEL", "INFO")
        )
    
    return _global_logger


def setup_cloud_logging(
    name: str = "ledgerx",
    project_id: Optional[str] = None,
    log_level: str = "INFO"
) -> CloudLogger:
    """
    Setup Cloud Logging (call this once at application startup)
    
    Args:
        name: Logger name
        project_id: GCP project ID
        log_level: Logging level
    
    Returns:
        CloudLogger instance
    """
    return get_logger(name=name, project_id=project_id, log_level=log_level)


# Convenience functions for quick logging
def log_info(message: str, **kwargs):
    """Quick info log"""
    get_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Quick warning log"""
    get_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Quick error log"""
    get_logger().error(message, **kwargs)


def log_exception(message: str, **kwargs):
    """Quick exception log with traceback"""
    get_logger().exception(message, **kwargs)


if __name__ == "__main__":
    """Test Cloud Logging setup"""
    print("="*80)
    print("LedgerX Cloud Logging - Test")
    print("="*80)
    print()
    print(f"Environment Detection:")
    print(f"  IS_CLOUD_RUN: {IS_CLOUD_RUN}")
    print(f"  IS_GCP: {IS_GCP}")
    print(f"  CLOUD_LOGGING_AVAILABLE: {CLOUD_LOGGING_AVAILABLE}")
    print()
    
    # Initialize logger
    logger = setup_cloud_logging(log_level="INFO")
    
    print(f"Logger Configuration:")
    print(f"  Name: {logger.name}")
    print(f"  Project ID: {logger.project_id}")
    print(f"  Using Cloud Logging: {logger.use_cloud_logging}")
    print(f"  Log Level: {logging.getLevelName(logger.log_level)}")
    print()
    
    # Test basic logging
    print("Testing basic logging methods...")
    logger.info("Test info message", test_field="test_value")
    logger.warning("Test warning message", warning_type="test")
    logger.error("Test error message", error_code="TEST_ERROR")
    
    # Test structured logging
    print("\nTesting structured logging methods...")
    logger.log_prediction(
        user_id="test_user",
        invoice_id="inv_12345",
        quality_prediction="good",
        failure_prediction="safe",
        latency_ms=125.5,
        model_version="v1.2.0"
    )
    
    logger.log_api_request(
        method="POST",
        endpoint="/api/v1/predict",
        user_id="test_user",
        status_code=200,
        latency_ms=250.3
    )
    
    logger.log_model_performance(
        model_name="quality_model",
        metric_name="f1_score",
        metric_value=0.977
    )
    
    logger.log_data_drift(
        feature_name="blur_score",
        drift_score=0.08,
        threshold=0.10,
        drift_detected=False
    )
    
    print()
    print("="*80)
    print("Test complete!")
    if logger.use_cloud_logging:
        print(f"✅ Logs sent to Cloud Logging: {logger.project_id}")
        print(f"   View at: https://console.cloud.google.com/logs/query?project={logger.project_id}")
    else:
        print("ℹ️  Using local logging (Cloud Logging not enabled)")
    print("="*80)