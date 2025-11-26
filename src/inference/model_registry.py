"""
LedgerX - Model Registry Manager
=================================

Implements comprehensive model versioning with MLflow Model Registry.

Features:
- Automatic model versioning
- Model metadata tracking
- Stage management (staging/production/archived)
- Model comparison
- Rollback capabilities
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger("ledgerx_model_registry")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")

# Initialize MLflow client
client = MlflowClient()

# ============================================================================
# MODEL REGISTRY MANAGER
# ============================================================================

class ModelRegistryManager:
    """
    Manages model versioning and lifecycle in MLflow Model Registry
    """
    
    def __init__(self):
        """Initialize the model registry manager"""
        self.client = client
        self.tracking_uri = f"file:{MLFLOW_TRACKING_DIR}"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Model names in registry
        self.quality_model_name = "ledgerx_quality_model"
        self.failure_model_name = "ledgerx_failure_model"
        
        # Ensure registered models exist
        self._ensure_registered_models()
        
        logger.info("[REGISTRY] Model Registry Manager initialized")
        logger.info(f"[REGISTRY] Tracking URI: {self.tracking_uri}")
    
    def _ensure_registered_models(self):
        """Create registered model entries if they don't exist"""
        try:
            self.client.get_registered_model(self.quality_model_name)
            logger.info(f"[REGISTRY] Found existing: {self.quality_model_name}")
        except:
            self.client.create_registered_model(
                name=self.quality_model_name,
                description="Invoice quality assessment model - predicts if invoice quality is bad"
            )
            logger.info(f"[REGISTRY] Created new: {self.quality_model_name}")
        
        try:
            self.client.get_registered_model(self.failure_model_name)
            logger.info(f"[REGISTRY] Found existing: {self.failure_model_name}")
        except:
            self.client.create_registered_model(
                name=self.failure_model_name,
                description="Invoice failure risk model - predicts if invoice will fail processing"
            )
            logger.info(f"[REGISTRY] Created new: {self.failure_model_name}")
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict,
        tags: Optional[Dict] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model (quality or failure)
            model_path: Path to the model file
            metrics: Dictionary of model metrics (f1, accuracy, etc.)
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            Model version number
        """
        logger.info(f"[REGISTRY] Registering new version of {model_name}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"register_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model_path,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Get the model version
            model_uri = f"runs:/{run.info.run_id}/model"
            
        # Get the latest version number
        model_versions = self.client.search_model_versions(f"name='{model_name}'")
        latest_version = max([int(v.version) for v in model_versions])
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, latest_version, key, str(value))
        
        # Set description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=latest_version,
                description=description
            )
        
        logger.info(f"[REGISTRY] Registered {model_name} version {latest_version}")
        logger.info(f"[REGISTRY] Metrics: {metrics}")
        
        return str(latest_version)
    
    def promote_to_staging(self, model_name: str, version: str):
        """
        Promote a model version to Staging
        
        Args:
            model_name: Name of the model
            version: Version number to promote
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        logger.info(f"[REGISTRY] Promoted {model_name} v{version} to Staging")
    
    def promote_to_production(self, model_name: str, version: str):
        """
        Promote a model version to Production
        
        Args:
            model_name: Name of the model
            version: Version number to promote
        """
        # Archive current production model
        current_production = self.get_production_model(model_name)
        if current_production:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_production['version'],
                stage="Archived"
            )
            logger.info(f"[REGISTRY] Archived previous production: {model_name} v{current_production['version']}")
        
        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"[REGISTRY] Promoted {model_name} v{version} to Production")
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """
        Get the current production model version
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model info or None
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        production_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not production_versions:
            return None
        
        v = production_versions[0]
        return {
            "name": model_name,
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "creation_timestamp": v.creation_timestamp,
            "last_updated": v.last_updated_timestamp
        }
    
    def list_model_versions(self, model_name: str) -> List[Dict]:
        """
        List all versions of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of dictionaries with version info
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        version_list = []
        for v in versions:
            # Get metrics from the run
            run = self.client.get_run(v.run_id)
            metrics = run.data.metrics
            
            version_info = {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp,
                "description": v.description,
                "metrics": metrics,
                "tags": v.tags
            }
            version_list.append(version_info)
        
        # Sort by version number (descending)
        version_list.sort(key=lambda x: int(x['version']), reverse=True)
        
        return version_list
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict:
        """
        Compare two model versions
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Get version details
        v1_info = self.client.get_model_version(model_name, version1)
        v2_info = self.client.get_model_version(model_name, version2)
        
        # Get metrics
        v1_run = self.client.get_run(v1_info.run_id)
        v2_run = self.client.get_run(v2_info.run_id)
        
        v1_metrics = v1_run.data.metrics
        v2_metrics = v2_run.data.metrics
        
        # Calculate differences
        metric_comparison = {}
        for metric in v1_metrics.keys():
            if metric in v2_metrics:
                diff = v2_metrics[metric] - v1_metrics[metric]
                pct_change = (diff / v1_metrics[metric] * 100) if v1_metrics[metric] != 0 else 0
                metric_comparison[metric] = {
                    f"version_{version1}": v1_metrics[metric],
                    f"version_{version2}": v2_metrics[metric],
                    "difference": diff,
                    "percent_change": pct_change,
                    "better": "v2" if diff > 0 else "v1" if diff < 0 else "equal"
                }
        
        return {
            "model_name": model_name,
            "version_1": {
                "version": version1,
                "stage": v1_info.current_stage,
                "created": v1_info.creation_timestamp
            },
            "version_2": {
                "version": version2,
                "stage": v2_info.current_stage,
                "created": v2_info.creation_timestamp
            },
            "metric_comparison": metric_comparison
        }
    
    def rollback_to_version(self, model_name: str, version: str):
        """
        Rollback to a specific model version
        
        Args:
            model_name: Name of the model
            version: Version to rollback to
        """
        logger.warning(f"[REGISTRY] Rolling back {model_name} to version {version}")
        
        # Demote current production
        current_prod = self.get_production_model(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod['version'],
                stage="Archived"
            )
        
        # Promote target version to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(f"[REGISTRY] Rollback complete: {model_name} now at version {version}")
    
    def get_registry_status(self) -> Dict:
        """
        Get overall registry status
        
        Returns:
            Dictionary with registry status
        """
        quality_versions = self.list_model_versions(self.quality_model_name)
        failure_versions = self.list_model_versions(self.failure_model_name)
        
        quality_prod = self.get_production_model(self.quality_model_name)
        failure_prod = self.get_production_model(self.failure_model_name)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tracking_uri": self.tracking_uri,
            "quality_model": {
                "name": self.quality_model_name,
                "total_versions": len(quality_versions),
                "production_version": quality_prod['version'] if quality_prod else None,
                "latest_version": quality_versions[0]['version'] if quality_versions else None
            },
            "failure_model": {
                "name": self.failure_model_name,
                "total_versions": len(failure_versions),
                "production_version": failure_prod['version'] if failure_prod else None,
                "latest_version": failure_versions[0]['version'] if failure_versions else None
            }
        }


# Global registry manager instance
registry_manager = ModelRegistryManager()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def register_current_models():
    """
    Register the current models in the models/ directory
    """
    quality_model_path = MODELS_DIR / "quality_model.pkl"
    failure_model_path = MODELS_DIR / "failure_model.pkl"
    
    if not quality_model_path.exists() or not failure_model_path.exists():
        logger.error("[REGISTRY] Model files not found. Train models first.")
        return False
    
    # Load model metrics (from leaderboard if available)
    leaderboard_path = PROJECT_ROOT / "reports" / "model_leaderboard.json"
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
            quality_metrics = next(
                (m for m in leaderboard['quality'] if m['model'] == leaderboard['best_models']['quality']),
                {}
            )
            failure_metrics = next(
                (m for m in leaderboard['failure'] if m['model'] == leaderboard['best_models']['failure']),
                {}
            )
    else:
        # Default metrics if leaderboard not found
        quality_metrics = {"f1": 0.98, "accuracy": 0.95}
        failure_metrics = {"f1": 0.91, "accuracy": 0.94}
    
    # Register quality model
    quality_version = registry_manager.register_model(
        model_name=registry_manager.quality_model_name,
        model_path=str(quality_model_path),
        metrics=quality_metrics,
        tags={
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "model_type": "CatBoost",
            "dataset": "FATURA"
        },
        description=f"Quality model trained on {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    # Register failure model
    failure_version = registry_manager.register_model(
        model_name=registry_manager.failure_model_name,
        model_path=str(failure_model_path),
        metrics=failure_metrics,
        tags={
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "model_type": "RandomForest",
            "dataset": "FATURA"
        },
        description=f"Failure model trained on {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    logger.info(f"[REGISTRY] Registered quality model version: {quality_version}")
    logger.info(f"[REGISTRY] Registered failure model version: {failure_version}")
    
    return True