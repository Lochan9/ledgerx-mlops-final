"""
LedgerX - Secret Manager Integration
=====================================

Securely manages secrets using Google Cloud Secret Manager.
Falls back to environment variables for local development.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Check if running in GCP
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"
IS_GCP = os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

# Try to import Secret Manager client
SECRET_MANAGER_AVAILABLE = False
if IS_PRODUCTION or IS_GCP:
    try:
        from google.cloud import secretmanager
        SECRET_MANAGER_AVAILABLE = True
        logger.info("✅ Secret Manager client available")
    except ImportError:
        logger.warning("⚠️ google-cloud-secret-manager not installed")

class SecretManager:
    """Manages secrets from Google Cloud Secret Manager with local fallback"""
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.use_secret_manager = SECRET_MANAGER_AVAILABLE and IS_PRODUCTION
        
        if self.use_secret_manager:
            self.client = secretmanager.SecretManagerServiceClient()
            logger.info(f"✅ Secret Manager initialized for project: {self.project_id}")
        else:
            self.client = None
            logger.info("📝 Using environment variables for secrets")
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_id: str, version: str = "latest") -> Optional[str]:
        """Get secret value from Secret Manager or environment variables"""
        # Try Secret Manager first (production)
        if self.use_secret_manager and self.project_id:
            try:
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                response = self.client.access_secret_version(request={"name": name})
                secret_value = response.payload.data.decode("UTF-8")
                logger.debug(f"✅ Retrieved secret: {secret_id}")
                return secret_value
            except Exception as e:
                logger.error(f"❌ Failed to get secret {secret_id}: {e}")
        
        # Fallback to environment variables
        env_var_name = secret_id.upper().replace("-", "_")
        secret_value = os.getenv(env_var_name)
        
        if secret_value:
            logger.debug(f"✅ Retrieved secret from env: {env_var_name}")
        else:
            logger.warning(f"⚠️ Secret not found: {secret_id}")
        
        return secret_value
    
    def get_required_secret(self, secret_id: str, version: str = "latest") -> str:
        """Get required secret or raise error"""
        secret = self.get_secret(secret_id, version)
        if secret is None:
            raise ValueError(f"Required secret not found: {secret_id}")
        return secret

# Global instance
_secret_manager = None

def get_secret_manager() -> SecretManager:
    """Get global SecretManager instance"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager

def get_secret(secret_id: str, version: str = "latest") -> Optional[str]:
    """Convenience function to get a secret"""
    return get_secret_manager().get_secret(secret_id, version)

def get_required_secret(secret_id: str, version: str = "latest") -> str:
    """Convenience function to get a required secret"""
    return get_secret_manager().get_required_secret(secret_id, version)

def get_jwt_secret() -> str:
    """Get JWT secret key"""
    return get_required_secret("jwt-secret-key")

def get_database_password() -> str:
    """Get database password"""
    return get_required_secret("database-password")

def get_slack_webhook() -> Optional[str]:
    """Get Slack webhook URL"""
    return get_secret("slack-webhook-url")

def get_email_config() -> dict:
    """Get email configuration"""
    return {
        "from": get_secret("email-from"),
        "password": get_secret("email-password"),
        "to": get_secret("email-to"),
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587"))
    }
