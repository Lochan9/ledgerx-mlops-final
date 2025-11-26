"""
LedgerX - Anomaly Detection & Alerting System
==============================================

Multi-channel alerting for:
- Data quality issues
- Model performance degradation
- Pipeline failures
- Schema violations
"""

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, List
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Find .env in project root (2 levels up from this file)
    env_path = Path(__file__).resolve().parents[2] / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"[DEBUG] Loaded .env from: {env_path}")
    print(f"[DEBUG] Webhook found: {bool(os.getenv('SLACK_WEBHOOK_URL'))}")
except ImportError:
    print("[WARNING] python-dotenv not installed. Using environment variables only.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_alerts")


# ============================================================================
# CONFIGURATION
# ============================================================================

class AlertConfig:
    """Alert configuration from environment variables"""
    
    # Slack
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_ENABLED = bool(SLACK_WEBHOOK_URL)
    
    # Email
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_FROM = os.getenv("EMAIL_FROM", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "").split(",") if os.getenv("EMAIL_TO") else []
    EMAIL_ENABLED = bool(EMAIL_FROM and EMAIL_PASSWORD and EMAIL_TO)
    
    # Alert thresholds
    MISSING_VALUE_THRESHOLD = float(os.getenv("MISSING_VALUE_THRESHOLD", "0.05"))  # 5%
    OUTLIER_THRESHOLD = float(os.getenv("OUTLIER_THRESHOLD", "0.10"))  # 10%
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.10"))  # 10%


# ============================================================================
# SLACK ALERTING
# ============================================================================

def send_slack_alert(message: str, severity: str = "info") -> bool:
    """
    Send alert to Slack
    
    Args:
        message: Alert message
        severity: 'info', 'warning', 'critical'
        
    Returns:
        True if successful, False otherwise
    """
    if not AlertConfig.SLACK_ENABLED:
        logger.warning("[SLACK] Slack not configured - skipping alert")
        return False
    
    # Emoji based on severity
    emoji_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨"
    }
    
    emoji = emoji_map.get(severity, "📢")
    
    payload = {
        "text": f"{emoji} *LedgerX Alert* ({severity.upper()})",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} LedgerX Alert"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:*\n{message}"
                }
            }
        ]
    }
    
    try:
        response = requests.post(
            AlertConfig.SLACK_WEBHOOK_URL,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"[SLACK] Alert sent successfully")
            return True
        else:
            logger.error(f"[SLACK] Failed to send alert: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"[SLACK] Error sending alert: {e}")
        return False


# ============================================================================
# EMAIL ALERTING
# ============================================================================

def send_email_alert(subject: str, message: str, severity: str = "info") -> bool:
    """
    Send alert via email
    
    Args:
        subject: Email subject
        message: Email body
        severity: 'info', 'warning', 'critical'
        
    Returns:
        True if successful, False otherwise
    """
    if not AlertConfig.EMAIL_ENABLED:
        logger.warning("[EMAIL] Email not configured - skipping alert")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[LedgerX {severity.upper()}] {subject}"
        msg['From'] = AlertConfig.EMAIL_FROM
        msg['To'] = ", ".join(AlertConfig.EMAIL_TO)
        
        # HTML body
        html = f"""
        <html>
          <body>
            <h2 style="color: {'red' if severity == 'critical' else 'orange' if severity == 'warning' else 'blue'};">
              LedgerX Alert - {severity.upper()}
            </h2>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Severity:</strong> {severity.upper()}</p>
            <hr>
            <p>{message.replace(chr(10), '<br>')}</p>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Send email
        with smtplib.SMTP(AlertConfig.SMTP_SERVER, AlertConfig.SMTP_PORT) as server:
            server.starttls()
            server.login(AlertConfig.EMAIL_FROM, AlertConfig.EMAIL_PASSWORD)
            server.sendmail(
                AlertConfig.EMAIL_FROM,
                AlertConfig.EMAIL_TO,
                msg.as_string()
            )
        
        logger.info(f"[EMAIL] Alert sent to {len(AlertConfig.EMAIL_TO)} recipients")
        return True
        
    except Exception as e:
        logger.error(f"[EMAIL] Error sending alert: {e}")
        return False


# ============================================================================
# UNIFIED ALERTING
# ============================================================================

def send_alert(message: str, severity: str = "info", channels: List[str] = None) -> Dict[str, bool]:
    """
    Send alert to multiple channels
    
    Args:
        message: Alert message
        severity: 'info', 'warning', 'critical'
        channels: List of channels ['slack', 'email'] or None for all
        
    Returns:
        Dictionary with success status for each channel
    """
    if channels is None:
        channels = []
        if AlertConfig.SLACK_ENABLED:
            channels.append('slack')
        if AlertConfig.EMAIL_ENABLED:
            channels.append('email')
    
    results = {}
    
    if 'slack' in channels:
        results['slack'] = send_slack_alert(message, severity)
    
    if 'email' in channels:
        subject = message.split('\n')[0][:50]  # First line as subject
        results['email'] = send_email_alert(subject, message, severity)
    
    return results


# ============================================================================
# ANOMALY DETECTION ALERTS
# ============================================================================

def alert_missing_values(column: str, missing_pct: float, threshold: float = None):
    """Alert when missing values exceed threshold"""
    if threshold is None:
        threshold = AlertConfig.MISSING_VALUE_THRESHOLD
    
    if missing_pct > threshold:
        message = f"""
🔍 Data Quality Alert: Missing Values

Column: {column}
Missing: {missing_pct*100:.2f}%
Threshold: {threshold*100:.2f}%

Action Required: Investigate data source and imputation strategy.
"""
        send_alert(message, severity="warning")


def alert_outliers(column: str, outlier_count: int, total_count: int):
    """Alert when outliers are detected"""
    outlier_pct = outlier_count / total_count
    
    if outlier_pct > AlertConfig.OUTLIER_THRESHOLD:
        message = f"""
📊 Data Quality Alert: Outliers Detected

Column: {column}
Outliers: {outlier_count} ({outlier_pct*100:.2f}%)
Total Records: {total_count}

Action Required: Review data preprocessing and outlier handling.
"""
        send_alert(message, severity="warning")


def alert_schema_violation(details: str):
    """Alert on schema validation failures"""
    message = f"""
⚠️ Schema Violation Detected

{details}

Action Required: Check data pipeline for schema changes.
"""
    send_alert(message, severity="critical")


def alert_model_drift(model_name: str, drift_magnitude: float, threshold: float):
    """Alert when model drift is detected"""
    message = f"""
📉 Model Drift Detected

Model: {model_name}
Drift Magnitude: {drift_magnitude:.4f}
Threshold: {threshold:.4f}

Action Required: Consider retraining the model with recent data.
"""
    send_alert(message, severity="critical")


def alert_performance_degradation(model_name: str, current_f1: float, baseline_f1: float):
    """Alert when model performance degrades"""
    degradation = (baseline_f1 - current_f1) / baseline_f1 * 100
    
    message = f"""
⚠️ Model Performance Degradation

Model: {model_name}
Current F1: {current_f1:.4f}
Baseline F1: {baseline_f1:.4f}
Degradation: {degradation:.2f}%

Action Required: Investigate and consider retraining.
"""
    send_alert(message, severity="critical")


def alert_training_complete(model_name: str, f1_score: float, training_time: float):
    """Alert when model training completes"""
    message = f"""
✅ Model Training Complete

Model: {model_name}
F1 Score: {f1_score:.4f}
Training Time: {training_time:.2f} seconds

Model is ready for evaluation and deployment.
"""
    send_alert(message, severity="info")


def alert_training_failed(model_name: str, error: str):
    """Alert when model training fails"""
    message = f"""
🚨 Model Training Failed

Model: {model_name}
Error: {error}

Action Required: Check logs and investigate failure.
"""
    send_alert(message, severity="critical")


# ============================================================================
# CONFIGURATION HELPER
# ============================================================================

def configure_alerts(slack_webhook: Optional[str] = None, email_config: Optional[Dict] = None):
    """
    Configure alert channels
    
    Args:
        slack_webhook: Slack webhook URL
        email_config: Dict with 'from', 'password', 'to', 'smtp_server', 'smtp_port'
    """
    if slack_webhook:
        os.environ["SLACK_WEBHOOK_URL"] = slack_webhook
        AlertConfig.SLACK_WEBHOOK_URL = slack_webhook
        AlertConfig.SLACK_ENABLED = True
        logger.info("[CONFIG] Slack alerts enabled")
    
    if email_config:
        os.environ["EMAIL_FROM"] = email_config.get('from', '')
        os.environ["EMAIL_PASSWORD"] = email_config.get('password', '')
        os.environ["EMAIL_TO"] = ",".join(email_config.get('to', []))
        os.environ["SMTP_SERVER"] = email_config.get('smtp_server', 'smtp.gmail.com')
        os.environ["SMTP_PORT"] = str(email_config.get('smtp_port', 587))
        
        AlertConfig.EMAIL_FROM = email_config.get('from', '')
        AlertConfig.EMAIL_PASSWORD = email_config.get('password', '')
        AlertConfig.EMAIL_TO = email_config.get('to', [])
        AlertConfig.SMTP_SERVER = email_config.get('smtp_server', 'smtp.gmail.com')
        AlertConfig.SMTP_PORT = email_config.get('smtp_port', 587)
        AlertConfig.EMAIL_ENABLED = True
        logger.info("[CONFIG] Email alerts enabled")


def get_alert_status() -> Dict:
    """Get current alert configuration status"""
    return {
        "slack_enabled": AlertConfig.SLACK_ENABLED,
        "slack_webhook_configured": bool(AlertConfig.SLACK_WEBHOOK_URL),
        "email_enabled": AlertConfig.EMAIL_ENABLED,
        "email_from": AlertConfig.EMAIL_FROM if AlertConfig.EMAIL_FROM else "Not configured",
        "thresholds": {
            "missing_values": AlertConfig.MISSING_VALUE_THRESHOLD,
            "outliers": AlertConfig.OUTLIER_THRESHOLD,
            "drift": AlertConfig.DRIFT_THRESHOLD
        }
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LedgerX Alerting System - Test")
    print("="*70)
    
    # Show status
    status = get_alert_status()
    print(f"\nSlack Enabled: {status['slack_enabled']}")
    print(f"Slack Webhook Configured: {status['slack_webhook_configured']}")
    print(f"Email Enabled: {status['email_enabled']}")
    print(f"Email From: {status['email_from']}")
    print(f"\nThresholds:")
    for key, value in status['thresholds'].items():
        print(f"  {key}: {value*100:.1f}%")
    
    # Test alert (will only work if configured)
    print("\nSending test alert...")
    result = send_alert(
        message="🧪 This is a test alert from LedgerX!\n\nIf you see this, alerts are working correctly.",
        severity="info"
    )
    
    print(f"\nResults: {result}")
    
    if result.get('slack'):
        print("✅ Check your Slack channel for the test message!")
    elif status['slack_webhook_configured']:
        print("⚠️  Slack configured but send failed - check webhook URL")
    else:
        print("ℹ️  Slack not configured - set SLACK_WEBHOOK_URL in .env file")
    
    print("="*70)