"""
LedgerX - Notifications & Alerts System
========================================

Comprehensive notification system for:
- Training completion/failure
- Model performance alerts
- Pipeline status updates
- Bias detection alerts
- Deployment notifications

Supports: Email, Slack, Console logging
"""

import os
import json
import smtplib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("ledgerx_notifications")


class NotificationManager:
    """Manages all notifications and alerts for LedgerX"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize notification manager
        
        Args:
            config_path: Path to notification config file (JSON)
        """
        self.config = self._load_config(config_path)
        self.enabled_channels = self.config.get('enabled_channels', ['console'])
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load notification configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'enabled_channels': ['console'],
            'email': {
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'sender_email': os.getenv('SENDER_EMAIL', ''),
                'sender_password': os.getenv('SENDER_PASSWORD', ''),
                'recipient_emails': os.getenv('RECIPIENT_EMAILS', '').split(',')
            },
            'slack': {
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL', '')
            },
            'thresholds': {
                'min_quality_f1': 0.90,
                'min_failure_f1': 0.85,
                'max_performance_drop': 0.05,
                'max_bias_disparity': 0.05
            }
        }
    
    def notify(self, 
               message_type: str,
               title: str, 
               message: str,
               metrics: Optional[Dict] = None,
               severity: str = 'info'):
        """
        Send notification through enabled channels
        
        Args:
            message_type: Type of notification (training, validation, alert, etc.)
            title: Notification title
            message: Notification message
            metrics: Optional metrics dictionary
            severity: info, warning, critical
        """
        # Log to console always
        self._log_console(title, message, metrics, severity)
        
        # Email notification
        if 'email' in self.enabled_channels:
            self._send_email(title, message, metrics, severity)
        
        # Slack notification
        if 'slack' in self.enabled_channels:
            self._send_slack(title, message, metrics, severity)
    
    def _log_console(self, title: str, message: str, 
                     metrics: Optional[Dict], severity: str):
        """Log notification to console"""
        separator = "=" * 70
        
        # Choose log level based on severity
        log_func = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.critical
        }.get(severity, logger.info)
        
        log_func(separator)
        log_func(f"{title}")
        log_func(separator)
        log_func(message)
        
        if metrics:
            log_func("\nMetrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_func(f"  {key}: {value:.4f}")
                else:
                    log_func(f"  {key}: {value}")
        
        log_func(separator)
    
    def _send_email(self, title: str, message: str, 
                    metrics: Optional[Dict], severity: str):
        """Send email notification"""
        email_config = self.config.get('email', {})
        
        # Check if email is configured
        if not email_config.get('sender_email') or not email_config.get('sender_password'):
            logger.warning("[EMAIL] Email not configured - skipping")
            return
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[LedgerX - {severity.upper()}] {title}"
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipient_emails'])
            
            # HTML body
            html_body = self._create_email_html(title, message, metrics, severity)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)
            
            logger.info(f"[EMAIL] Sent to {len(email_config['recipient_emails'])} recipients")
            
        except Exception as e:
            logger.error(f"[EMAIL] Failed to send: {e}")
    
    def _send_slack(self, title: str, message: str, 
                    metrics: Optional[Dict], severity: str):
        """Send Slack notification"""
        webhook_url = self.config.get('slack', {}).get('webhook_url')
        
        if not webhook_url:
            logger.warning("[SLACK] Webhook not configured - skipping")
            return
        
        try:
            # Choose emoji based on severity
            emoji = {
                'info': ':white_check_mark:',
                'warning': ':warning:',
                'critical': ':x:'
            }.get(severity, ':bell:')
            
            # Build Slack message
            slack_message = {
                'text': f"{emoji} *{title}*",
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f"{emoji} {title}"
                        }
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': message
                        }
                    }
                ]
            }
            
            # Add metrics if provided
            if metrics:
                metrics_text = '\n'.join([
                    f"*{k}:* {v:.4f}" if isinstance(v, float) else f"*{k}:* {v}"
                    for k, v in metrics.items()
                ])
                slack_message['blocks'].append({
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f"*Metrics:*\n{metrics_text}"
                    }
                })
            
            # Send to Slack
            response = requests.post(webhook_url, json=slack_message)
            
            if response.status_code == 200:
                logger.info("[SLACK] Notification sent successfully")
            else:
                logger.error(f"[SLACK] Failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[SLACK] Failed to send: {e}")
    
    def _create_email_html(self, title: str, message: str, 
                          metrics: Optional[Dict], severity: str) -> str:
        """Create HTML email body"""
        
        # Color based on severity
        color = {
            'info': '#28a745',      # Green
            'warning': '#ffc107',   # Yellow
            'critical': '#dc3545'   # Red
        }.get(severity, '#007bff')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; }}
                .metric {{ margin: 5px 0; }}
                .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{title}</h2>
            </div>
            <div class="content">
                <p>{message}</p>
        """
        
        if metrics:
            html += '<div class="metrics"><h3>Metrics</h3>'
            for key, value in metrics.items():
                if isinstance(value, float):
                    html += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
                else:
                    html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
            html += '</div>'
        
        html += f"""
                <div class="footer">
                    <p>LedgerX MLOps Platform</p>
                    <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    # ========================================================================
    # PRE-DEFINED NOTIFICATION TEMPLATES
    # ========================================================================
    
    def notify_training_started(self, n_models: int):
        """Notify that training has started"""
        self.notify(
            message_type='training',
            title='Training Started',
            message=f'Model training initiated for {n_models} models.',
            severity='info'
        )
    
    def notify_training_completed(self, results: Dict):
        """Notify that training completed successfully"""
        self.notify(
            message_type='training',
            title='✅ Training Completed Successfully',
            message='All models trained and evaluated successfully.',
            metrics=results,
            severity='info'
        )
    
    def notify_training_failed(self, error: str):
        """Notify that training failed"""
        self.notify(
            message_type='training',
            title='❌ Training Failed',
            message=f'Model training failed with error: {error}',
            severity='critical'
        )
    
    def notify_performance_validated(self, quality_f1: float, failure_f1: float):
        """Notify that models passed performance gates"""
        self.notify(
            message_type='validation',
            title='✅ Performance Validation Passed',
            message='Models meet or exceed performance thresholds.',
            metrics={
                'quality_f1': quality_f1,
                'quality_target': 0.90,
                'quality_margin': quality_f1 - 0.90,
                'failure_f1': failure_f1,
                'failure_target': 0.85,
                'failure_margin': failure_f1 - 0.85
            },
            severity='info'
        )
    
    def notify_performance_failed(self, quality_f1: float, failure_f1: float):
        """Alert that models failed performance gates"""
        self.notify(
            message_type='validation',
            title='❌ Performance Validation Failed',
            message='Models do not meet performance thresholds. Deployment blocked.',
            metrics={
                'quality_f1': quality_f1,
                'quality_target': 0.90,
                'failure_f1': failure_f1,
                'failure_target': 0.85
            },
            severity='critical'
        )
    
    def notify_bias_detected(self, slice_name: str, disparity: float):
        """Alert that bias was detected"""
        self.notify(
            message_type='bias',
            title='⚠️ Bias Detected',
            message=f'Performance disparity detected in {slice_name} slice.',
            metrics={
                'slice': slice_name,
                'disparity': disparity,
                'threshold': 0.05
            },
            severity='warning'
        )
    
    def notify_bias_clear(self):
        """Notify that no bias was detected"""
        self.notify(
            message_type='bias',
            title='✅ Bias Check Passed',
            message='No significant bias detected across all data slices.',
            severity='info'
        )
    
    def notify_model_registered(self, model_name: str, version: int, f1_score: float):
        """Notify that model was registered"""
        self.notify(
            message_type='registry',
            title='✅ Model Registered',
            message=f'Model {model_name} version {version} registered in MLflow.',
            metrics={
                'model': model_name,
                'version': version,
                'f1_score': f1_score
            },
            severity='info'
        )
    
    def notify_deployment_started(self, model_name: str, version: int):
        """Notify that deployment started"""
        self.notify(
            message_type='deployment',
            title='🚀 Deployment Started',
            message=f'Deploying {model_name} version {version}...',
            severity='info'
        )
    
    def notify_deployment_completed(self, model_name: str, version: int, endpoint: str):
        """Notify that deployment completed"""
        self.notify(
            message_type='deployment',
            title='✅ Deployment Completed',
            message=f'Model {model_name} v{version} deployed successfully.',
            metrics={
                'model': model_name,
                'version': version,
                'endpoint': endpoint
            },
            severity='info'
        )
    
    def notify_rollback_triggered(self, model_name: str, 
                                  failed_version: int, 
                                  rollback_version: int):
        """Notify that rollback was triggered"""
        self.notify(
            message_type='rollback',
            title='🔄 Rollback Triggered',
            message=f'Rolling back {model_name} from v{failed_version} to v{rollback_version}',
            metrics={
                'model': model_name,
                'failed_version': failed_version,
                'rollback_version': rollback_version
            },
            severity='warning'
        )


# ============================================================================
# STANDALONE NOTIFICATION FUNCTIONS
# ============================================================================

def send_slack_notification(webhook_url: str, title: str, message: str, 
                            metrics: Optional[Dict] = None, 
                            severity: str = 'info'):
    """
    Send notification to Slack
    
    Args:
        webhook_url: Slack webhook URL
        title: Message title
        message: Message body
        metrics: Optional metrics dictionary
        severity: info, warning, critical
    """
    emoji = {
        'info': ':white_check_mark:',
        'warning': ':warning:',
        'critical': ':x:'
    }.get(severity, ':bell:')
    
    payload = {
        'text': f"{emoji} *{title}*",
        'blocks': [
            {
                'type': 'header',
                'text': {
                    'type': 'plain_text',
                    'text': f"{emoji} {title}"
                }
            },
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': message
                }
            }
        ]
    }
    
    if metrics:
        metrics_text = '\n'.join([
            f"*{k}:* `{v:.4f}`" if isinstance(v, float) else f"*{k}:* `{v}`"
            for k, v in metrics.items()
        ])
        payload['blocks'].append({
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f"*Metrics:*\n{metrics_text}"
            }
        })
    
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logger.info(f"[SLACK] Notification sent: {title}")
        else:
            logger.error(f"[SLACK] Failed: {response.status_code}")
    except Exception as e:
        logger.error(f"[SLACK] Error: {e}")


def send_email_notification(smtp_config: Dict, title: str, message: str,
                            metrics: Optional[Dict] = None,
                            severity: str = 'info'):
    """
    Send email notification
    
    Args:
        smtp_config: Email configuration dictionary
        title: Email subject
        message: Email body
        metrics: Optional metrics
        severity: info, warning, critical
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[LedgerX - {severity.upper()}] {title}"
        msg['From'] = smtp_config['sender_email']
        msg['To'] = ', '.join(smtp_config['recipient_emails'])
        
        # Create HTML body
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {'#28a745' if severity == 'info' else '#dc3545'};">
                {title}
            </h2>
            <p>{message}</p>
        """
        
        if metrics:
            html += "<h3>Metrics:</h3><ul>"
            for key, value in metrics.items():
                if isinstance(value, float):
                    html += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                else:
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += f"""
            <hr>
            <p style="color: #6c757d; font-size: 12px;">
                LedgerX MLOps Platform<br>
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Send
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            server.starttls()
            server.login(smtp_config['sender_email'], smtp_config['sender_password'])
            server.send_message(msg)
        
        logger.info(f"[EMAIL] Sent to {len(smtp_config['recipient_emails'])} recipients")
        
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send: {e}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_training_notification():
    """Example: Notify on training completion"""
    
    notifier = NotificationManager()
    
    # Training completed
    notifier.notify_training_completed({
        'quality_f1': 0.9768,
        'failure_f1': 0.9134,
        'quality_best_model': 'catboost',
        'failure_best_model': 'logreg',
        'training_time': '8m 34s'
    })


def example_validation_notification():
    """Example: Notify on performance validation"""
    
    notifier = NotificationManager()
    
    # Performance validated
    notifier.notify_performance_validated(
        quality_f1=0.9768,
        failure_f1=0.9134
    )


def example_bias_notification():
    """Example: Notify on bias detection"""
    
    notifier = NotificationManager()
    
    # No bias detected
    notifier.notify_bias_clear()
    
    # Or if bias detected:
    # notifier.notify_bias_detected('blur_quality', 0.08)


def example_custom_notification():
    """Example: Send custom notification"""
    
    notifier = NotificationManager()
    
    notifier.notify(
        message_type='custom',
        title='Hyperparameter Tuning Completed',
        message='Bayesian optimization finished with 50 trials.',
        metrics={
            'best_quality_f1': 0.9768,
            'best_failure_f1': 0.9134,
            'total_trials': 100
        },
        severity='info'
    )


# ============================================================================
# INTEGRATION WITH TRAINING PIPELINE
# ============================================================================

def integrate_with_training():
    """
    Example integration with train_all_models.py
    
    Add these lines to your training script:
    """
    example_code = """
# At the start of train_all_models.py:
from src.utils.notifications import NotificationManager

notifier = NotificationManager()

def main():
    # Notify training started
    notifier.notify_training_started(n_models=6)
    
    try:
        # ... training code ...
        
        quality_results, quality_best = train_quality()
        failure_results, failure_best = train_failure()
        
        # Notify training completed
        notifier.notify_training_completed({
            'quality_f1': quality_best['f1'],
            'failure_f1': failure_best['f1'],
            'quality_best_model': quality_best['model_name'],
            'failure_best_model': failure_best['model_name']
        })
        
    except Exception as e:
        # Notify training failed
        notifier.notify_training_failed(str(e))
        raise
    """
    
    print(example_code)


# ============================================================================
# CONFIGURATION FILE TEMPLATE
# ============================================================================

def create_config_template():
    """Create notification configuration template"""
    
    config = {
        "enabled_channels": ["console", "email", "slack"],
        
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_emails": [
                "team-member1@example.com",
                "team-member2@example.com"
            ]
        },
        
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        },
        
        "thresholds": {
            "min_quality_f1": 0.90,
            "min_failure_f1": 0.85,
            "max_performance_drop": 0.05,
            "max_bias_disparity": 0.05
        }
    }
    
    with open('notification_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Configuration template created: notification_config.json")


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    # Create config template
    create_config_template()
    
    # Test notifications
    notifier = NotificationManager()
    
    # Test training notification
    notifier.notify_training_completed({
        'quality_f1': 0.9768,
        'failure_f1': 0.9134,
        'quality_best_model': 'catboost',
        'failure_best_model': 'logreg'
    })
    
    # Test performance validation
    notifier.notify_performance_validated(0.9768, 0.9134)
    
    # Test bias check
    notifier.notify_bias_clear()
    
    logger.info("✅ Notification system test complete!")