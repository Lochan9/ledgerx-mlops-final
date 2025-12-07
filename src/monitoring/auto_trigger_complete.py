"""
Auto-Trigger Retraining with Notifications
Connects: Drift Detection → GitHub Actions → Slack/Email Alerts
"""

import json
import requests
import os
from pathlib import Path
from datetime import datetime

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your-github-token")
REPO_OWNER = "Lochan9"
REPO_NAME = "ledgerx-mlops-final"
WORKFLOW_FILE = "train-models.yml"
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")

def trigger_github_workflow():
    """Trigger GitHub Actions workflow via API"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {"ref": "main"}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 204:
            print("✅ GitHub Actions workflow triggered successfully")
            return True
        else:
            print(f"❌ Failed to trigger workflow: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error triggering workflow: {e}")
        return False

def send_slack_notification(message, drift_score, drifted_features):
    """Send Slack notification"""
    if not SLACK_WEBHOOK:
        print("⚠️ Slack webhook not configured")
        return False
    
    payload = {
        "text": "🔄 LedgerX Model Retraining Triggered",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔄 Model Retraining Triggered"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Drift Score:*\n{drift_score:.1%}"},
                    {"type": "mrkdwn", "text": f"*Drifted Features:*\n{len(drifted_features)}"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"},
                    {"type": "mrkdwn", "text": f"*Status:*\nRetraining initiated"}
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Drifted Features:*\n{', '.join(drifted_features)}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Workflow"},
                        "url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/actions"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(SLACK_WEBHOOK, json=payload)
        if response.status_code == 200:
            print("✅ Slack notification sent")
            return True
        else:
            print(f"❌ Slack failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Slack error: {e}")
        return False

def send_email_notification(drift_score, drifted_features):
    """Send email notification (using SendGrid or SMTP)"""
    # Simplified: Just log for now
    # In production: integrate SendGrid, AWS SES, or SMTP
    
    email_content = f"""
Subject: LedgerX Model Retraining Triggered

Dear Team,

Data drift has been detected in the LedgerX invoice intelligence system:

- Drift Score: {drift_score:.1%}
- Drifted Features: {', '.join(drifted_features)}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Automated retraining has been initiated.

Monitor progress: https://github.com/{REPO_OWNER}/{REPO_NAME}/actions

Best regards,
LedgerX MLOps System
"""
    
    # Save to file (in production, send via SMTP)
    email_log = Path("reports/notifications/emails.txt")
    email_log.parent.mkdir(exist_ok=True, parents=True)
    
    with open(email_log, "a") as f:
        f.write(email_content + "\n" + "="*70 + "\n")
    
    print(f"📧 Email notification logged: {email_log}")
    return True

def check_and_trigger_retraining():
    """Main function: Check drift → Trigger retraining → Notify"""
    
    print("="*70)
    print("AUTO-RETRAIN TRIGGER")
    print("="*70)
    
    # Import drift checker
    from drift_threshold_checker import DriftThresholdChecker
    
    # Check drift
    print("\n1. Checking for data drift...")
    checker = DriftThresholdChecker(
        reference_data_path="../../data/processed/quality_training.csv",
        production_data_path="../../data/production/simulated_drift.csv"
    )
    
    result = checker.detect_drift()
    
    drift_score = result['drift_score']
    drifted_features = result['drifted_features']
    should_retrain = result['should_retrain']
    
    print(f"   Drift Score: {drift_score:.1%}")
    print(f"   Drifted Features: {drifted_features}")
    print(f"   Should Retrain: {should_retrain}")
    
    # Decision
    DRIFT_THRESHOLD = 0.08  # 15%
    
    if drift_score >= DRIFT_THRESHOLD or should_retrain:
        print(f"\n⚠️ DRIFT THRESHOLD EXCEEDED ({drift_score:.1%} >= {DRIFT_THRESHOLD:.0%})")
        print("\n2. Triggering automated retraining...")
        
        # Trigger GitHub Actions
        workflow_triggered = trigger_github_workflow()
        
        # Send notifications
        print("\n3. Sending notifications...")
        send_slack_notification(
            message="Data drift detected",
            drift_score=drift_score,
            drifted_features=drifted_features
        )
        send_email_notification(drift_score, drifted_features)
        
        # Log event
        log_path = Path("reports/retraining_triggers.json")
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        events = []
        if log_path.exists():
            with open(log_path) as f:
                events = json.load(f)
        
        events.append({
            "timestamp": datetime.now().isoformat(),
            "drift_score": float(drift_score),
            "drifted_features": drifted_features,
            "workflow_triggered": workflow_triggered,
            "notifications_sent": True
        })
        
        with open(log_path, 'w') as f:
            json.dump(events, f, indent=2)
        
        print(f"\n✅ Retraining event logged: {log_path}")
        print("\n" + "="*70)
        print("✅ AUTO-RETRAIN COMPLETE")
        print("="*70)
        
    else:
        print(f"\n✅ No action needed - drift below threshold ({drift_score:.1%} < {DRIFT_THRESHOLD:.0%})")

if __name__ == "__main__":
    check_and_trigger_retraining()
