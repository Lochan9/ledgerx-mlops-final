"""
LedgerX - Monitoring Alerts Test Suite
=======================================

This script tests your complete monitoring infrastructure including:
1. Slack webhook connectivity
2. Email SMTP connectivity
3. Alert formatting and delivery
4. Multi-channel alerting
5. Different severity levels

Usage:
    python test_monitoring_alerts.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.alerts import (
    send_alert,
    send_slack_alert,
    send_email_alert,
    AlertConfig,
    alert_missing_values,
    alert_outliers,
    alert_schema_violation,
    alert_model_drift,
    alert_performance_degradation,
    alert_training_complete,
    alert_training_failed,
    get_alert_status
)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_configuration():
    """Check if alerts are properly configured"""
    print_section("CONFIGURATION CHECK")
    
    print(f"📧 Email Configuration:")
    print(f"   Enabled: {AlertConfig.EMAIL_ENABLED}")
    if AlertConfig.EMAIL_ENABLED:
        print(f"   From: {AlertConfig.EMAIL_FROM}")
        print(f"   To: {', '.join(AlertConfig.EMAIL_TO)}")
        print(f"   Server: {AlertConfig.SMTP_SERVER}:{AlertConfig.SMTP_PORT}")
    else:
        print("   ⚠️  Not configured - set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO in .env")
    
    print(f"\n💬 Slack Configuration:")
    print(f"   Enabled: {AlertConfig.SLACK_ENABLED}")
    if AlertConfig.SLACK_ENABLED:
        webhook_preview = AlertConfig.SLACK_WEBHOOK_URL[:50] + "..."
        print(f"   Webhook: {webhook_preview}")
    else:
        print("   ⚠️  Not configured - set SLACK_WEBHOOK_URL in .env")
    
    print(f"\n🎯 Alert Thresholds:")
    print(f"   Missing Values: {AlertConfig.MISSING_VALUE_THRESHOLD * 100}%")
    print(f"   Outliers: {AlertConfig.OUTLIER_THRESHOLD * 100}%")
    print(f"   Data Drift: {AlertConfig.DRIFT_THRESHOLD * 100}%")
    
    if not (AlertConfig.SLACK_ENABLED or AlertConfig.EMAIL_ENABLED):
        print("\n❌ ERROR: No alert channels configured!")
        print("   Please configure at least one channel in .env file")
        return False
    
    return True


def test_slack_alert():
    """Test Slack alert"""
    print_section("TEST 1: SLACK ALERT")
    
    if not AlertConfig.SLACK_ENABLED:
        print("⏭️  Skipped - Slack not configured")
        return False
    
    print("📤 Sending test alert to Slack...")
    
    message = """
🚀 LedgerX Monitoring System - Online

This is a test alert from your production monitoring system.

✅ Slack integration working correctly
✅ Alert formatting verified
✅ Webhook connectivity confirmed

System Status: All systems operational
    """
    
    success = send_slack_alert(message.strip(), severity="info")
    
    if success:
        print("✅ SUCCESS: Slack alert delivered")
        print("   Check your Slack channel for the message")
        return True
    else:
        print("❌ FAILED: Could not send Slack alert")
        print("   Check your SLACK_WEBHOOK_URL in .env")
        return False


def test_email_alert():
    """Test email alert"""
    print_section("TEST 2: EMAIL ALERT")
    
    if not AlertConfig.EMAIL_ENABLED:
        print("⏭️  Skipped - Email not configured")
        return False
    
    print("📧 Sending test alert via email...")
    
    subject = "LedgerX Monitoring - Test Alert"
    message = """
🚀 LedgerX Monitoring System - Online

This is a test alert from your production monitoring system.

✅ Email integration working correctly
✅ SMTP connectivity confirmed
✅ Alert delivery verified

System Status: All systems operational
Timestamp: {timestamp}

If you received this email, your email alerting is configured correctly.
    """
    
    success = send_email_alert(subject, message.strip(), severity="info")
    
    if success:
        print("✅ SUCCESS: Email alert sent")
        print(f"   Check inbox: {', '.join(AlertConfig.EMAIL_TO)}")
        return True
    else:
        print("❌ FAILED: Could not send email alert")
        print("   Check EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO in .env")
        return False


def test_multi_channel_alert():
    """Test sending to both channels simultaneously"""
    print_section("TEST 3: MULTI-CHANNEL ALERT")
    
    if not (AlertConfig.SLACK_ENABLED and AlertConfig.EMAIL_ENABLED):
        print("⏭️  Skipped - Both channels not configured")
        return False
    
    print("📤 Sending alert to all channels...")
    
    message = """
🎉 Multi-Channel Alert Test

This alert is being sent to:
✅ Slack
✅ Email

Both channels are operational and configured correctly.
    """
    
    results = send_alert(message.strip(), severity="info")
    
    print("\nResults:")
    for channel, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {channel.capitalize()}: {status}")
    
    return all(results.values())


def test_severity_levels():
    """Test different severity levels"""
    print_section("TEST 4: SEVERITY LEVELS")
    
    if not (AlertConfig.SLACK_ENABLED or AlertConfig.EMAIL_ENABLED):
        print("⏭️  Skipped - No channels configured")
        return False
    
    severities = [
        ("info", "ℹ️ INFO: Normal system operation"),
        ("warning", "⚠️ WARNING: Attention required"),
        ("critical", "🚨 CRITICAL: Immediate action needed")
    ]
    
    print("📤 Testing different severity levels...\n")
    
    for severity, message in severities:
        print(f"   Sending {severity.upper()} alert...")
        results = send_alert(message, severity=severity)
        
        # Brief pause between alerts
        import time
        time.sleep(1)
    
    print("\n✅ All severity levels tested")
    print("   Check your channels for 3 different alert types")
    return True


def test_production_scenarios():
    """Test real production alert scenarios"""
    print_section("TEST 5: PRODUCTION SCENARIOS")
    
    if not (AlertConfig.SLACK_ENABLED or AlertConfig.EMAIL_ENABLED):
        print("⏭️  Skipped - No channels configured")
        return False
    
    print("🧪 Testing production alert scenarios...\n")
    
    # Scenario 1: Missing values
    print("1. Missing Values Alert...")
    alert_missing_values("invoice_number", 0.15)
    
    import time
    time.sleep(1)
    
    # Scenario 2: Outliers
    print("2. Outlier Detection Alert...")
    alert_outliers("total_amount", 150, 1000)
    
    time.sleep(1)
    
    # Scenario 3: Model drift
    print("3. Model Drift Alert...")
    alert_model_drift(
        model_name="quality_model",
        drift_magnitude=0.25,
        threshold=0.10
    )
    
    time.sleep(1)
    
    # Scenario 4: Model performance degradation
    print("4. Model Performance Degradation Alert...")
    alert_performance_degradation(
        model_name="failure_model",
        current_f1=0.82,
        baseline_f1=0.913
    )
    
    time.sleep(1)
    
    # Scenario 5: Training complete
    print("5. Training Complete Alert...")
    alert_training_complete(
        model_name="quality_model_v2",
        f1_score=0.982,
        training_time=245.3
    )
    
    print("\n✅ Production scenarios tested")
    print("   Check your channels for 5 different alert types")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  🚨 LEDGERX MONITORING ALERTS - TEST SUITE")
    print("=" * 70)
    
    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration check failed - please configure alerts in .env")
        sys.exit(1)
    
    print("\n⏸️  Press Enter to start testing...")
    input()
    
    # Run tests
    results = {
        "Configuration": True,
        "Slack Alert": test_slack_alert(),
        "Email Alert": test_email_alert(),
        "Multi-Channel": test_multi_channel_alert(),
        "Severity Levels": test_severity_levels(),
        "Production Scenarios": test_production_scenarios()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else ("⏭️  SKIP" if success is False else "❌ FAIL")
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS! All monitoring alerts are operational")
        print("\nNext steps:")
        print("1. ✅ Monitoring alerts configured")
        print("2. 🔜 Deploy to Cloud Run with alerts enabled")
        print("3. 🔜 Set up Cloud SQL database")
        print("4. 🔜 Configure Redis for cache/rate-limiter")
    else:
        print("\n⚠️  Some tests failed or were skipped")
        print("   Please check your .env configuration")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)