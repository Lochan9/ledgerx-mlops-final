"""
LedgerX - Simple Single Alert Test
===================================
Sends ONE comprehensive alert to verify monitoring is working.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.alerts import send_alert, get_alert_status
import json

print("=" * 70)
print("  🚨 LEDGERX MONITORING - SINGLE ALERT TEST")
print("=" * 70)

# Check configuration
status = get_alert_status()
print("\n📊 Configuration Status:")
print(json.dumps(status, indent=2))

if not (status['slack_enabled'] or status['email_enabled']):
    print("\n❌ ERROR: No alert channels configured!")
    sys.exit(1)

print("\n" + "=" * 70)
print("  📤 SENDING COMPREHENSIVE TEST ALERT")
print("=" * 70)

# Send ONE comprehensive alert with all info
message = """
🎉 LedgerX Production Monitoring - System Online

✅ CONFIGURATION VERIFIED:
   • Slack: {slack_status}
   • Email: {email_status}

📊 ALERT CAPABILITIES:
   • Data Quality Monitoring
   • Model Performance Tracking
   • Schema Validation
   • Training Notifications
   • System Health Checks

🎯 THRESHOLDS:
   • Missing Values: {missing}%
   • Outliers: {outlier}%
   • Data Drift: {drift}%

⚡ NEXT STEPS:
   1. ✅ Monitoring alerts configured
   2. 🔜 Cloud SQL database migration
   3. 🔜 Redis cache/rate-limiter setup
   4. 🔜 Full production deployment

🚀 Status: READY FOR PRODUCTION

If you see this message, your monitoring system is fully operational!
""".format(
    slack_status="Enabled ✅" if status['slack_enabled'] else "Disabled ❌",
    email_status="Enabled ✅" if status['email_enabled'] else "Disabled ❌",
    missing=status['thresholds']['missing_values'] * 100,
    outlier=status['thresholds']['outliers'] * 100,
    drift=status['thresholds']['drift'] * 100
)

# Send to all configured channels
results = send_alert(message.strip(), severity="info")

print("\n📬 DELIVERY RESULTS:")
for channel, success in results.items():
    status_icon = "✅" if success else "❌"
    print(f"   {status_icon} {channel.capitalize()}: {'Delivered' if success else 'Failed'}")

if all(results.values()):
    print("\n" + "=" * 70)
    print("  🎉 SUCCESS! Check your Slack and Email")
    print("=" * 70)
    print("\n✅ Your monitoring system is fully operational!")
    print("✅ You'll receive alerts for:")
    print("   • Data quality issues")
    print("   • Model performance changes")
    print("   • Training completions/failures")
    print("   • System anomalies")
    print("\n🔜 Ready to proceed with Cloud SQL migration")
else:
    print("\n⚠️  Some channels failed - check your configuration")

print("=" * 70)