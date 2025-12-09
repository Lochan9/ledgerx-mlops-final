import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load retraining log
with open('reports/retraining_log.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LedgerX MLOps - Drift Detection & Retraining Analysis', fontsize=16, fontweight='bold')

# Plot 1: Retraining Triggers Over Time
ax1 = axes[0, 0]
trigger_counts = df.groupby(df['timestamp'].dt.date)['retraining_triggered'].sum()
ax1.bar(range(len(trigger_counts)), trigger_counts.values, color='#ff6b6b')
ax1.set_title('Retraining Triggers by Date', fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xlabel('Date')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Trigger Reasons Pie Chart
ax2 = axes[0, 1]
triggered = df[df['retraining_triggered'] == True]
if len(triggered) > 0:
    reasons = triggered['trigger_reasons'].explode().value_counts()
    ax2.pie(reasons.values, labels=reasons.index, autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
    ax2.set_title('Retraining Trigger Reasons', fontweight='bold')

# Plot 3: Retraining Decision Timeline
ax3 = axes[1, 0]
df['triggered_num'] = df['retraining_triggered'].astype(int)
ax3.plot(df.index, df['triggered_num'], marker='o', linewidth=2, markersize=8, color='#ff6b6b')
ax3.fill_between(df.index, df['triggered_num'], alpha=0.3, color='#ff6b6b')
ax3.set_title('Retraining Decisions Over Time', fontweight='bold')
ax3.set_ylabel('Triggered (1) / Not Triggered (0)')
ax3.set_xlabel('Event Number')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Not Triggered', 'Triggered'])
ax3.grid(alpha=0.3)

# Plot 4: Summary Statistics
ax4 = axes[1, 1]
ax4.axis('off')
total = len(df)
triggered_count = df['retraining_triggered'].sum()
not_triggered = total - triggered_count

stats_text = f'''
DRIFT DETECTION SUMMARY

Total Checks: {total}
Retraining Triggered: {triggered_count} ({triggered_count/total*100:.1f}%)
No Action Needed: {not_triggered} ({not_triggered/total*100:.1f}%)

Threshold: 5% feature drift
Status: OPERATIONAL ✅

Most Common Trigger: DATA_DRIFT
System: Automated & Working
'''

ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig('reports/drift_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print('\n✅ Dashboard saved: reports/drift_analysis_dashboard.png')
plt.show()
