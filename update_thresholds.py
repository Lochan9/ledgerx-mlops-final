# Script to update drift thresholds to be more sensitive
import re

file_path = 'src/monitoring/drift_threshold_checker.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace thresholds
content = re.sub(
    r'DRIFT_SCORE_THRESHOLD = 0\.15',
    'DRIFT_SCORE_THRESHOLD = 0.05',
    content
)
content = re.sub(
    r'FEATURE_DRIFT_THRESHOLD = 0\.30',
    'FEATURE_DRIFT_THRESHOLD = 0.05',
    content
)

# Update comments
content = content.replace(
    '# 15% of features drifting',
    '# 5% of features drifting (MORE SENSITIVE FOR DEMO)'
)
content = content.replace(
    '# 30% drift in any feature',
    '# 5% drift in any feature (MORE SENSITIVE FOR DEMO)'
)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Updated drift thresholds!')
print('   DRIFT_SCORE_THRESHOLD: 0.15 → 0.05 (now triggers at 5% of features)')
print('   FEATURE_DRIFT_THRESHOLD: 0.30 → 0.05 (now triggers at 5% per feature)')
print('\nNow run: python test_drift_detection.py')
