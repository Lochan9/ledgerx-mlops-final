import requests
import json

webhook_url = 'https://hooks.slack.com/services/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'

message = {
    'text': '🚀 LedgerX MLOps Alert',
    'blocks': [
        {'type': 'header', 'text': {'type': 'plain_text', 'text': '⚠️ Model Retraining Triggered'}},
        {'type': 'section', 'text': {'type': 'mrkdwn', 'text': '*Reason:* Data drift detected\n*Drift Score:* 9.1%\n*Features:* blur_score, ocr_confidence\n*Action:* Retraining started'}}
    ]
}

response = requests.post(webhook_url, json=message)
print(f'Status: {response.status_code}')
print('✅ Sent!' if response.status_code == 200 else f'Failed: {response.text}')
