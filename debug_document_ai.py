"""
Debug script to see what Document AI actually extracts
"""

from google.cloud import documentai_v1 as documentai
from pathlib import Path

PROJECT_ID = "671429123152"
LOCATION = "us"
PROCESSOR_ID = "1903e3a537160b1f"

client = documentai.DocumentProcessorServiceClient()
processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

# Load test invoice
image_path = Path("test_ocr.jpg")
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Process
raw_document = documentai.RawDocument(
    content=image_bytes,
    mime_type="image/jpeg"
)

request = documentai.ProcessRequest(
    name=processor_name,
    raw_document=raw_document
)

result = client.process_document(request=request)
document = result.document

# Print ALL entities to see what Document AI returns
print("="*70)
print("DOCUMENT AI ENTITIES")
print("="*70)

for entity in document.entities:
    print(f"\nType: {entity.type_}")
    print(f"Text: {entity.mention_text}")
    print(f"Confidence: {entity.confidence:.2%}")
    
    if entity.properties:
        print("Properties:")
        for prop in entity.properties:
            print(f"  - {prop.type_}: {prop.mention_text}")

print("\n" + "="*70)
print(f"Total entities found: {len(document.entities)}")
print("="*70)