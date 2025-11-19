import cv2
import pandas as pd
from pathlib import Path

from augment_images import apply_random_augmentations
from synthetic_failures import corrupt_invoice_record

def preprocess_with_augmentations(df, image_dir):
    enhanced_records = []

    for _, row in df.iterrows():
        
        img_path = Path(image_dir) / row["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply random augmentations 40% of the time
        if random.random() < 0.4:
            img = apply_random_augmentations(img)
            row["quality"] = 0
        else:
            row["quality"] = 1

        # Save the augmented image
        save_path = Path(image_dir) / f"aug_{row['file_name']}"
        cv2.imwrite(str(save_path), img)

        row["file_name"] = f"aug_{row['file_name']}"

        # OCR corruption 20% of the time
        if random.random() < 0.2:
            row = corrupt_invoice_record(row)

        enhanced_records.append(row)

    return pd.DataFrame(enhanced_records)
