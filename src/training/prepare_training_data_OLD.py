"""
Prepare Training Data - Correct Feature Selection
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    logger.info("="*80)
    logger.info("LedgerX Prepare Training Data Stage")
    logger.info("="*80)
    
    INPUT_CSV = Path("data/processed/fatura_enterprise_preprocessed.csv")
    OUTPUT_QUALITY = Path("data/processed/quality_training.csv")
    OUTPUT_FAILURE = Path("data/processed/failure_training.csv")
    
    logger.info(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} records")
    
    # QUALITY MODEL: Select numeric features + labels
    quality_features = [
        'file_name',  # Keep for reference
        'blur_score', 
        'contrast_score', 
        'ocr_confidence', 
        'num_pages_fake', 
        'quality_score', 
        'label_quality_bad'
    ]
    quality_data = df[quality_features].copy()
    
    # FAILURE MODEL: Select numeric features + labels
    failure_features = [
        'file_name',  # Keep for reference
        'total_amount', 
        'subtotal', 
        'tax', 
        'blur_score', 
        'ocr_confidence', 
        'label_failure'
    ]
    failure_data = df[failure_features].copy()
    
    # Save
    OUTPUT_QUALITY.parent.mkdir(parents=True, exist_ok=True)
    quality_data.to_csv(OUTPUT_QUALITY, index=False)
    failure_data.to_csv(OUTPUT_FAILURE, index=False)
    
    logger.info("="*80)
    logger.info(f"✅ Quality training: {OUTPUT_QUALITY} ({len(quality_data)} records)")
    logger.info(f"   Features: {quality_features}")
    logger.info(f"✅ Failure training: {OUTPUT_FAILURE} ({len(failure_data)} records)")
    logger.info(f"   Features: {failure_features}")
    logger.info("="*80)