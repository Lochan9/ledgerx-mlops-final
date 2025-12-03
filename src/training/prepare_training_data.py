"""
Prepare Training Data - PRODUCTION VERSION (Step 2)
Selects optimal features, no data leakage, proper validation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    logger.info("="*80)
    logger.info("LedgerX PRODUCTION Prepare Training Data (Step 2)")
    logger.info("="*80)
    
    INPUT_CSV = Path("data/processed/fatura_enterprise_preprocessed.csv")
    OUTPUT_QUALITY = Path("data/processed/quality_training.csv")
    OUTPUT_FAILURE = Path("data/processed/failure_training.csv")
    
    logger.info(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # ================================================================
    # QUALITY MODEL: Select features (NO LEAKAGE!)
    # ================================================================
    
    quality_features = [
        'file_name',
        # RAW OCR METRICS
        'blur_score',
        'contrast_score',
        'ocr_confidence',
        'num_pages_fake',
        # ENGINEERED OCR FEATURES
        'blur_ocr_interaction',
        'blur_contrast_ratio',
        'ocr_blur_product',
        'blur_squared',
        'ocr_squared',
        'contrast_squared',
        'overall_image_quality',
        # QUALITY FLAGS
        'is_critical_low_blur',
        'is_low_blur',
        'is_excellent_blur',
        'is_low_ocr',
        'is_medium_ocr',
        'is_high_ocr',
        'is_low_contrast',
        'is_multipage',
        'is_high_risk_ocr',
        'is_multipage_low_quality',
        # TARGET
        'label_quality_bad'
    ]
    
    # Select only available columns
    available_quality = [f for f in quality_features if f in df.columns]
    quality_data = df[available_quality].copy()
    
    # Remove quality_score if accidentally included (CRITICAL!)
    if 'quality_score' in quality_data.columns:
        quality_data = quality_data.drop(columns=['quality_score'])
        logger.warning("⚠️ Removed quality_score (target leakage)")
    
    logger.info(f"Quality features selected: {len(available_quality) - 2}")  # -2 for file_name and label
    logger.info(f"Quality features: {[f for f in available_quality if f not in ['file_name', 'label_quality_bad']]}")
    
    # ================================================================
    # FAILURE MODEL: Select features (NO LEAKAGE!)
    # ================================================================
    
    failure_features = [
        'file_name',
        # FINANCIAL FEATURES
        'total_amount',
        'subtotal',
        'tax',
        'tax_rate',
        'tax_to_total_ratio',
        'math_error',
        'math_error_pct',
        'total_amount_log',
        'subtotal_log',
        # AMOUNT BINS
        'is_small_invoice',
        'is_medium_invoice',
        'is_large_invoice',
        'is_very_large_invoice',
        'is_amount_outlier',
        'amount_zscore',
        # OCR QUALITY
        'blur_score',
        'ocr_confidence',
        'overall_image_quality',
        'is_low_ocr',
        'is_high_risk_ocr',
        # TEMPORAL
        'day_of_week',
        'is_weekend',
        'is_monday',
        'month',
        'is_month_end',
        'quarter',
        # VENDOR
        'vendor_name_length',
        'vendor_has_numbers',
        'vendor_frequency',
        'is_rare_vendor',
        'is_frequent_vendor',
        'vendor_avg_amount',
        'amount_vs_vendor_avg',
        # AGGREGATE
        'amount_rolling_mean',
        'amount_rolling_std',
        # TARGET
        'label_failure'
    ]
    
    # Select only available columns
    available_failure = [f for f in failure_features if f in df.columns]
    failure_data = df[available_failure].copy()
    
    logger.info(f"Failure features selected: {len(available_failure) - 2}")
    logger.info(f"Failure features: {[f for f in available_failure if f not in ['file_name', 'label_failure']][:10]}...")  # Show first 10
    
    # ================================================================
    # VALIDATION CHECKS
    # ================================================================
    
    logger.info("="*80)
    logger.info("DATA LEAKAGE VALIDATION")
    logger.info("="*80)
    
    # Check 1: No target in features
    quality_feat_names = [f for f in quality_data.columns if f not in ['file_name', 'label_quality_bad']]
    failure_feat_names = [f for f in failure_data.columns if f not in ['file_name', 'label_failure']]
    
    leakage_found = False
    
    if 'quality_score' in quality_feat_names:
        logger.error("🚨 TARGET LEAKAGE: quality_score in features!")
        leakage_found = True
    
    if 'label_failure' in quality_feat_names:
        logger.error("🚨 LEAKAGE: label_failure in quality features!")
        leakage_found = True
    
    if 'label_quality_bad' in failure_feat_names:
        logger.error("🚨 LEAKAGE: label_quality_bad in failure features!")
        leakage_found = True
    
    if not leakage_found:
        logger.info("✅ No data leakage detected")
    
    # Check 2: Class balance
    qual_dist = quality_data['label_quality_bad'].value_counts()
    fail_dist = failure_data['label_failure'].value_counts()
    
    logger.info(f"✅ Quality class balance: {qual_dist.to_dict()}")
    logger.info(f"✅ Failure class balance: {fail_dist.to_dict()}")
    
    # Check 3: Sufficient samples
    if qual_dist.min() < 100:
        logger.warning(f"⚠️ Quality minority class: {qual_dist.min()} samples (recommend >1000)")
    else:
        logger.info(f"✅ Quality minority class: {qual_dist.min()} samples")
    
    if fail_dist.min() < 100:
        logger.warning(f"⚠️ Failure minority class: {fail_dist.min()} samples (recommend >1000)")
    else:
        logger.info(f"✅ Failure minority class: {fail_dist.min()} samples")
    
    # Check 4: Feature correlation with target
    logger.info("="*80)
    logger.info("FEATURE-TARGET CORRELATION CHECK")
    logger.info("="*80)
    
    # Top 5 correlations for quality
    quality_numeric = quality_data.select_dtypes(include=[np.number])
    quality_corr = quality_numeric.corrwith(quality_data['label_quality_bad']).abs().sort_values(ascending=False)
    
    logger.info("Top 5 Quality Features (by correlation):")
    for feat, corr in quality_corr.head(6).items():  # 6 to skip label itself
        if feat != 'label_quality_bad':
            logger.info(f"  - {feat}: {corr:.4f}")
    
    # Top 5 correlations for failure
    failure_numeric = failure_data.select_dtypes(include=[np.number])
    failure_corr = failure_numeric.corrwith(failure_data['label_failure']).abs().sort_values(ascending=False)
    
    logger.info("Top 5 Failure Features (by correlation):")
    for feat, corr in failure_corr.head(6).items():
        if feat != 'label_failure':
            logger.info(f"  - {feat}: {corr:.4f}")
    
    # ================================================================
    # SAVE OUTPUTS
    # ================================================================
    
    OUTPUT_QUALITY.parent.mkdir(parents=True, exist_ok=True)
    quality_data.to_csv(OUTPUT_QUALITY, index=False)
    failure_data.to_csv(OUTPUT_FAILURE, index=False)
    
    logger.info("="*80)
    logger.info(f"✅ Quality training: {OUTPUT_QUALITY}")
    logger.info(f"   Records: {len(quality_data)}")
    logger.info(f"   Features: {len(quality_feat_names)}")
    logger.info(f"✅ Failure training: {OUTPUT_FAILURE}")
    logger.info(f"   Records: {len(failure_data)}")
    logger.info(f"   Features: {len(failure_feat_names)}")
    logger.info("="*80)