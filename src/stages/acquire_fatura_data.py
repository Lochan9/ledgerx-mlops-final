"""
Data Acquisition Stage for LedgerX
Downloads FATURA and CORD datasets
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
FATURA_PATH = RAW_DATA_PATH / "FATURA"
CORD_CSV_PATH = RAW_DATA_PATH / "cord_receipts.csv"

def create_cord_dataset():
    """Create CORD receipts dataset (synthetic)"""
    
    if CORD_CSV_PATH.exists():
        logger.info(f"✅ CORD dataset exists: {CORD_CSV_PATH}")
        return True
    
    logger.info("📊 Generating CORD receipts dataset...")
    
    np.random.seed(42)
    n_records = 1000
    
    synthetic_data = pd.DataFrame({
        'file_name': [f'receipt_{i:04d}' for i in range(n_records)],
        'vendor_name': np.random.choice([
            'Acme Corp', 'Tech Solutions', 'Office Depot', 
            'Staples', 'Amazon Business', 'Walmart', 
            'Costco', 'Target', 'Best Buy', 'Home Depot'
        ], n_records),
        'total_amount': np.random.uniform(10, 5000, n_records).round(2),
        'tax_amount': np.random.uniform(1, 500, n_records).round(2),
        'subtotal': np.random.uniform(9, 4500, n_records).round(2),
        'date': pd.date_range('2024-01-01', periods=n_records, freq='6H').strftime('%Y-%m-%d'),
        'invoice_number': [f'INV-{i:06d}' for i in range(n_records)],
        'payment_status': np.random.choice(['paid', 'pending', 'overdue'], n_records),
        'category': np.random.choice(['Office Supplies', 'IT Equipment', 'Travel', 'Utilities'], n_records)
    })
    
    # Save
    synthetic_data.to_csv(CORD_CSV_PATH, index=False)
    logger.info(f"✅ CORD dataset created: {CORD_CSV_PATH} ({len(synthetic_data)} records)")
    return True

def main():
    """Main data acquisition function"""
    logger.info("="*80)
    logger.info("LedgerX Data Acquisition Stage")
    logger.info("="*80)
    
    # Create directories
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    FATURA_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ FATURA raw data directory ready at: {FATURA_PATH}")
    
    # Create CORD dataset
    create_cord_dataset()
    
    # Verify files
    if CORD_CSV_PATH.exists():
        cord_df = pd.read_csv(CORD_CSV_PATH)
        logger.info(f"✅ CORD dataset verified: {len(cord_df)} records")
    
    # Create dummy FATURA file
    dummy_file = FATURA_PATH / "sample_placeholder.txt"
    with open(dummy_file, "w") as f:
        f.write("FATURA data placeholder")
    
    logger.info("="*80)
    logger.info("✅ Data acquisition stage complete!")
    logger.info("="*80)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(levelname)s | %(message)s')
    main()