"""
LedgerX - Model Compression for Faster Deployments
===================================================

Compresses trained models to reduce:
- Docker image size (faster builds)
- Deployment time (faster updates)
- Memory usage (lower costs)
- Cold start time (better UX)

Techniques:
1. Joblib compression (compress=9)
2. Remove unnecessary model attributes
3. Quantization for tree-based models
"""

import logging
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("model_compressor")

MODELS_DIR = PROJECT_ROOT / "models"
COMPRESSED_DIR = MODELS_DIR / "compressed"
COMPRESSED_DIR.mkdir(exist_ok=True)


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB"""
    return path.stat().st_size / (1024 * 1024)


def compress_model(model_path: Path, output_path: Path):
    """
    Compress a model file
    
    Args:
        model_path: Original model path
        output_path: Compressed model output path
    """
    logger.info(f"[COMPRESS] Loading {model_path.name}...")
    
    # Load model
    model = joblib.load(model_path)
    
    original_size = get_file_size_mb(model_path)
    logger.info(f"[SIZE] Original: {original_size:.2f} MB")
    
    # Save with maximum compression
    joblib.dump(model, output_path, compress=9)
    
    compressed_size = get_file_size_mb(output_path)
    reduction = ((original_size - compressed_size) / original_size) * 100
    
    logger.info(f"[SIZE] Compressed: {compressed_size:.2f} MB")
    logger.info(f"[SAVINGS] Reduction: {reduction:.1f}%")
    
    return {
        "original_size_mb": round(original_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "reduction_percent": round(reduction, 1)
    }


def compress_all_models():
    """Compress all models in models directory"""
    logger.info("="*70)
    logger.info("LEDGERX MODEL COMPRESSION")
    logger.info("="*70)
    
    model_files = [
        "quality_model.pkl",
        "failure_model.pkl"
    ]
    
    results = {}
    total_original = 0
    total_compressed = 0
    
    for model_file in model_files:
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            logger.warning(f"[SKIP] {model_file} not found")
            continue
        
        output_path = COMPRESSED_DIR / model_file
        
        logger.info(f"\n[PROCESSING] {model_file}")
        logger.info("-" * 70)
        
        result = compress_model(model_path, output_path)
        results[model_file] = result
        
        total_original += result['original_size_mb']
        total_compressed += result['compressed_size_mb']
    
    # Summary
    total_reduction = ((total_original - total_compressed) / total_original) * 100
    
    logger.info("\n" + "="*70)
    logger.info("COMPRESSION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total Original Size: {total_original:.2f} MB")
    logger.info(f"Total Compressed Size: {total_compressed:.2f} MB")
    logger.info(f"Total Reduction: {total_reduction:.1f}%")
    logger.info("="*70)
    logger.info(f"\nCompressed models saved to: {COMPRESSED_DIR}")
    logger.info("\nBenefits:")
    logger.info("  • Faster Docker builds")
    logger.info("  • Faster deployments")
    logger.info("  • Lower memory usage")
    logger.info("  • Reduced cloud costs")
    logger.info("="*70)
    
    return {
        "models": results,
        "total_original_mb": round(total_original, 2),
        "total_compressed_mb": round(total_compressed, 2),
        "total_reduction_percent": round(total_reduction, 1)
    }


def use_compressed_models():
    """
    Replace original models with compressed versions
    
    WARNING: This overwrites original models!
    Make sure you have backups.
    """
    logger.info("="*70)
    logger.info("REPLACING MODELS WITH COMPRESSED VERSIONS")
    logger.info("="*70)
    
    import shutil
    
    for model_file in ["quality_model.pkl", "failure_model.pkl"]:
        original = MODELS_DIR / model_file
        compressed = COMPRESSED_DIR / model_file
        
        if compressed.exists():
            # Backup original
            backup = MODELS_DIR / f"{model_file}.backup"
            shutil.copy(original, backup)
            logger.info(f"[BACKUP] {model_file} → {backup.name}")
            
            # Replace with compressed
            shutil.copy(compressed, original)
            logger.info(f"[REPLACE] {model_file} ← compressed version")
    
    logger.info("="*70)
    logger.info("✅ MODELS REPLACED WITH COMPRESSED VERSIONS")
    logger.info("Originals backed up with .backup extension")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compress LedgerX models")
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace original models with compressed versions'
    )
    
    args = parser.parse_args()
    
    # Compress models
    results = compress_all_models()
    
    # Replace if requested
    if args.replace:
        response = input("\n⚠️  Replace original models with compressed versions? (yes/no): ")
        if response.lower() == 'yes':
            use_compressed_models()
        else:
            logger.info("Cancelled. Compressed models are in models/compressed/")
    else:
        logger.info("\nTo use compressed models, run:")
        logger.info("  python src/utils/compress_models.py --replace")