# src/analytics/price_validation.py

from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.ensemble import IsolationForest


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("ledgerx.vendor_price_validation")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class PriceValidationConfig:
    input_csv: Path = Path("data/processed/fatura_structured.csv")
    vendor_col: str = "vendor_name"
    amount_col: str = "total_amount"
    item_col: Optional[str] = None  # e.g., item_code, product_id if exists
    date_col: Optional[str] = "invoice_date"
    output_dir: Path = Path("reports/vendor_price")


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
def load_data(cfg: PriceValidationConfig) -> pd.DataFrame:
    logger.info(f"Loading invoice data from: {cfg.input_csv}")
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_csv}")

    df = pd.read_csv(cfg.input_csv)

    required = [cfg.vendor_col, cfg.amount_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df[cfg.amount_col] = pd.to_numeric(df[cfg.amount_col], errors="coerce")
    df = df.dropna(subset=[cfg.amount_col])

    if cfg.date_col and cfg.date_col in df.columns:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")

    logger.info(f"Loaded {len(df):,} rows for price validation.")
    return df


# -----------------------------------------------------------------------------
# Vendor Baseline Computation
# -----------------------------------------------------------------------------
def compute_vendor_baselines(df: pd.DataFrame, cfg: PriceValidationConfig) -> pd.DataFrame:
    logger.info("Computing vendor median prices...")

    vendor_baselines = (
        df.groupby(cfg.vendor_col)[cfg.amount_col]
        .median()
        .reset_index()
        .rename(columns={cfg.amount_col: "median_vendor_price"})
    )

    logger.info(f"Computed baselines for {len(vendor_baselines):,} vendors.")
    return vendor_baselines


# -----------------------------------------------------------------------------
# Price Anomaly Detection
# -----------------------------------------------------------------------------
def detect_price_anomalies(df: pd.DataFrame, cfg: PriceValidationConfig) -> pd.DataFrame:
    logger.info("Running IsolationForest price anomaly detection...")

    # Model
    model = IsolationForest(
        contamination=0.03,  # ~3% anomalies
        random_state=42,
        n_estimators=200,
    )

    df["price_score"] = model.fit_predict(df[[cfg.amount_col]])
    df["anomaly"] = df["price_score"].apply(lambda x: "anomaly" if x == -1 else "normal")

    logger.info(f"Anomalies detected: {df['anomaly'].value_counts().to_dict()}")
    return df


# -----------------------------------------------------------------------------
# Summary Metrics
# -----------------------------------------------------------------------------
def summarize(df: pd.DataFrame, cfg: PriceValidationConfig) -> dict:
    total = len(df)
    anomalies = int((df["anomaly"] == "anomaly").sum())

    summary = {
        "total_records": total,
        "anomalies": anomalies,
        "anomaly_rate": round(anomalies / total, 4),
    }

    logger.info(f"Summary: {summary}")
    return summary


# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
def run_vendor_price_validation(cfg: Optional[PriceValidationConfig] = None):
    cfg = cfg or PriceValidationConfig()

    ensure_dir(cfg.output_dir)

    # Load
    df = load_data(cfg)

    # Vendor baselines
    vendor_base = compute_vendor_baselines(df, cfg)
    vendor_base_path = cfg.output_dir / "vendor_baselines.csv"
    vendor_base.to_csv(vendor_base_path, index=False)
    logger.info(f"Saved vendor baselines → {vendor_base_path}")

    # Detect anomalies
    df_anom = detect_price_anomalies(df, cfg)
    anomalies_path = cfg.output_dir / "price_anomalies.csv"
    df_anom.to_csv(anomalies_path, index=False)
    logger.info(f"Saved price anomalies → {anomalies_path}")

    # Summary
    summary = summarize(df_anom, cfg)
    summary_path = cfg.output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary → {summary_path}")

    logger.info("Vendor Price Validation Completed Successfully.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_vendor_price_validation()
