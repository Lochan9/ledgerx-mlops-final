# src/analytics/spend_analytics.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("ledgerx.spend_analytics")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class SpendAnalyticsConfig:
    # Input
    input_csv: Path = Path("data/processed/fatura_structured.csv")
    date_col: str = "invoice_date"
    vendor_col: str = "vendor_name"
    amount_col: str = "total_amount"
    category_cols: Optional[List[str]] = None  # e.g. ["gl_code"] or ["category"]

    # Outputs
    output_dir: Path = Path("reports/spend")


# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------
def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_invoice_data(cfg: SpendAnalyticsConfig) -> pd.DataFrame:
    logger.info(f"Loading invoice data from: {cfg.input_csv}")
    if not cfg.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {cfg.input_csv}. "
            f"Update SpendAnalyticsConfig.input_csv to point to your structured invoice file."
        )

    df = pd.read_csv(cfg.input_csv)

    required_cols = [cfg.date_col, cfg.vendor_col, cfg.amount_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {cfg.input_csv}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse dates
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col])

    # Ensure numeric amount
    df[cfg.amount_col] = pd.to_numeric(df[cfg.amount_col], errors="coerce")
    df = df.dropna(subset=[cfg.amount_col])

    logger.info(
        f"Loaded {len(df):,} invoices with columns: "
        f"{[cfg.date_col, cfg.vendor_col, cfg.amount_col]}"
    )
    return df


def compute_vendor_totals(df: pd.DataFrame, cfg: SpendAnalyticsConfig) -> pd.DataFrame:
    logger.info("Computing total spend per vendor...")
    vendor_totals = (
        df.groupby(cfg.vendor_col)[cfg.amount_col]
        .sum()
        .reset_index()
        .rename(columns={cfg.amount_col: "total_spend"})
        .sort_values("total_spend", ascending=False)
    )
    logger.info(f"Computed vendor totals for {len(vendor_totals):,} vendors.")
    return vendor_totals


def compute_vendor_monthly_totals(df: pd.DataFrame, cfg: SpendAnalyticsConfig) -> pd.DataFrame:
    logger.info("Computing monthly spend per vendor...")
    tmp = df.copy()
    tmp["year_month"] = tmp[cfg.date_col].dt.to_period("M").dt.to_timestamp()
    vendor_monthly = (
        tmp.groupby([cfg.vendor_col, "year_month"])[cfg.amount_col]
        .sum()
        .reset_index()
        .rename(columns={cfg.amount_col: "monthly_spend"})
        .sort_values(["year_month", "monthly_spend"], ascending=[True, False])
    )
    logger.info(f"Computed vendor-month totals: {len(vendor_monthly):,} rows.")
    return vendor_monthly


def compute_category_totals(df: pd.DataFrame, cfg: SpendAnalyticsConfig) -> Optional[pd.DataFrame]:
    if not cfg.category_cols:
        logger.info("No category columns configured; skipping category totals.")
        return None

    missing = [c for c in cfg.category_cols if c not in df.columns]
    if missing:
        logger.warning(
            f"Configured category columns not found: {missing}. "
            "Skipping category totals."
        )
        return None

    logger.info(f"Computing category totals for: {cfg.category_cols}")
    cat_totals = (
        df.groupby(cfg.category_cols)[cfg.amount_col]
        .sum()
        .reset_index()
        .rename(columns={cfg.amount_col: "total_spend"})
        .sort_values("total_spend", ascending=False)
    )
    logger.info(f"Computed {len(cat_totals):,} category rows.")
    return cat_totals


def compute_summary_metrics(
    df: pd.DataFrame,
    vendor_totals: pd.DataFrame,
    cfg: SpendAnalyticsConfig,
) -> dict:
    logger.info("Computing high-level summary metrics...")
    total_spend = float(df[cfg.amount_col].sum())
    n_invoices = int(len(df))
    n_vendors = int(df[cfg.vendor_col].nunique())

    # Top 10 vendors
    top_vendors = (
        vendor_totals.head(10)
        .to_dict(orient="records")
    )

    # Monthly totals
    tmp = df.copy()
    tmp["year_month"] = tmp[cfg.date_col].dt.to_period("M").dt.to_timestamp()
    monthly_totals = (
        tmp.groupby("year_month")[cfg.amount_col]
        .sum()
        .reset_index()
        .rename(columns={cfg.amount_col: "total_spend"})
    )
    monthly_totals["year_month"] = monthly_totals["year_month"].dt.strftime("%Y-%m")
    monthly_records = monthly_totals.to_dict(orient="records")

    summary = {
        "total_spend": total_spend,
        "n_invoices": n_invoices,
        "n_vendors": n_vendors,
        "top_vendors": top_vendors,
        "monthly_totals": monthly_records,
    }

    logger.info(
        f"Summary: total_spend={total_spend:,.2f}, "
        f"n_invoices={n_invoices}, n_vendors={n_vendors}"
    )
    return summary


def run_spend_analytics(cfg: Optional[SpendAnalyticsConfig] = None) -> None:
    """
    Main entry point for pipeline and Airflow.
    """
    cfg = cfg or SpendAnalyticsConfig()
    _ensure_output_dir(cfg.output_dir)

    # 1) Load data
    df = load_invoice_data(cfg)

    # 2) Vendor-level totals
    vendor_totals = compute_vendor_totals(df, cfg)
    vendor_totals_path = cfg.output_dir / "vendor_totals.csv"
    vendor_totals.to_csv(vendor_totals_path, index=False)
    logger.info(f"Saved vendor totals to: {vendor_totals_path}")

    # 3) Vendor-monthly totals
    vendor_monthly = compute_vendor_monthly_totals(df, cfg)
    vendor_monthly_path = cfg.output_dir / "vendor_monthly_totals.csv"
    vendor_monthly.to_csv(vendor_monthly_path, index=False)
    logger.info(f"Saved vendor-month totals to: {vendor_monthly_path}")

    # 4) Category totals (optional)
    category_totals = compute_category_totals(df, cfg)
    if category_totals is not None:
        category_path = cfg.output_dir / "category_totals.csv"
        category_totals.to_csv(category_path, index=False)
        logger.info(f"Saved category totals to: {category_path}")

    # 5) Summary metrics (JSON)
    summary = compute_summary_metrics(df, vendor_totals, cfg)
    summary_path = cfg.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary metrics to: {summary_path}")

    logger.info("Spend analytics completed successfully.")


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_spend_analytics()
