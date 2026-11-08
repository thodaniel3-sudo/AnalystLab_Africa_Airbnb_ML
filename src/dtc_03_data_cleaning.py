"""
Stage 5 - Data Cleaning.

Reads:   data/raw/Listings.csv (Latin-1 encoded)
Writes:  data/processed/listings_clean.parquet
         data/processed/cleaning_metadata.parquet (row count audit trail)
         reports/cleaning_report.md

Cleaning decisions (all recorded in reports/decision_log.md):
  * Drop 113 zero-price rows              (invalid listing price)
  * Drop rows with negative price         (should be 0 in this dataset; guarded)
  * Clip extreme prices at the 99.5th     (documented - see report)
    percentile per city
  * Add `price_clean` numeric column      (will be the model target)
  * Convert `host_response_time` NaN       to "Unknown" category
  * Leave `bedrooms` NaN as-is             (trees handle natively)
  * Leave `review_scores_*` NaN as-is      (trees handle natively)
  * Encode the `t`/`f` boolean strings     as Python bool
  * Write as UTF-8 parquet                 (no more encoding pain downstream)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_listings_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False), "utf-8"
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False), "latin-1"


def main() -> None:
    cfg = load_config()

    listings_path = ROOT / cfg["paths"]["listings_raw"]
    processed_path = ROOT / cfg["paths"]["listings_clean"]
    meta_path = processed_path.parent / "cleaning_metadata.parquet"
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    report_path = reports_dir / "cleaning_report.md"

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not listings_path.exists():
        sys.exit(f"ERROR: Listings.csv not found: {listings_path}")

    log: list[str] = []
    log.append("# Cleaning Report")
    log.append(f"\n**Project:** {cfg['project']['name']}")
    log.append(f"**Author:** {cfg['project']['author']}")
    log.append(f"**Cohort:** {cfg['project']['cohort']}")
    log.append(f"**Track:** {cfg['project']['track']}")

    # ---------- LOAD ----------
    print("[1/6] Loading raw Listings.csv ...")
    df, enc = load_listings_with_fallback(listings_path)
    rows_before = len(df)
    log.append(f"\n## 1. Input")
    log.append(f"- Raw rows: **{rows_before:,}**")
    log.append(f"- Raw columns: **{df.shape[1]}**")
    log.append(f"- Encoding used: **{enc}**")

    # ---------- PRICE ----------
    print("[2/6] Cleaning price ...")
    log.append("\n## 2. Price cleaning")

    price_numeric = pd.to_numeric(df["price"], errors="coerce")
    n_non_numeric = int(price_numeric.isna().sum())
    log.append(f"- Price dtype in raw file: **{df['price'].dtype}**")
    log.append(f"- Non-numeric prices: **{n_non_numeric}**")

    if n_non_numeric > 0:
        # If price were a string in a future version of the file, strip currency
        # symbols and re-parse. Current file already has clean ints, so this
        # branch will not fire today, but the pipeline stays robust.
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
        )
        price_numeric = pd.to_numeric(df["price"], errors="coerce")

    n_zero = int((price_numeric == 0).sum())
    n_negative = int((price_numeric < 0).sum())
    log.append(f"- Zero-price rows: **{n_zero}**")
    log.append(f"- Negative-price rows: **{n_negative}**")

    # Drop zero and negative price
    keep_mask = (price_numeric > 0) & price_numeric.notna()
    df = df.loc[keep_mask].copy()
    df["price_clean"] = pd.to_numeric(df["price"], errors="coerce")

    rows_after_price = len(df)
    log.append(f"- Dropped for price ≤ 0 or non-numeric: **{rows_before - rows_after_price}**")
    log.append(f"- Rows after price cleaning: **{rows_after_price:,}**")

    # ---------- EXTREME PRICES ----------
    print("[3/6] Clipping extreme prices per city (99.5th percentile) ...")
    log.append("\n## 3. Extreme price clipping")
    log.append(
        "- Strategy: clip `price_clean` at the **99.5th percentile within each city**.\n"
        "  Motivation: per-city currency means we cannot pool prices; extremely\n"
        "  high values in any one city would otherwise dominate MAE and RMSE in\n"
        "  that city's model. Clipping at 99.5% removes the top 0.5% of prices\n"
        "  per city, which is < 350 rows total in most cities. Those rows remain\n"
        "  in the dataset — only the price value is capped.\n"
        "  Rationale for 99.5 not 99: 99% would clip ~2,800 listings across the\n"
        "  whole dataset. 99.5% balances outlier control against data loss."
    )

    city_caps = {}
    for city, group in df.groupby("city"):
        cap = float(group["price_clean"].quantile(0.995))
        city_caps[city] = cap
    df["price_clean"] = df.apply(
        lambda row: min(row["price_clean"], city_caps.get(row["city"], row["price_clean"])),
        axis=1,
    )
    log.append("\n| City | 99.5th pct cap |")
    log.append("|---|---|")
    for city, cap in sorted(city_caps.items()):
        log.append(f"| {city} | {cap:,.0f} |")

    # ---------- CATEGORICAL / BOOLEAN ----------
    print("[4/6] Normalising categoricals and booleans ...")
    log.append("\n## 4. Categorical and boolean normalisation")

    # host_response_time: fill NaN with "unknown" so the category can be modeled
    n_missing_rt = int(df["host_response_time"].isna().sum())
    df["host_response_time"] = df["host_response_time"].fillna("unknown")
    log.append(
        f"- `host_response_time`: filled **{n_missing_rt:,}** missing values with `'unknown'`."
    )

    # Boolean strings 't'/'f' → Python bool
    bool_cols = ["host_is_superhost", "host_identity_verified", "instant_bookable"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False})
            n_nan = int(df[col].isna().sum())
            log.append(f"- `{col}`: mapped 't'/'f' → bool; **{n_nan:,}** unmapped (NaN).")

    # ---------- FEATURES TO KEEP ----------
    print("[5/6] Selecting output columns ...")
    log.append("\n## 5. Output column selection")

    numeric_feats = cfg["features"]["numeric"]
    cat_feats = cfg["features"]["categorical"]
    target = cfg["target"]["column"]
    group_key = cfg["split"]["group_by"]
    price_clean = "price_clean"

    keep = [target, price_clean, group_key] + numeric_feats + cat_feats
    # Deduplicate while preserving order
    seen = set()
    keep_dedup = [c for c in keep if not (c in seen or seen.add(c))]

    missing = [c for c in keep_dedup if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: columns missing after cleaning: {missing}")

    out = df[keep_dedup].copy()
    log.append(f"- Kept columns ({len(keep_dedup)}): `{keep_dedup}`")
    log.append(f"- Dropped columns: {df.shape[1] - len(keep_dedup)}")

    # ---------- SAVE ----------
    print("[6/6] Writing parquet and metadata ...")
    out.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    log.append("\n## 6. Output")
    log.append(f"- Output file: `{processed_path.relative_to(ROOT)}`")
    log.append(f"- Final rows: **{len(out):,}**")
    log.append(f"- Final columns: **{out.shape[1]}**")
    log.append(f"- Target column present: `{target}`")
    log.append(f"- Model target: `{price_clean}`")

    # Row-count audit trail
    meta = pd.DataFrame([
        {"stage": "raw", "rows": rows_before},
        {"stage": "after_price_filter", "rows": rows_after_price},
        {"stage": "final", "rows": len(out)},
    ])
    meta.to_parquet(meta_path, index=False)

    # Verify read-back
    check = pd.read_parquet(processed_path)
    assert len(check) == len(out), "parquet round-trip row count mismatch"
    assert check["price_clean"].notna().all(), "price_clean has NaNs after cleaning"
    assert (check["price_clean"] > 0).all(), "price_clean has non-positive values"

    log.append("\n## 7. Post-write validation")
    log.append(f"- ✅ Parquet round-trip: **{len(check):,}** rows")
    log.append("- ✅ `price_clean` has no NaN and no non-positive values")
    log.append(f"- ✅ Row count reconciliation: {rows_before:,} → {rows_after_price:,} → {len(out):,}")

    report_path.write_text("\n".join(log), encoding="utf-8")
    print(f"\nDONE. Clean data → {processed_path}")
    print(f"      Report     → {report_path}")


if __name__ == "__main__":
    main()