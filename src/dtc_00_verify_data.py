"""
Stage 3 verification: confirm all four raw CSVs exist,
are readable, and match expected basic structure.

Does NOT load Reviews.csv fully into memory - uses DuckDB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def check_listings(path: Path) -> None:
    print("\n--- Listings.csv ---")
    df = pd.read_csv(path, nrows=5)
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print("  Head (5 rows):")
    print(df.head().to_string())


def check_dictionary(path: Path, label: str) -> None:
    print(f"\n--- {label} ---")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(3).to_string())


def check_reviews_size(path: Path) -> None:
    print("\n--- Reviews.csv ---")
    con = duckdb.connect()
    n = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{path.as_posix()}')"
    ).fetchone()[0]
    print(f"  Row count: {n:,}")
    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{path.as_posix()}') LIMIT 0"
    ).fetchall()
    print(f"  Columns ({len(cols)}): {[c[0] for c in cols]}")
    con.close()


def main() -> None:
    cfg = load_config()
    files = {
        "Listings.csv":        ROOT / cfg["paths"]["listings_raw"],
        "Reviews.csv":         ROOT / cfg["paths"]["reviews_raw"],
        "Listings dictionary": ROOT / cfg["paths"]["listings_dict_raw"],
        "Reviews dictionary":  ROOT / cfg["paths"]["reviews_dict_raw"],
    }

    print("=" * 60)
    print("STAGE 3 - DATA VERIFICATION")
    print("=" * 60)

    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        print("MISSING FILES:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    for name, p in files.items():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [OK] {name:22s} {size_mb:8.2f} MB   {p}")

    check_listings(files["Listings.csv"])
    check_dictionary(files["Listings dictionary"], "Listings data dictionary")
    check_dictionary(files["Reviews dictionary"],  "Reviews data dictionary")
    check_reviews_size(files["Reviews.csv"])

    print("\nSTAGE 3 VERIFICATION COMPLETE")


if __name__ == "__main__":
    main()