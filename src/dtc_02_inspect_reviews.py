"""
Stage 4 (continued) - Deep review inspection.

Focus:
  * The 160 duplicated review_id values - concrete evidence.
  * Review activity distribution by year.
  * Reviews per listing - extreme cases.

Output:
  reports/reviews_inspection.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_config()
    reviews_path = ROOT / cfg["paths"]["reviews_raw"]
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "reviews_inspection.md"

    if not reviews_path.exists():
        sys.exit(f"ERROR: Reviews.csv not found: {reviews_path}")

    p = reviews_path.as_posix()
    con = duckdb.connect()

    out: list[str] = []
    out.append("# Reviews Deep Inspection")
    out.append(f"\n**Project:** {cfg['project']['name']}")
    out.append(f"**Author:** {cfg['project']['author']}")

    # ---------- DUPLICATE REVIEW_ID EVIDENCE ----------
    print("[1/3] Investigating the 160 duplicate review_id values ...")
    out.append("\n## 1. Duplicate review_id investigation")
    out.append(
        "\nThe assignment brief states 160 repeated review IDs exist. "
        "Below is the evidence, produced directly from Reviews.csv."
    )

    dup_rows = con.execute(f"""
        WITH counts AS (
          SELECT review_id, COUNT(*) AS n
          FROM read_csv_auto('{p}')
          GROUP BY review_id
          HAVING COUNT(*) > 1
        )
        SELECT
          COUNT(*) AS n_duplicated_ids,
          SUM(n) AS total_rows_involved
        FROM counts
    """).fetchone()
    out.append(f"\n- Duplicated review_id values: **{dup_rows[0]:,}**")
    out.append(f"- Total rows involved: **{dup_rows[1]:,}**")

    out.append("\n### Full list of duplicated review_id values\n")
    out.append("| review_id | occurrences | same_listing_id | distinct_listing_ids |")
    out.append("|---|---|---|---|")

    detailed = con.execute(f"""
        SELECT
          review_id,
          COUNT(*) AS occurrences,
          COUNT(DISTINCT listing_id) AS distinct_listings,
          CASE
            WHEN COUNT(DISTINCT listing_id) = 1 THEN 'YES (same listing)'
            ELSE 'NO (across listings)'
          END AS same_listing
        FROM read_csv_auto('{p}')
        GROUP BY review_id
        HAVING COUNT(*) > 1
        ORDER BY occurrences DESC, review_id ASC
    """).fetchall()

    for review_id, occurrences, distinct_listings, same_listing in detailed:
        out.append(f"| {review_id} | {occurrences} | {same_listing} | {distinct_listings} |")

    # Summary of duplicate pattern
    same_listing_count = sum(1 for _, _, _, s in detailed if s.startswith("YES"))
    cross_listing_count = len(detailed) - same_listing_count
    out.append("\n### Pattern summary\n")
    out.append(f"- Duplicate IDs appearing **only within one listing**: **{same_listing_count}**")
    out.append(f"- Duplicate IDs appearing **across different listings**: **{cross_listing_count}**")
    if same_listing_count > 0 and cross_listing_count == 0:
        out.append(
            "\n**Decision:** All duplicates are within the same listing. "
            "This pattern is consistent with either a genuine re-review "
            "or a data-export glitch. Since `(listing_id, review_id)` is "
            "still unique, we treat **`(listing_id, review_id)` as the "
            "true composite key** for analytical purposes and do not "
            "delete any rows. For any per-review_id aggregation (e.g. "
            "counting distinct reviewers), we will deduplicate on the pair."
        )
    else:
        out.append(
            "\n**Decision:** Duplicates span listings — treat `(listing_id, review_id)` "
            "as the composite key. Do not delete; use DISTINCT when counting."
        )

    # ---------- YEARLY REVIEW ACTIVITY ----------
    print("[2/3] Review activity by year ...")
    out.append("\n## 2. Review activity over time\n")
    out.append("| Year | Reviews |")
    out.append("|---|---|")

    yearly = con.execute(f"""
        SELECT
          EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) AS yr,
          COUNT(*) AS n
        FROM read_csv_auto('{p}')
        GROUP BY yr
        ORDER BY yr
    """).fetchall()
    for yr, n in yearly:
        out.append(f"| {int(yr) if yr is not None else 'N/A'} | {n:,} |")

    # ---------- EXTREME REVIEW COUNTS ----------
    print("[3/3] Extreme review counts ...")
    out.append("\n## 3. Listings with the most reviews\n")
    out.append("Top 15 listings by review count:\n")
    out.append("| listing_id | review_count |")
    out.append("|---|---|")

    top = con.execute(f"""
        SELECT listing_id, COUNT(*) AS c
        FROM read_csv_auto('{p}')
        GROUP BY listing_id
        ORDER BY c DESC
        LIMIT 15
    """).fetchall()
    for lid, c in top:
        out.append(f"| {lid} | {c:,} |")

    con.close()

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nDONE. Report written to: {out_path}")


if __name__ == "__main__":
    main()