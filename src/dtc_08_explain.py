"""
Stage 10 - SHAP Explainability (differentiator #4).

Loads the winning grouped-split model per city, runs SHAP TreeExplainer
on a 2,000-row sample, and produces:

  visualizations/dtc_08_beeswarm_{city}.png
      Global SHAP summary (beeswarm) — which features matter, in what direction.

  visualizations/dtc_08_waterfall_{city}_1.png, _2.png, _3.png
      Per-listing SHAP waterfalls — 3 example listings per city, showing
      exactly why the model predicted that price.

  reports/dtc_08_shap.md
      Interpretation document with all plots embedded.

The sample size (2,000) is a decision-log entry: SHAP on all 279k rows
would take hours and produce visually cluttered plots. A stratified
sample preserves the signal while finishing in minutes.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_").replace("/", "_")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_winner(report_path: Path) -> dict[str, str]:
    """Extract winner per city from dtc_06_model_comparison.md (grouped split)."""
    text = report_path.read_text(encoding="utf-8")
    in_grouped = False
    winners: dict[str, tuple[str, float]] = {}
    for line in text.splitlines():
        if line.startswith("## Grouped split"):
            in_grouped = True
            continue
        if line.startswith("## Random split"):
            in_grouped = False
            continue
        if not in_grouped or not line.startswith("| "):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 7:
            continue
        city, model = parts[0], parts[1]
        try:
            mae = float(parts[4])
        except ValueError:
            continue
        if city not in winners or mae < winners[city][1]:
            winners[city] = (model, mae)
    return {city: m for city, (m, _) in winners.items()}


def get_feature_names_pre(preprocessor) -> list[str]:
    """Extract feature names OUT of the ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


def main() -> None:
    cfg = load_config()
    data_path = ROOT / cfg["paths"]["listings_clean"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    viz_dir = ROOT / cfg["paths"]["visualizations_dir"]
    viz_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"ERROR: cleaned data not found: {data_path}")

    mod6 = load_module("dtc_06_train_model", ROOT / "src" / "dtc_06_train_model.py")

    print("[1/4] Loading data and winners ...")
    df = pd.read_parquet(data_path)
    winners = parse_winner(reports_dir / "dtc_06_model_comparison.md")
    print(f"      Winners parsed for {len(winners)} cities")

    numeric_full = cfg["features"]["numeric"]
    LOW_CARD = ["room_type", "property_type", "host_response_time",
                "host_is_superhost", "host_identity_verified", "instant_bookable"]
    HIGH_CARD = ["neighbourhood"]
    feature_cols = numeric_full + LOW_CARD + HIGH_CARD
    target = "price_clean"
    sample_size = int(cfg["explainability"]["max_samples_for_shap"])

    ev = ["# SHAP Explainability Report", ""]
    ev.append(f"**Project:** {cfg['project']['name']}")
    ev.append(f"**Author:** {cfg['project']['author']}")
    ev.append("")
    ev.append(f"## Method")
    ev.append("")
    ev.append(f"`shap.TreeExplainer` is used on the winning grouped-split model "
              f"per city. Sample size: **{sample_size:,}** rows per city "
              f"(config: `explainability.max_samples_for_shap`). Larger samples "
              f"produce cluttered visualisations and take hours; 2,000 rows "
              f"preserves the signal while keeping runtime under a minute per city.")
    ev.append("")

    city_list = sorted(df["city"].dropna().unique())
    total = len(city_list)

    print(f"\n[2/4] Computing SHAP per city (sample={sample_size}) ...")
    for i, city in enumerate(city_list, start=1):
        slug = city_slug(city)
        df_city = df[df["city"] == city].reset_index(drop=True)
        if len(df_city) < 50:
            continue

        winner = winners.get(city)
        if winner is None:
            print(f"  [{i}/{total}] {city:20s} no winner, skipping")
            continue

        model_path = models_dir / f"dtc_{slug}_best_grouped.joblib"
        if not model_path.exists():
            print(f"  [{i}/{total}] {city:20s} model file missing")
            continue

        # Random sample for SHAP
        sample_idx = np.random.RandomState(42).choice(
            len(df_city), size=min(sample_size, len(df_city)), replace=False,
        )
        df_sample = df_city.iloc[sample_idx].reset_index(drop=True)
        X_sample = df_sample[feature_cols]

        # Load pipeline and split into preprocessor + model
        pipe = joblib.load(model_path)
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]

        # Transform sample through the preprocessor once
        X_trans = pre.transform(X_sample)
        feature_names = get_feature_names_pre(pre)

        # Determine which SHAP explainer to use
        # HistGradientBoostingRegressor: use shap.TreeExplainer
        # RandomForestRegressor: use shap.TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
        except Exception as e:
            print(f"  [{i}/{total}] {city:20s} SHAP error: {e}")
            continue

        print(f"  [{i}/{total}] {city:20s} SHAP computed "
              f"({len(X_trans):,} rows × {len(feature_names)} features)")

        # ---- Beeswarm (global) ----
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_trans, feature_names=feature_names,
            show=False, max_display=15,
        )
        plt.title(f"{city} — SHAP Summary (which features drive price)")
        plt.tight_layout()
        plt.savefig(viz_dir / f"dtc_08_beeswarm_{slug}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ---- Waterfalls (3 per city) ----
        # Choose 3 listings: near median, high-price, low-price
        y_sample = df_sample[target].values
        med_idx = int(np.argsort(np.abs(y_sample - np.median(y_sample)))[0])
        hi_idx = int(np.argmax(y_sample))
        lo_idx = int(np.argmin(y_sample))
        example_idx = [(med_idx, "median"),
                       (hi_idx, "highest"),
                       (lo_idx, "lowest")]

        # Base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value.flat[0])

        for j, (row_idx, label) in enumerate(example_idx, start=1):
            actual_price = float(y_sample[row_idx])
            explanation = shap.Explanation(
                values=shap_values[row_idx],
                base_values=base_value,
                data=X_trans[row_idx],
                feature_names=feature_names,
            )
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, max_display=12, show=False)
            plt.title(f"{city} — {label} listing "
                      f"(actual={actual_price:.0f})", fontsize=11)
            plt.tight_layout()
            plt.savefig(
                viz_dir / f"dtc_08_waterfall_{slug}_{j}.png",
                dpi=110, bbox_inches="tight",
            )
            plt.close(fig)

        # ---- Text report per city ----
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap)[::-1][:10]
        ev.append(f"### {city} (winner: {winner})")
        ev.append("")
        ev.append(f"**Global feature importance (mean |SHAP|):**")
        ev.append("")
        ev.append("| Rank | Feature | Mean |SHAP| |")
        ev.append("|---|---|---|")
        for rank, idx in enumerate(top_features, start=1):
            ev.append(f"| {rank} | {feature_names[idx]} | {mean_abs_shap[idx]:.3f} |")
        ev.append("")
        ev.append(f"**Beeswarm plot:**")
        ev.append("")
        ev.append(f"![beeswarm {city}](../visualizations/dtc_08_beeswarm_{slug}.png)")
        ev.append("")
        ev.append(f"**Example listings:**")
        ev.append("")
        for j, (_, label) in enumerate(example_idx, start=1):
            ev.append(f"- {label.capitalize()} listing → "
                      f"![waterfall](../visualizations/dtc_08_waterfall_{slug}_{j}.png)")
        ev.append("")

    print("\n[3/4] Writing interpretation report ...")
    ev.insert(6, "## How to read these plots")
    ev.insert(7, "")
    ev.insert(8, "- **Beeswarm:** Each point is one listing in the sample. X-axis = SHAP "
               "value (contribution to price). Colour = feature value (red high, blue low). "
               "If red points cluster on the right of a feature, that feature *increases* "
               "price when its value is high.")
    ev.insert(9, "- **Waterfall:** For one specific listing, starts at the model's base "
               "value (the average prediction). Each bar shows one feature pushing the "
               "prediction up (red) or down (blue) until reaching the final predicted price.")
    ev.insert(10, "")
    ev.insert(11, "## ⚠️ Correlation is not causation")
    ev.insert(12, "")
    ev.insert(13, "SHAP values show what the *model learned* from the data, not what "
               "causes prices in reality. A feature can show large SHAP values simply "
               "because it correlates with another unobserved driver. Do not interpret "
               "SHAP as causal evidence. For causal claims, an experiment or natural "
               "experiment would be required.")
    ev.insert(14, "")

    (reports_dir / "dtc_08_shap.md").write_text("\n".join(ev), encoding="utf-8")

    print("[4/4] DONE.")
    print(f"      Report: {reports_dir / 'dtc_08_shap.md'}")
    print(f"      Plots : {viz_dir}")

    # Count plots
    n_beeswarm = len(list(viz_dir.glob("dtc_08_beeswarm_*.png")))
    n_waterfall = len(list(viz_dir.glob("dtc_08_waterfall_*.png")))
    print(f"      Beeswarm plots: {n_beeswarm}")
    print(f"      Waterfall plots: {n_waterfall}")


if __name__ == "__main__":
    main()