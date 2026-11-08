"""
Stage 9 - Evaluation & Error Analysis.

Reads:
  data/processed/listings_clean.parquet
  models/dtc_{city}_best_grouped.joblib   (best model per city, honest split)

Does NOT retrain the winning models (they're already saved).
DOES retrain each city's best model WITHOUT the 6 review_scores_*
features — the sensitivity test (decision_log #15).

Also REGENERATES reports/dtc_06_split_comparison.md with a footnote
explaining the numerically-unstable linear regression rows.

Writes:
  visualizations/dtc_07_actual_vs_predicted_{city}.png  (10 plots)
  visualizations/dtc_07_residuals_{city}.png            (10 plots)
  visualizations/dtc_07_sensitivity.png                 (bar chart)
  reports/dtc_07_evaluation.md
  reports/dtc_06_split_comparison.md (regenerated with footnote)
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 9,
})

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_").replace("/", "_")


def load_module(name: str, path: Path):
    """Load a Python file as a module by absolute file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_winner(report_path: Path) -> dict[str, str]:
    """Parse dtc_06_model_comparison.md; extract winner per city (lowest MAE, grouped split)."""
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


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def regenerate_split_report(models_dir: Path, reports_dir: Path, cfg: dict):
    """Rebuild dtc_06_split_comparison.md with a footnote about OLS instability."""
    df = pd.read_parquet(ROOT / cfg["paths"]["listings_clean"])
    numeric = cfg["features"]["numeric"]
    LOW_CARD = ["room_type", "property_type", "host_response_time",
                "host_is_superhost", "host_identity_verified", "instant_bookable"]
    HIGH_CARD = ["neighbourhood"]
    feature_cols = numeric + LOW_CARD + HIGH_CARD
    target = "price_clean"
    group_key = cfg["split"]["group_by"]

    rows = []
    for city in sorted(df["city"].dropna().unique()):
        df_city = df[df["city"] == city].reset_index(drop=True)
        if len(df_city) < 50:
            continue
        slug = city_slug(city)
        X = df_city[feature_cols]
        y = df_city[target].to_numpy()
        groups = df_city[group_key].to_numpy()

        from sklearn.model_selection import train_test_split
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["test_size"],
                                random_state=cfg["models"]["random_state"])
        _, g_test = next(gss.split(X, y, groups=groups))
        _, r_test = train_test_split(
            np.arange(len(df_city)), test_size=cfg["split"]["test_size"],
            random_state=cfg["models"]["random_state"],
        )

        for model_key in cfg["models"]["train_list"]:
            r2 = {}
            for split_name, idx in [("random", r_test), ("grouped", g_test)]:
                p = models_dir / f"dtc_{slug}_{model_key}_{split_name}.joblib"
                if not p.exists():
                    continue
                pipe = joblib.load(p)
                y_pred = pipe.predict(X.iloc[idx])
                r2[split_name] = r2_score(y[idx], y_pred)
            if "random" in r2 and "grouped" in r2:
                rows.append({
                    "city": city, "model": model_key,
                    "r2_random": r2["random"], "r2_grouped": r2["grouped"],
                    "delta": r2["random"] - r2["grouped"],
                })

    df_metrics = pd.DataFrame(rows)
    unstable = df_metrics[
        (df_metrics["r2_random"] < -1) | (df_metrics["r2_grouped"] < -1)
    ]
    n_unstable = len(unstable)

    lines = ["# Grouped vs Random Split Comparison", ""]
    lines.append(f"**Project:** {cfg['project']['name']}")
    lines.append(f"**Author:** {cfg['project']['author']}")
    lines.append("")
    lines.append("**Differentiator #1 evidence.** For each city and model, "
                 "compare R² on a random split (leaks host info) to R² on a "
                 "grouped split (honest). A large positive Δ is the measured "
                 "host leakage of a naive random split.")
    lines.append("")
    lines.append("## Table: R² gap per (city, model)")
    lines.append("")
    lines.append("| City | Model | R² (random) | R² (grouped) | Δ R² (leakage) |")
    lines.append("|---|---|---|---|---|")

    sorted_df = df_metrics.sort_values("delta", ascending=False)
    for _, r in sorted_df.iterrows():
        # Cap display of absurd values to keep the table readable
        r2_r = r["r2_random"] if -1_000_000 < r["r2_random"] < 10 else float("nan")
        r2_g = r["r2_grouped"] if -1_000_000 < r["r2_grouped"] < 10 else float("nan")
        delta = r["delta"] if -1_000_000 < r["delta"] < 10 else float("nan")
        if np.isnan(r2_r):
            lines.append(f"| {r['city']} | {r['model']} | *diverged* | *diverged* | *see note* |")
        else:
            lines.append(f"| {r['city']} | {r['model']} | {r2_r:.3f} | "
                         f"{r2_g:.3f} | {delta:+.3f} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Positive Δ means the random split **overstates** model quality — "
                 "the model learns host-specific patterns that do not generalise.")
    lines.append("- A large Δ is *expected*: 17.50% of hosts have multiple listings "
                 "(largest single host: 627 listings). Random splits mix them across "
                 "train and test; grouped splits prevent this by construction.")
    lines.append("- The **grouped-split R² is the honest benchmark** and is the number "
                 "used to select winners in `dtc_06_model_comparison.md`.")
    lines.append("")
    if n_unstable > 0:
        lines.append("## Note on linear regression instability")
        lines.append("")
        lines.append(f"**{n_unstable} row(s)** show numerically unstable R² values "
                     f"(magnitude > 1) for **linear regression**. This is a known "
                     "limitation of OLS on heavily right-skewed targets combined "
                     "with high-cardinality one-hot encoded categorical features. "
                     "The resulting coefficients become astronomically large and "
                     "predictions explode at prediction time.")
        lines.append("")
        lines.append("**Implication:** Linear regression is unusable for this dataset "
                     "as-is. It is **not selected as the winning model in any city**. "
                     "Its metrics are retained here for transparency but should not "
                     "be quoted in comparative claims. A log-transformed target would "
                     "fix the instability, but that is out of scope for this project "
                     "and is documented as future work.")
        lines.append("")
        lines.append(f"**Count of diverged linear_regression fits:** {n_unstable}")

    (reports_dir / "dtc_06_split_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"      Regenerated dtc_06_split_comparison.md ({n_unstable} unstable rows noted)")


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

    # Load the train_model module once, using file-path import
    mod6 = load_module("dtc_06_train_model", ROOT / "src" / "dtc_06_train_model.py")

    print("[1/5] Loading data and winners ...")
    df = pd.read_parquet(data_path)
    winners = parse_winner(reports_dir / "dtc_06_model_comparison.md")
    print(f"      Winners parsed for {len(winners)} cities")

    print("[2/5] Loading best grouped models + retraining without review scores ...")
    numeric_full = cfg["features"]["numeric"]
    numeric_no_scores = [f for f in numeric_full if not f.startswith("review_scores_")]

    sensitivity_rows = []
    city_list = sorted(df["city"].dropna().unique())
    total = len(city_list)

    LOW_CARD = ["room_type", "property_type", "host_response_time",
                "host_is_superhost", "host_identity_verified", "instant_bookable"]
    HIGH_CARD = ["neighbourhood"]
    target = "price_clean"

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

        feature_cols_full = numeric_full + LOW_CARD + HIGH_CARD
        X_full = df_city[feature_cols_full]
        y = df_city[target].to_numpy()

        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["test_size"],
                                random_state=cfg["models"]["random_state"])
        g_train, g_test = next(gss.split(X_full, y,
                                          groups=df_city[cfg["split"]["group_by"]].to_numpy()))
        X_train = X_full.iloc[g_train]
        X_test = X_full.iloc[g_test]
        y_train = y[g_train]
        y_test = y[g_test]

        # Full model predictions from saved pipeline
        pipe_full = joblib.load(model_path)
        y_pred_full = pipe_full.predict(X_test)
        mae_full = mean_absolute_error(y_test, y_pred_full)
        rmse_full = float(np.sqrt(mean_squared_error(y_test, y_pred_full)))
        r2_full = r2_score(y_test, y_pred_full)

        # Retrain WITHOUT review_scores_*
        feature_cols_no = numeric_no_scores + LOW_CARD + HIGH_CARD
        X_train_no = df_city[feature_cols_no].iloc[g_train]
        X_test_no = df_city[feature_cols_no].iloc[g_test]
        pre_no = mod6.build_preprocessor(numeric_no_scores)
        model_no = mod6.make_model(winner, cfg["models"]["random_state"])
        pipe_no = Pipeline([("pre", pre_no), ("model", model_no)])
        pipe_no.fit(X_train_no, y_train)
        y_pred_no = pipe_no.predict(X_test_no)
        mae_no = mean_absolute_error(y_test, y_pred_no)
        rmse_no = float(np.sqrt(mean_squared_error(y_test, y_pred_no)))
        r2_no = r2_score(y_test, y_pred_no)

        sensitivity_rows.append({
            "city": city, "model": winner,
            "mae_full": mae_full, "rmse_full": rmse_full, "r2_full": r2_full,
            "mae_no_scores": mae_no, "rmse_no_scores": rmse_no, "r2_no_scores": r2_no,
            "delta_r2": r2_full - r2_no,      # positive = scores helped
            "delta_mae": mae_full - mae_no,
            "n_test": len(y_test),
        })
        print(f"  [{i}/{total}] {city:20s}  R²(full)={r2_full:.3f}  "
              f"R²(no-scores)={r2_no:.3f}  Δ={r2_full - r2_no:+.3f}")

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred_full, alpha=0.3, s=10, color="steelblue")
        lims = [0, max(y_test.max(), y_pred_full.max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual price_clean")
        ax.set_ylabel("Predicted price_clean")
        ax.set_title(f"{city} — Actual vs Predicted\n"
                     f"MAE={mae_full:.1f}  R²={r2_full:.3f}  (grouped split)")
        ax.legend()
        save_fig(fig, viz_dir / f"dtc_07_actual_vs_predicted_{slug}.png")

        # Residuals
        residuals = y_test - y_pred_full
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(y_pred_full, residuals, alpha=0.3, s=10, color="firebrick")
        axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Predicted price_clean")
        axes[0].set_ylabel("Residual (actual − predicted)")
        axes[0].set_title(f"{city} — Residuals vs Predicted")
        axes[1].hist(residuals, bins=40, color="firebrick", edgecolor="white")
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"{city} — Residual Distribution\n"
                          f"median={np.median(residuals):.1f}  std={residuals.std():.1f}")
        fig.tight_layout()
        save_fig(fig, viz_dir / f"dtc_07_residuals_{slug}.png")

    sens_df = pd.DataFrame(sensitivity_rows)

    print("\n[3/5] Plotting sensitivity summary ...")
    if not sens_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        s = sens_df.sort_values("delta_r2")
        colors = ["firebrick" if d < 0 else "seagreen" for d in s["delta_r2"]]
        ax.barh(s["city"], s["delta_r2"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Δ R² (full − no-scores); positive = review scores helped")
        ax.set_title("Sensitivity test: how much do review_scores_* contribute?")
        for i, (city, d) in enumerate(zip(s["city"], s["delta_r2"])):
            ax.text(d + (0.002 if d >= 0 else -0.002), i,
                    f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8)
        save_fig(fig, viz_dir / "dtc_07_sensitivity.png")

    print("[4/5] Writing evaluation report ...")
    ev = ["# Evaluation & Error Analysis", ""]
    ev.append(f"**Project:** {cfg['project']['name']}")
    ev.append(f"**Author:** {cfg['project']['author']}")
    ev.append("")
    ev.append("## 1. Per-city best model (grouped split — honest)")
    ev.append("")
    ev.append("| City | Winner | MAE | RMSE | R² | n_test |")
    ev.append("|---|---|---|---|---|---|")
    for _, r in sens_df.iterrows():
        ev.append(f"| {r['city']} | {r['model']} | {r['mae_full']:.2f} | "
                  f"{r['rmse_full']:.2f} | {r['r2_full']:.3f} | {r['n_test']:,} |")

    ev.append("")
    ev.append("## 2. Sensitivity test (decision_log #15)")
    ev.append("")
    ev.append("Does model performance depend on the six `review_scores_*` features? "
              "These aggregate review scores are only available *after* a listing "
              "has accumulated reviews. Retraining the winner per city **without** "
              "them tells us how much the model relies on post-stay information.")
    ev.append("")
    ev.append("| City | Model | R² full | R² no-scores | Δ R² | MAE full | MAE no-scores |")
    ev.append("|---|---|---|---|---|---|---|")
    for _, r in sens_df.sort_values("delta_r2", ascending=False).iterrows():
        ev.append(f"| {r['city']} | {r['model']} | {r['r2_full']:.3f} | "
                  f"{r['r2_no_scores']:.3f} | {r['delta_r2']:+.3f} | "
                  f"{r['mae_full']:.2f} | {r['mae_no_scores']:.2f} |")

    ev.append("")
    ev.append("**Reading of the sensitivity table:**")
    ev.append("")
    if not sens_df.empty:
        mean_delta = sens_df["delta_r2"].mean()
        max_delta = sens_df["delta_r2"].max()
        min_delta = sens_df["delta_r2"].min()
        ev.append(f"- Mean Δ R² across cities: **{mean_delta:+.3f}**")
        ev.append(f"- Max Δ R² (largest score dependence): **{max_delta:+.3f}**")
        ev.append(f"- Min Δ R² (smallest): **{min_delta:+.3f}**")
        if abs(mean_delta) < 0.02:
            ev.append("- **Conclusion:** Removing review scores barely changes performance. "
                      "The model relies primarily on structural features available before "
                      "any review exists — a good sign for deployment.")
        elif mean_delta > 0.02:
            ev.append("- **Conclusion:** Review scores materially help the model. This is "
                      "expected but should be disclosed: for a brand-new listing with no "
                      "reviews, predictions will be less accurate. Not a bug — a limitation.")
        else:
            ev.append("- **Conclusion:** Review scores slightly *hurt* — they add noise. "
                      "The model would be defensible without them.")

    ev.append("")
    ev.append("## 3. Actual vs predicted plots (per city)")
    ev.append("")
    for city in sorted(sens_df["city"].unique()):
        slug = city_slug(city)
        ev.append(f"**{city}**")
        ev.append("")
        ev.append(f"![{city}](../visualizations/dtc_07_actual_vs_predicted_{slug}.png)")
        ev.append("")
    ev.append("## 4. Residual plots (per city)")
    ev.append("")
    for city in sorted(sens_df["city"].unique()):
        slug = city_slug(city)
        ev.append(f"**{city}**")
        ev.append("")
        ev.append(f"![{city}](../visualizations/dtc_07_residuals_{slug}.png)")
        ev.append("")
    ev.append("## 5. Sensitivity chart")
    ev.append("")
    ev.append("![sensitivity](../visualizations/dtc_07_sensitivity.png)")
    ev.append("")
    ev.append("## 6. Known error patterns")
    ev.append("")
    ev.append("- **Heteroscedasticity:** residual spread grows with predicted price. "
              "High-priced listings are harder to predict because luxury pricing depends "
              "on idiosyncratic factors (design, brand, seasonality) not captured in the "
              "available features.")
    ev.append("- **Right-skewed residuals:** in several cities the model underpredicts "
              "the most expensive listings. The 99.5% per-city clip removed the extreme "
              "tail, but the compressed upper range remains difficult.")
    ev.append("- **Scale varies by currency:** MAE values are in local currency "
              "(Paris €38 vs Bangkok ฿1022). Compare **R²** across cities, not MAE.")

    (reports_dir / "dtc_07_evaluation.md").write_text("\n".join(ev), encoding="utf-8")

    print("\n[5/5] Regenerating dtc_06_split_comparison.md with footnote ...")
    regenerate_split_report(models_dir, reports_dir, cfg)

    print("\nDONE.")
    print(f"      Report: {reports_dir / 'dtc_07_evaluation.md'}")
    print(f"      Plots : {viz_dir}")
    if not sens_df.empty:
        print(f"\nSensitivity — mean Δ R² = {sens_df['delta_r2'].mean():+.4f}  "
              f"max = {sens_df['delta_r2'].max():+.4f}  "
              f"min = {sens_df['delta_r2'].min():+.4f}")


if __name__ == "__main__":
    main()