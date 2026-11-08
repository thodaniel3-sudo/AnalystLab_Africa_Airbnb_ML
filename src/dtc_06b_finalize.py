"""
Stage 8b - Finalize training.

Recovers from a failure in the [3/4] step of dtc_06_train_model.py.
Does NOT retrain any models. It:
  1. Loads the already-saved per-city per-model per-split metrics
     by re-fitting each saved pipeline on the SAME split (recovered
     from a deterministic recomputation of the split indices).
  2. Saves the full-data inference preprocessor.
  3. Writes both comparison reports.

Runs in ~90 seconds (no model training).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_").replace("/", "_")


LOW_CARD_CATS = [
    "room_type", "property_type", "host_response_time",
    "host_is_superhost", "host_identity_verified", "instant_bookable",
]
HIGH_CARD_CATS = ["neighbourhood"]


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("low_cat", low_cat_pipe, LOW_CARD_CATS),
            ("high_cat", high_cat_pipe, HIGH_CARD_CATS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def main() -> None:
    cfg = load_config()
    data_path = ROOT / cfg["paths"]["listings_clean"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    reports_dir = ROOT / cfg["paths"]["reports_dir"]

    if not data_path.exists():
        sys.exit(f"ERROR: cleaned data not found: {data_path}")

    print("[1/4] Loading cleaned data ...")
    df = pd.read_parquet(data_path)
    print(f"      Rows: {len(df):,}  |  Cities: {df['city'].nunique()}")

    numeric_features = cfg["features"]["numeric"]
    cat_features = LOW_CARD_CATS + HIGH_CARD_CATS
    feature_cols = numeric_features + cat_features
    target = "price_clean"
    group_key = cfg["split"]["group_by"]
    random_state = cfg["models"]["random_state"]
    test_size = cfg["split"]["test_size"]

    # ---- [2/4] Recover per-city metrics from saved models ----
    print("\n[2/4] Recomputing test metrics from saved models ...")
    all_results: list[dict] = []
    city_list = sorted(df["city"].dropna().unique())
    total = len(city_list)

    for i, city in enumerate(city_list, start=1):
        df_city = df[df["city"] == city].reset_index(drop=True)
        if len(df_city) < 50:
            print(f"  [{i}/{total}] {city:20s} SKIPPED")
            continue

        slug = city_slug(city)
        X = df_city[feature_cols].copy()
        y = df_city[target].to_numpy()
        groups = df_city[group_key].to_numpy()
        n = len(df_city)

        # Recover split indices — same seed, same indices as training
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        g_train_idx, g_test_idx = next(gss.split(X, y, groups=groups))
        r_train_idx, r_test_idx = train_test_split(
            np.arange(n), test_size=test_size, random_state=random_state,
        )
        split_indices = {
            "grouped": (g_train_idx, g_test_idx),
            "random":  (r_train_idx, r_test_idx),
        }

        for split_name, (tr_idx, te_idx) in split_indices.items():
            X_test = X.iloc[te_idx]
            y_test = y[te_idx]
            for model_key in cfg["models"]["train_list"]:
                model_path = models_dir / f"dtc_{slug}_{model_key}_{split_name}.joblib"
                if not model_path.exists():
                    print(f"    WARNING: missing {model_path.name}")
                    continue
                pipe = joblib.load(model_path)
                y_pred = pipe.predict(X_test)
                m = evaluate(y_test, y_pred)
                all_results.append({
                    "city": city, "city_slug": slug, "split": split_name,
                    "model": model_key,
                    "n_train": len(tr_idx), "n_test": len(te_idx),
                    **m,
                })
        print(f"  [{i}/{total}] {city:20s} metrics recovered")

    results_df = pd.DataFrame(all_results)

    # ---- [3/4] Save full-data inference preprocessor ----
    print("\n[3/4] Building inference preprocessor on ALL data ...")
    # Import build_feature_frame by module path (not package — src/ is not a package)
    spec = importlib.util.spec_from_file_location(
        "dtc_05_feature_engineering",
        ROOT / "src" / "dtc_05_feature_engineering.py",
    )
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    X_full, _, _ = fe_mod.build_feature_frame(df, cfg)
    pre_full = build_preprocessor(numeric_features)
    t0 = time.time()
    pre_full.fit(X_full)
    joblib.dump(pre_full, models_dir / "dtc_preprocessor.joblib")
    print(f"      Saved dtc_preprocessor.joblib in {time.time() - t0:.1f}s")

    # ---- [4/4] Write comparison reports ----
    print("\n[4/4] Writing comparison reports ...")

    # Model comparison report
    mc = ["# Model Comparison Report", ""]
    mc.append(f"**Project:** {cfg['project']['name']}")
    mc.append(f"**Author:** {cfg['project']['author']}")
    mc.append("")
    mc.append("Per-city, per-split, per-model test-set metrics.")
    mc.append("")
    mc.append("## Grouped split (differentiator #1 — HONEST split)")
    mc.append("")
    mc.append("Training and testing sets are disjoint at the **host level** — "
              "no host appears in both. This is the score we trust.")
    mc.append("")
    mc.append("| City | Model | n_train | n_test | MAE | RMSE | R² |")
    mc.append("|---|---|---|---|---|---|---|")
    g = results_df[results_df["split"] == "grouped"].sort_values(["city", "mae"])
    for _, r in g.iterrows():
        mc.append(f"| {r['city']} | {r['model']} | {r['n_train']:,} | "
                  f"{r['n_test']:,} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.3f} |")
    mc.append("")
    mc.append("## Random split (comparison — leaks host information)")
    mc.append("")
    mc.append("Training and testing sets are split randomly, so listings from the "
              "same host land in both. Test scores are **inflated** relative to "
              "true out-of-sample performance.")
    mc.append("")
    mc.append("| City | Model | n_train | n_test | MAE | RMSE | R² |")
    mc.append("|---|---|---|---|---|---|---|")
    rd = results_df[results_df["split"] == "random"].sort_values(["city", "mae"])
    for _, r in rd.iterrows():
        mc.append(f"| {r['city']} | {r['model']} | {r['n_train']:,} | "
                  f"{r['n_test']:,} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.3f} |")
    (reports_dir / "dtc_06_model_comparison.md").write_text(
        "\n".join(mc), encoding="utf-8"
    )

    # Split comparison report
    sc = ["# Grouped vs Random Split Comparison", ""]
    sc.append(f"**Project:** {cfg['project']['name']}")
    sc.append(f"**Author:** {cfg['project']['author']}")
    sc.append("")
    sc.append("**Differentiator #1 evidence.** For each city and model, compare "
              "R² on a random split (leaks host info) to R² on a grouped split "
              "(honest). The gap is the amount of leakage.")
    sc.append("")
    sc.append("## R² gap per (city, model)")
    sc.append("")
    sc.append("| City | Model | R² (random) | R² (grouped) | Δ R² (leakage) |")
    sc.append("|---|---|---|---|---|")
    pivot = results_df.pivot_table(
        index=["city", "model"], columns="split", values="r2"
    ).reset_index()
    pivot["delta"] = pivot["random"] - pivot["grouped"]
    pivot = pivot.sort_values("delta", ascending=False)
    for _, r in pivot.iterrows():
        sc.append(f"| {r['city']} | {r['model']} | {r['random']:.3f} | "
                  f"{r['grouped']:.3f} | {r['delta']:+.3f} |")
    sc.append("")
    sc.append("## Interpretation")
    sc.append("")
    sc.append("- Positive Δ = random split **overstates** model quality.")
    sc.append("- Large Δ is *expected* — it measures the leakage that a naive "
              "random split introduces.")
    sc.append("- This is why Stages 9–12 use the **grouped split** as the "
              "honest benchmark.")
    (reports_dir / "dtc_06_split_comparison.md").write_text(
        "\n".join(sc), encoding="utf-8"
    )

    # Summary
    print("\n" + "=" * 60)
    print("FINALIZE COMPLETE")
    print("=" * 60)
    print(f"Models re-evaluated : {len(results_df)}")
    print(f"Preprocessor saved  : dtc_preprocessor.joblib")
    print(f"Reports written     :")
    print(f"  - reports/dtc_06_model_comparison.md")
    print(f"  - reports/dtc_06_split_comparison.md")
    print("\nGrouped-split winners per city:")
    winner = g.sort_values("mae").groupby("city").first().reset_index()
    for _, r in winner.iterrows():
        print(f"  {r['city']:20s} -> {r['model']:22s} "
              f"MAE={r['mae']:.2f}  R²={r['r2']:.3f}")


if __name__ == "__main__":
    main()