"""
Stage 8 - Model Training.

Strategy:
  * Per-city models (currency-safe, per decision_log #12).
  * GroupShuffleSplit on host_id (differentiator #1) as the honest split.
  * Random train_test_split as a comparison to quantify host-level leakage.
  * 4 models per city per split: dummy, linear_regression, random_forest,
    hist_gradient_boosting.

Saves:
  models/dtc_preprocessor.joblib
      The ColumnTransformer fitted on ALL data (for later inference).
      NOTE: for the honest grouped-split evaluation, the preprocessor
      is refitted inside each split's pipeline. The saved object is
      for Stage 11 inference on NEW listings, where full-data fit is
      correct.
  models/dtc_{city_slug}_{model}_{split}.joblib
      Fitted model per (city, model_type, split).
  models/dtc_{city_slug}_best_{split}.joblib
      The best model per (city, split) by test MAE.

Writes:
  reports/dtc_06_model_comparison.md
      Per-city, per-split, per-model metrics (MAE / RMSE / R²).
  reports/dtc_06_split_comparison.md
      Differentiator #1 evidence: grouped vs random split gap.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Silence sklearn's convergence warnings — we'll report metrics regardless
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def city_slug(city: str) -> str:
    """Filesystem-safe slug for a city name."""
    return city.lower().replace(" ", "_").replace("/", "_")


# ---- Feature column definitions (from config, but reproduced for clarity) ----
LOW_CARD_CATS = [
    "room_type",
    "property_type",
    "host_response_time",
    "host_is_superhost",
    "host_identity_verified",
    "instant_bookable",
]
HIGH_CARD_CATS = ["neighbourhood"]


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Reproducible ColumnTransformer — one per city's model pipeline."""
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
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
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


def make_model(model_key: str, random_state: int):
    """Return an unfitted model given its key."""
    if model_key == "dummy":
        return DummyRegressor(strategy="median")
    if model_key == "linear_regression":
        return LinearRegression()
    if model_key == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_key == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=None,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_one_city(
    df_city: pd.DataFrame,
    city: str,
    numeric_features: list[str],
    cfg: dict,
    models_dir: Path,
) -> list[dict]:
    """Train all models (both splits) for one city. Return metrics rows."""
    target = "price_clean"
    group_key = cfg["split"]["group_by"]
    random_state = cfg["models"]["random_state"]
    test_size = cfg["split"]["test_size"]

    cat_features = LOW_CARD_CATS + HIGH_CARD_CATS
    feature_cols = numeric_features + cat_features

    X = df_city[feature_cols].copy()
    y = df_city[target].to_numpy()
    groups = df_city[group_key].to_numpy()

    n = len(df_city)
    slug = city_slug(city)

    # ---- GROUPED split (differentiator #1 — honest split) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    g_train_idx, g_test_idx = next(gss.split(X, y, groups=groups))

    # ---- RANDOM split (comparison — leaks host info) ----
    r_train_idx, r_test_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=random_state,
    )

    split_indices = {
        "grouped": (g_train_idx, g_test_idx),
        "random":  (r_train_idx, r_test_idx),
    }

    results: list[dict] = []
    best_per_split: dict[str, dict] = {}

    for split_name, (train_idx, test_idx) in split_indices.items():
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_key in cfg["models"]["train_list"]:
            t0 = time.time()
            pre = build_preprocessor(numeric_features)
            model = make_model(model_key, random_state)
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            elapsed = time.time() - t0

            row = {
                "city": city,
                "city_slug": slug,
                "split": split_name,
                "model": model_key,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "fit_seconds": round(elapsed, 1),
                **metrics,
            }
            results.append(row)

            model_path = models_dir / f"dtc_{slug}_{model_key}_{split_name}.joblib"
            joblib.dump(pipe, model_path)

            if (split_name not in best_per_split
                    or metrics["mae"] < best_per_split[split_name]["mae"]):
                best_per_split[split_name] = {
                    "model": model_key,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "pipe": pipe,
                }

    # Save best per split
    for split_name, info in best_per_split.items():
        best_path = models_dir / f"dtc_{slug}_best_{split_name}.joblib"
        joblib.dump(info["pipe"], best_path)

    return results


def main() -> None:
    cfg = load_config()

    data_path = ROOT / cfg["paths"]["listings_clean"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"ERROR: cleaned data not found: {data_path}")

    print("[1/4] Loading cleaned data ...")
    df = pd.read_parquet(data_path)
    print(f"      Rows: {len(df):,}  |  Cities: {df['city'].nunique()}")

    numeric_features = [f for f in cfg["features"]["numeric"]]
    print(f"      Numeric features ({len(numeric_features)}): {numeric_features}")

    print("\n[2/4] Training models per city (this may take 8-15 minutes) ...")
    all_results: list[dict] = []
    city_list = sorted(df["city"].dropna().unique())
    total = len(city_list)

    for i, city in enumerate(city_list, start=1):
        df_city = df[df["city"] == city].reset_index(drop=True)
        if len(df_city) < 50:
            print(f"  [{i}/{total}] {city:20s} SKIPPED (only {len(df_city)} rows)")
            continue
        print(f"  [{i}/{total}] {city:20s} rows={len(df_city):>7,} ...", end="", flush=True)
        t0 = time.time()
        rows = train_one_city(df_city, city, numeric_features, cfg, models_dir)
        all_results.extend(rows)
        print(f" done in {time.time() - t0:.1f}s")

    print("\n[3/4] Saving preprocessor fitted on ALL data (for inference) ...")
    # ------------------------------------------------------------------
    # FIX: Load dtc_05_feature_engineering.py by FILE PATH, not by
    # package import. `src/` is not a Python package (no __init__.py),
    # so `from src.dtc_05_feature_engineering import ...` fails with
    # ModuleNotFoundError. importlib.util.spec_from_file_location
    # loads the module from its absolute path — works regardless of
    # working directory or package layout.
    # ------------------------------------------------------------------
    _spec = importlib.util.spec_from_file_location(
        "dtc_05_feature_engineering",
        ROOT / "src" / "dtc_05_feature_engineering.py",
    )
    _fe = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fe)
    X_full, _, _ = _fe.build_feature_frame(df, cfg)

    # This preprocessor is for Stage 11 inference on NEW listings.
    # It is deliberately fitted on ALL data (correct for inference).
    # The honest evaluation in the reports uses per-split pipelines.
    pre_full = build_preprocessor(numeric_features)
    pre_full.fit(X_full)
    joblib.dump(pre_full, models_dir / "dtc_preprocessor.joblib")
    print(f"      Saved to {models_dir / 'dtc_preprocessor.joblib'}")

    print("\n[4/4] Writing comparison reports ...")
    results_df = pd.DataFrame(all_results)

    # ---- Model comparison report ----
    mc_lines = ["# Model Comparison Report", ""]
    mc_lines.append(f"**Project:** {cfg['project']['name']}")
    mc_lines.append(f"**Author:** {cfg['project']['author']}")
    mc_lines.append("")
    mc_lines.append("Per-city, per-split, per-model test-set metrics.")
    mc_lines.append("")
    mc_lines.append("## Grouped split (differentiator #1 — HONEST split)")
    mc_lines.append("")
    mc_lines.append("Training and testing sets are disjoint at the **host level** — "
                    "no host appears in both sets. This is the score we trust.")
    mc_lines.append("")
    mc_lines.append("| City | Model | n_train | n_test | MAE | RMSE | R² |")
    mc_lines.append("|---|---|---|---|---|---|---|")
    g = results_df[results_df["split"] == "grouped"].sort_values(["city", "mae"])
    for _, r in g.iterrows():
        mc_lines.append(
            f"| {r['city']} | {r['model']} | {r['n_train']:,} | {r['n_test']:,} "
            f"| {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.3f} |"
        )

    mc_lines.append("")
    mc_lines.append("## Random split (comparison — leaks host information)")
    mc_lines.append("")
    mc_lines.append("Training and testing sets are split randomly, so listings "
                    "from the same host land in both sets. Test scores are "
                    "**inflated** relative to true out-of-sample performance.")
    mc_lines.append("")
    mc_lines.append("| City | Model | n_train | n_test | MAE | RMSE | R² |")
    mc_lines.append("|---|---|---|---|---|---|---|")
    rd = results_df[results_df["split"] == "random"].sort_values(["city", "mae"])
    for _, r in rd.iterrows():
        mc_lines.append(
            f"| {r['city']} | {r['model']} | {r['n_train']:,} | {r['n_test']:,} "
            f"| {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.3f} |"
        )

    (reports_dir / "dtc_06_model_comparison.md").write_text(
        "\n".join(mc_lines), encoding="utf-8"
    )

    # ---- Split comparison report (differentiator #1 evidence) ----
    sc_lines = ["# Grouped vs Random Split Comparison", ""]
    sc_lines.append(f"**Project:** {cfg['project']['name']}")
    sc_lines.append(f"**Author:** {cfg['project']['author']}")
    sc_lines.append("")
    sc_lines.append("**Differentiator #1 evidence.** For each city and model, "
                    "compare the R² on a random split (leaks host info) to the "
                    "R² on a grouped split (honest). The gap is the amount of "
                    "leakage a random split introduces.")
    sc_lines.append("")
    sc_lines.append("## Table: R² gap per (city, model)")
    sc_lines.append("")
    sc_lines.append("| City | Model | R² (random) | R² (grouped) | Δ R² (leakage) |")
    sc_lines.append("|---|---|---|---|---|")

    pivot = results_df.pivot_table(
        index=["city", "model"], columns="split", values="r2"
    ).reset_index()
    pivot["delta"] = pivot["random"] - pivot["grouped"]
    pivot = pivot.sort_values("delta", ascending=False)
    for _, r in pivot.iterrows():
        sc_lines.append(
            f"| {r['city']} | {r['model']} "
            f"| {r['random']:.3f} | {r['grouped']:.3f} | {r['delta']:+.3f} |"
        )

    sc_lines.append("")
    sc_lines.append("## Interpretation")
    sc_lines.append("")
    sc_lines.append("- Positive Δ means the random split **overstates** model "
                    "quality — the model learned host-specific patterns that "
                    "won't generalise.")
    sc_lines.append("- A large Δ is *not* a bug in our code — it is the "
                    "measured leakage that a naive random split would introduce.")
    sc_lines.append("- This is why every downstream stage (9–12) uses the "
                    "**grouped split** as the honest benchmark.")

    (reports_dir / "dtc_06_split_comparison.md").write_text(
        "\n".join(sc_lines), encoding="utf-8"
    )

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models fitted        : {len(results_df)}")
    print(f"Model files saved    : {len(list(models_dir.glob('dtc_*_*.joblib')))}")
    print(f"Best-model files     : "
          f"{len(list(models_dir.glob('dtc_*_best_*.joblib')))}")
    print(f"Reports written      :")
    print(f"  - reports/dtc_06_model_comparison.md")
    print(f"  - reports/dtc_06_split_comparison.md")
    print("\nFirst look at grouped-split winners per city:")
    winner = g.sort_values("mae").groupby("city").first().reset_index()
    for _, r in winner.iterrows():
        print(f"  {r['city']:20s} -> {r['model']:22s} "
              f"MAE={r['mae']:.2f}  R²={r['r2']:.3f}")


if __name__ == "__main__":
    main()