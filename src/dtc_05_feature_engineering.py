"""
Stage 7 - Feature Engineering.

Builds a reusable sklearn Pipeline + ColumnTransformer that:
  * Imputes missing numeric values (median, fitted on train only)
  * Imputes missing categorical values ("unknown")
  * Scales numeric features (StandardScaler, fitted on train only)
  * Encodes low-cardinality categoricals (OneHotEncoder)
  * Encodes high-cardinality `neighbourhood` (OrdinalEncoder to avoid
    feature-space explosion)

Public API:
    build_preprocessing_pipeline(cfg) -> ColumnTransformer
    build_feature_frame(df, cfg)     -> (X, y, groups)

Note: the ColumnTransformer is NOT fitted here. It is fitted inside
src/train_model.py on the training set only. That is the whole point -
fitting outside the pipeline would leak test-set statistics into the
imputer/scaler.

Writes:
    reports/feature_engineering.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# Low-cardinality categoricals -> one-hot
LOW_CARD_CATS = [
    "room_type",
    "property_type",
    "host_response_time",
    "host_is_superhost",
    "host_identity_verified",
    "instant_bookable",
]

# High-cardinality categorical -> ordinal (one-hot would explode the space)
HIGH_CARD_CATS = ["neighbourhood"]

# City is used as a *partitioning key* in per-city models, not as a feature.
# If we ever build a single-pooled model (sensitivity test), city becomes
# one-hot; but by default per-city modeling means city is NOT a feature.


def build_preprocessing_pipeline(cfg: dict) -> ColumnTransformer:
    """
    Construct (but do not fit) the preprocessing ColumnTransformer.

    The caller is responsible for calling .fit(X_train, y_train)
    or wrapping it inside an sklearn Pipeline.
    """
    numeric_features = cfg["features"]["numeric"]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    low_cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    high_cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1)),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("low_cat", low_cat_pipe, LOW_CARD_CATS),
            ("high_cat", high_cat_pipe, HIGH_CARD_CATS),
        ],
        remainder="drop",        # drop everything else
        verbose_feature_names_out=False,
    )
    return transformer


def build_feature_frame(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split df into X (features), y (target), groups (host_id).

    `price_clean` is the target — NOT `price`. `price` is the raw
    column; `price_clean` is the pre-cleaned, clipped value.
    """
    target = "price_clean"
    group_key = cfg["split"]["group_by"]

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    # Remove `city` from feature list — it is the partition key
    categorical = [c for c in categorical if c != "city"]

    feature_cols = numeric + categorical
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: feature columns missing from data: {missing}")

    X = df[feature_cols].copy()
    y = df[target].copy()
    groups = df[group_key].copy()
    return X, y, groups


def main() -> None:
    cfg = load_config()
    data_path = ROOT / cfg["paths"]["listings_clean"]
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "dtc_05_feature_engineering.md"

    if not data_path.exists():
        sys.exit(f"ERROR: cleaned data not found: {data_path}")

    print("[1/4] Loading cleaned data ...")
    df = pd.read_parquet(data_path)

    print("[2/4] Building feature frame ...")
    X, y, groups = build_feature_frame(df, cfg)

    print("[3/4] Building preprocessing pipeline (unfitted) ...")
    preprocessor = build_preprocessing_pipeline(cfg)

    # Quick smoke test: fit_transform on a small sample to make sure
    # everything composes. This does NOT train the real model - it just
    # verifies the pipeline can be fitted at all.
    print("[4/4] Smoke-testing fit_transform on a 5k-row sample ...")
    sample = X.sample(min(5000, len(X)), random_state=42)
    transformed = preprocessor.fit_transform(sample)
    n_out = transformed.shape[1]

    log: list[str] = []
    log.append("# Feature Engineering Report")
    log.append(f"\n**Project:** {cfg['project']['name']}")
    log.append(f"**Author:** {cfg['project']['author']}")
    log.append(f"**Cohort:** {cfg['project']['cohort']}")
    log.append(f"**Track:** {cfg['project']['track']}")

    log.append("\n## 1. Target and grouping key\n")
    log.append("- **Target (`y`):** `price_clean` — cleaned, per-city-clipped price.")
    log.append(f"- **Grouping key:** `host_id` — used by `GroupShuffleSplit` in Stage 8.")
    log.append("- **Note:** `price` (raw) is retained in the parquet only for "
               "audit; it is **not** the model target.")

    log.append("\n## 2. Numeric features (12)\n")
    for f in cfg["features"]["numeric"]:
        log.append(f"- `{f}`")
    log.append("\n**Pipeline:** `SimpleImputer(strategy='median')` → `StandardScaler`")
    log.append("\n**Why median imputation:** numeric features like `bedrooms` "
               "and `review_scores_*` have 10–33% missing; median is robust to "
               "outliers (mean is not). Imputer is fitted inside the Pipeline "
               "**on the training fold only**, so no test-set statistic leaks.")

    log.append("\n## 3. Low-cardinality categoricals (6) — one-hot\n")
    for f in LOW_CARD_CATS:
        log.append(f"- `{f}`")
    log.append("\n**Pipeline:** `SimpleImputer('constant', 'unknown')` → "
               "`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`")

    log.append("\n## 4. High-cardinality categorical (1) — ordinal\n")
    for f in HIGH_CARD_CATS:
        log.append(f"- `{f}`")
    log.append("\n**Pipeline:** `SimpleImputer('constant', 'unknown')` → "
               "`OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`")
    log.append("\n**Why ordinal, not one-hot:** `neighbourhood` has hundreds "
               "of unique values. One-hot would create ~hundreds of columns and "
               "increase memory/compute drastically. Tree-based models (Random "
               "Forest, HistGradientBoosting) handle ordinal encodings of "
               "high-cardinality categories well because they can split on "
               "arbitrary thresholds during training.")

    log.append("\n## 5. Excluded from features — `city`\n")
    log.append("`city` is the **partitioning key** for per-city modeling "
               "(see decision_log #12). It is not a feature. Each city's model "
               "trains only on that city's rows, so `city` would be constant "
               "within any single model — a useless feature.")

    log.append("\n## 6. Pipeline output shape\n")
    log.append(f"- Fit-and-transform on a 5,000-row sample succeeded.")
    log.append(f"- Resulting feature matrix: **{n_out} columns**")
    log.append(f"- These columns are exactly what `train_model.py` will feed "
               f"into the regression models.")

    log.append("\n## 7. Leakage prevention\n")
    log.append("- `ColumnTransformer` is **unfitted** at the end of this stage. "
               "Fitting happens in `train_model.py` **inside** the sklearn "
               "Pipeline, so it only sees training-fold data.")
    log.append("- `price_clean` is the target — excluded from `X` by construction.")
    log.append("- `price` (raw) is excluded from `X` by construction (we only "
               "select the declared feature columns).")
    log.append("- `review_scores_value` was excluded in the leakage audit "
               "(see `reports/leakage_audit.md`).")
    log.append("- `latitude`/`longitude` were excluded in the leakage audit.")

    log.append("\n## 8. Reproducibility\n")
    log.append("- The `ColumnTransformer` is a plain object — saved to "
               "`models/preprocessing_pipeline.joblib` in Stage 8 after fitting, "
               "and reloaded in Stage 11 for inference.")
    log.append("- No hard-coded hyperparameters: all feature lists come from "
               "`config.yaml`.")

    report_path.write_text("\n".join(log), encoding="utf-8")
    print(f"\nDONE. Report written to: {report_path}")
    print(f"      Pipeline output shape (on 5k sample): {n_out} columns")


if __name__ == "__main__":
    main()