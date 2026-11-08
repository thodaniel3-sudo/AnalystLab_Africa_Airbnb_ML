# Feature Engineering Report

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1
**Track:** Track C - Machine Learning

## 1. Target and grouping key

- **Target (`y`):** `price_clean` — cleaned, per-city-clipped price.
- **Grouping key:** `host_id` — used by `GroupShuffleSplit` in Stage 8.
- **Note:** `price` (raw) is retained in the parquet only for audit; it is **not** the model target.

## 2. Numeric features (12)

- `accommodates`
- `bedrooms`
- `minimum_nights`
- `maximum_nights`
- `host_total_listings_count`
- `review_scores_rating`
- `review_scores_accuracy`
- `review_scores_cleanliness`
- `review_scores_checkin`
- `review_scores_communication`
- `review_scores_location`

**Pipeline:** `SimpleImputer(strategy='median')` → `StandardScaler`

**Why median imputation:** numeric features like `bedrooms` and `review_scores_*` have 10–33% missing; median is robust to outliers (mean is not). Imputer is fitted inside the Pipeline **on the training fold only**, so no test-set statistic leaks.

## 3. Low-cardinality categoricals (6) — one-hot

- `room_type`
- `property_type`
- `host_response_time`
- `host_is_superhost`
- `host_identity_verified`
- `instant_bookable`

**Pipeline:** `SimpleImputer('constant', 'unknown')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

## 4. High-cardinality categorical (1) — ordinal

- `neighbourhood`

**Pipeline:** `SimpleImputer('constant', 'unknown')` → `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`

**Why ordinal, not one-hot:** `neighbourhood` has hundreds of unique values. One-hot would create ~hundreds of columns and increase memory/compute drastically. Tree-based models (Random Forest, HistGradientBoosting) handle ordinal encodings of high-cardinality categories well because they can split on arbitrary thresholds during training.

## 5. Excluded from features — `city`

`city` is the **partitioning key** for per-city modeling (see decision_log #12). It is not a feature. Each city's model trains only on that city's rows, so `city` would be constant within any single model — a useless feature.

## 6. Pipeline output shape

- Fit-and-transform on a 5,000-row sample succeeded.
- Resulting feature matrix: **95 columns**
- These columns are exactly what `train_model.py` will feed into the regression models.

## 7. Leakage prevention

- `ColumnTransformer` is **unfitted** at the end of this stage. Fitting happens in `train_model.py` **inside** the sklearn Pipeline, so it only sees training-fold data.
- `price_clean` is the target — excluded from `X` by construction.
- `price` (raw) is excluded from `X` by construction (we only select the declared feature columns).
- `review_scores_value` was excluded in the leakage audit (see `reports/leakage_audit.md`).
- `latitude`/`longitude` were excluded in the leakage audit.

## 8. Reproducibility

- The `ColumnTransformer` is a plain object — saved to `models/preprocessing_pipeline.joblib` in Stage 8 after fitting, and reloaded in Stage 11 for inference.
- No hard-coded hyperparameters: all feature lists come from `config.yaml`.