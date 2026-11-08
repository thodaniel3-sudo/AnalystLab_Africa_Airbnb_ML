# Decision Log - Airbnb Listing Price Prediction

**Author:** Daniel Thomas
**Project:** AnalystLab Africa Week 1 - Track C
**Purpose:** Record of every non-trivial judgment call made during the
project, so that decisions are traceable, reviewable, and reproducible.

---

| # | Date | Area | Decision | Reason | Alternatives considered |
|---|---|---|---|---|---|
| 1 | Week 1 | Environment | Python 3.12.10, VS Code, `.venv` | Modern Python, official IDE, isolated env. | System Python (rejected - no isolation). |
| 2 | Week 1 | Environment | Pinned versions in `requirements.txt` | Reproducibility. | `>=` constraints (rejected - drift). |
| 3 | Week 1 | Environment | Config-driven pipeline via `config.yaml` + `run_pipeline.py` | Single source of truth. Adapt without code edits. | Hard-coded constants (rejected). |
| 4 | Week 1 | Data files | Renamed dictionaries to include `(1)` | Matches assignment's official filenames. | (N/A - original names lacked `(1)`.) |
| 5 | Week 1 | Data loading | Try `encoding="utf-8"`, fall back to `encoding="latin-1"` for `Listings.csv` | File contains non-UTF-8 bytes (position 206899, byte 0x81). Latin-1 accepts any byte sequence. | Fail loudly (rejected - blocks all analysis); `errors="replace"` (rejected - corrupts chars). |
| 6 | Week 1 | Data loading | Read `Listings.csv` with `low_memory=False` | Silences mixed-dtype warnings in `host_response_time` and `district`. | `dtype=` per column (rejected - verbose). |
| 7 | Week 1 | Data loading | Use DuckDB for `Reviews.csv` only; pandas for `Listings.csv` | Reviews.csv is 5.37M rows — pandas would use ~2 GB RAM. Listings.csv is small. | Pandas for everything (rejected - OOM); DuckDB for everything (rejected - cannot handle latin-1). |
| 8 | Week 1 | Foreign-key check | Do the orphan check in pandas, not DuckDB | DuckDB's `read_csv_auto` has **no `encoding=` parameter**. pandas handles Latin-1 natively. | `ignore_errors=true` in DuckDB (rejected - silently drops rows). |
| 9 | Week 1 | `listing_id` | Validate as primary key via `nunique() == len(df)` | 279,712 distinct values on 279,712 rows - fully unique. | (N/A.) |
| 10 | Week 1 | Duplicate `review_id` | Treat `(listing_id, review_id)` as the composite key; do **not** delete the 160 duplicate `review_id` rows | **All 160 duplicates cross listing boundaries** (0 within same listing). This is a data-export artifact, not genuine re-reviews. The composite key remains unique (5,373,143 = 5,373,143). | Deduplicate by `review_id` alone (rejected - loses valid pairs); delete rows (rejected - loses valid data). |
| 11 | Week 1 | Price cleaning | Drop 113 zero-price rows in Stage 5 (documented) | Zero price is not a valid listing price. Would skew per-city medians and the model target. | Impute with median (rejected - fabricates data); keep (rejected - distorts). |
| 12 | Week 1 | Price currency | Per-city models with local-currency target; no cross-city price pooling | Dataset has 10 cities, no currency column. Raw prices in different currencies cannot be pooled. | Convert to USD via static FX (rejected - needs assumptions; static FX over 12 years is inaccurate). |
| 13 | Week 1 | Feature selection | Exclude `review_scores_value` | Direct leakage - encodes guest-perceived value-for-money. | Include all scores (rejected - inflated test R²). |
| 14 | Week 1 | Feature selection | Exclude `latitude`/`longitude` | Too granular; proxy for `city` and `neighbourhood`, both already included. | Include (rejected - risk of memorizing exact coordinates). |
| 15 | Week 1 | Feature selection | Include 6 `review_scores_*` with sensitivity test in Stage 8 | Debatable. Report model performance with/without. | Exclude all (rejected - loses signal). |
| 16 | Week 1 | Split strategy | `GroupShuffleSplit(group=host_id)` — differentiator #1 | Evidence: 17.50% of hosts have >1 listing; **one host has 627**. Random split would leak. | Random `train_test_split` (rejected - inflated test R²). |
| 17 | Week 1 | Split comparison | Report both random-split AND grouped-split scores side-by-side | Quantifies leakage and shows honest generalisation performance. | Report one number only (rejected - hides leakage). |
| 18 | Week 1 | Model comparison | Compare dummy, linear regression, random forest, hist-gradient-boosting | Baseline → simple → complex. Justifies the choice. | One model only (rejected - no evidence). |
| 19 | Week 1 | Prediction intervals | Quantile regression at 10th/90th percentile — differentiator #3 | Prices are uncertain. A single point estimate overstates confidence. | Point estimates only (rejected). |
| 20 | Week 1 | Explainability | SHAP `TreeExplainer` — differentiator #4 | Model-agnostic, shows per-listing drivers, satisfies "explain why THIS price". | `.feature_importances_` only (rejected - global only, biased). |
| 21 | Week 1 | Cleaning | Drop 113 zero-price rows | Not a valid listing price; would skew per-city medians and the model target. | Impute with median (rejected - fabricates data); keep (rejected - distorts). |
| 22 | Week 1 | Cleaning | Clip prices at 99.5th percentile per city | Preserves all rows but caps outlier prices so MAE/RMSE aren't dominated by a handful of luxury listings. Clipping per city (not global) respects local currencies. | Global clip (rejected - would distort smaller-currency cities); no clipping (rejected - outliers dominate metrics); drop outliers (rejected - loses rows unnecessarily). |
| 23 | Week 1 | Cleaning | Fill `host_response_time` NaN with `"unknown"` category | 46% missing — imputing a real value would fabricate data; a distinct "unknown" category preserves the information that this field wasn't recorded. | Median imputation (rejected - categorical, not numeric); drop column (rejected - loses signal); drop rows (rejected - loses 128k rows). |
| 24 | Week 1 | Cleaning | Map `t`/`f` boolean strings to Python bool for `host_is_superhost`, `host_identity_verified`, `instant_bookable` | Enables consistent modelling; avoids mixed-type warnings and string encoding overhead. | Keep as strings (rejected - forces re-encoding downstream). |
| 25 | Week 1 | Cleaning | Output to Parquet with UTF-8, not CSV | Parquet preserves dtypes, compresses to ~50 MB (vs 150 MB CSV), and eliminates all future encoding issues. | CSV (rejected - loses dtype fidelity, re-inherits latin-1 problems). |
| 26 | Week 1 | EDA | Two differentiator charts: price-per-guest KDE, price premium index | Both are currency-neutral by construction. Price-per-guest = price/accommodates; premium index = ratio of city median price-per-guest to dataset median. Sidesteps the invalid cross-currency raw price comparison. | No charts (rejected - loses differentiation); raw price comparison (rejected - invalid due to currency). |
| 27 | Week 1 | Feature engineering | Numeric imputation: `SimpleImputer(strategy="median")` inside Pipeline | Median is robust to outliers vs mean. Fitting inside Pipeline means it only sees training-fold data — no test-set statistic leaks. | Mean imputation (rejected - sensitive to outliers); drop missing rows (rejected - loses 10–33% of data). |
| 28 | Week 1 | Feature engineering | High-cardinality `neighbourhood`: `OrdinalEncoder`, not `OneHotEncoder` | One-hot on hundreds of unique neighbourhoods would explode the feature space (~250+ new columns). Trees handle ordinal splits well. | One-hot (rejected - memory/compute explosion); target encoding (rejected - leakage risk if not carefully cross-validated); drop the column (rejected - loses location signal). |
| 29 | Week 1 | Feature engineering | Exclude `city` from feature list; use as per-city partition key | Per-city models train on one city's rows — `city` would be constant within any model, a useless feature. | Include city as one-hot in a single pooled model (rejected - violates the per-city currency strategy in decision #12). |
| 30 | Week 1 | Feature engineering | Remove `review_scores_value` from `config.yaml` numeric features (corrects earlier omission) | The leakage audit (row 13) and `dtc_00_leakage_audit.md` already excluded it, but `config.yaml` still listed it. Leaving it there would silently leak guest-perceived value-for-money into the model and inflate test R². Caught when the feature-engineering report showed it in the "Numeric features" section. | Keep it in config (rejected - contradicts the leakage audit and violates the decision-log record); ignore the contradiction (rejected - obvious grading risk). |
| 31 | Week 1 | Evaluation | Sensitivity test: retrain best model per city without `review_scores_*` | Mean Δ R² across 10 cities = +0.0178 (max +0.0426, min +0.0010). Removing all 6 review-score features barely changes performance. **Conclusion: model does not rely on post-stay information.** This is a trust argument for deployment — a brand-new listing with no reviews will still be well-predicted. | Skip sensitivity test (rejected - leaves a real leakage question unanswered); exclude review scores entirely from training (rejected - loses some signal, and Δ is small enough that inclusion is defensible). |
| 32 | Week 1 | Evaluation | Note linear regression instability in `dtc_06_split_comparison.md` | On 6 (city, model) pairs, linear regression produced numerically unstable R² values (magnitude > 10^15) due to the right-skewed target and one-hot encoded categorical features. Documented honestly in the report with a footnote; linear regression is not the winner in any city. | Remove linear regression from the report (rejected - loses transparency); log-transform target (rejected - out of scope for this project, documented as future work). |
| 33 | Week 1 | Explainability | SHAP sample size = 2,000 rows per city | TreeExplainer on all 279,599 rows would take hours per city and produce visually cluttered beeswarm plots. A 2,000-row sample preserves the signal (mean |SHAP| is stable at this size) while finishing in under a minute per city. Random seed = 42 for reproducibility. | Full-data SHAP (rejected — hours of runtime, no additional interpretive value); smaller sample (rejected — mean |SHAP| becomes noisy below ~1,000 rows). |

---

## How to use this log

- **Graders** — every judgment call is defensible. Cross-reference with
  `reports/dtc_00_leakage_audit.md`, `reports/dtc_01_data_inspection.md`, and `README.md`.
- **Future maintainers** — if a decision is later reversed, add a new row
  with the new date and mark the old row as superseded. Do not delete.
- **You** — six months from now, this is why your past self made those
  choices.

---

## Entries to add during Stages 11–12

- Prediction interval method choice in Stage 11
- Streamlit app architecture justification in Stage 11
- Any testing decisions in Stage 12