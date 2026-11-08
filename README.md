# Airbnb Listing Price Prediction

**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1
**Track:** Track C — Machine Learning

An end-to-end ML pipeline predicting Airbnb prices across 10 cities, with per-city models, grouped-split evaluation, SHAP explainability, and a Streamlit app.

---

## 1. Why this project exists

Airbnb prices are in local currencies. Pooling prices across cities is invalid. We train one model per city.

---

## 2. Key results

| City | Model | MAE | R2 |
|---|---|---|---|
| Paris | HistGradientBoosting | 38.47 EUR | 0.470 |
| Cape Town | HistGradientBoosting | 959.32 ZAR | 0.508 |
| Istanbul | HistGradientBoosting | 251.97 TRY | 0.411 |
| New York | HistGradientBoosting | 54.98 USD | 0.410 |
| Sydney | HistGradientBoosting | 101.44 AUD | 0.380 |
| Mexico City | HistGradientBoosting | 496.77 MXN | 0.293 |
| Rio de Janeiro | HistGradientBoosting | 378.30 BRL | 0.258 |
| Rome | HistGradientBoosting | 42.92 EUR | 0.211 |
| Hong Kong | RandomForest | 451.72 HKD | 0.118 |
| Bangkok | RandomForest | 1022.16 THB | 0.098 |

Host leakage: random split inflates R2 by up to 22 percentage points (Sydney 0.38 -> 0.60). We use grouped split.

Sensitivity test: removing review scores barely changes R2 (mean delta +0.018).

---

## 3. Project structure

Every script and report is prefixed dtc_ (Daniel Thomas - Track C).

---

## 4. Reproducing this project

### Step 1: Environment

git clone <repo>
cd AnalystLab_Africa_Airbnb_ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

### Step 2: Datasets

Four files in data/raw/: Listings.csv, Reviews.csv, and two dictionaries.

### Step 3: Run pipeline

python src/dtc_01_data_inspection.py
python src/dtc_03_data_cleaning.py
python src/dtc_04_eda.py
python src/dtc_05_feature_engineering.py
python src/dtc_06_train_model.py
python src/dtc_06b_finalize.py
python src/dtc_07_evaluate_model.py
python src/dtc_08_explain.py
streamlit run app/dtc_streamlit_app.py

### Step 4: Tests

python -m pytest tests -v
Expected: 11 passed.

---

## 5. Data preparation

Raw 279,712 rows -> 279,599 after dropping 113 zero-price rows. Clipped at 99.5th percentile per city. Parquet output.

---

## 6. Feature selection

Excluded 12 columns (identifiers, free text, geographic proxies, high-missingness, leakage). Included 17 features (11 numeric + 6 categorical). See reports/dtc_00_leakage_audit.md.

---

## 7. What makes this different

1. Grouped split with measured leakage (Sydney R2 0.38 -> 0.60)
2. Per-city models (currency-correct)
3. Prediction intervals (not just point estimates)
4. SHAP per-listing explanations
5. Full documentation (33 decisions, leakage audit, model card, business memo, config-driven pipeline)

---

## 8. Limitations

1. Predictions are advertised prices, not revenue
2. Historical data ends 2021
3. Bangkok and Hong Kong have weak signal (R2 ~ 0.1)
4. Linear regression was numerically unstable (-10^15 range)
5. Review scores are post-stay info
6. SHAP shows correlation, not causation
7. Streamlit SHAP rendering: in some environments st.image silently fails to render inline. Chart saved to visualizations/_last_shap_prediction.png.

---

## 9. Attribution

Dataset: Airbnb Listings and Reviews - AnalystLab Africa
Programme: AnalystLab Africa Experience Lab, Week 1
Track: Track C - Machine Learning
Author: Daniel Thomas

---

## 10. Quick links

| Resource | Path |
|---|---|
| Decision log | reports/dtc_00_decision_log.md |
| Leakage audit | reports/dtc_00_leakage_audit.md |
| Model comparison | reports/dtc_06_model_comparison.md |
| Split comparison | reports/dtc_06_split_comparison.md |
| Evaluation | reports/dtc_07_evaluation.md |
| SHAP | reports/dtc_08_shap.md |
| App | app/dtc_streamlit_app.py |
