# Model Card - Airbnb Listing Price Prediction

**Author:** Daniel Thomas
**Programme:** AnalystLab Africa Experience Lab, Week 1
**Track:** Track C - Machine Learning

---

## 1. Model overview

- Task: Regression (predict advertised Airbnb price)
- Target: price_clean (per-city 99.5th percentile clipped)
- Training data: 279,599 listings across 10 cities, 2008-2021
- Architecture: Per-city pipelines; HistGradientBoosting (8 cities), RandomForest (2 cities)
- Features: 11 numeric + 6 categorical
- Grouping key: host_id (used for GroupShuffleSplit)
- Metrics: MAE, RMSE, R2 on grouped split

## 2. Intended use

Intended:
- Educational demonstration of ML for market pricing
- Comparing relative pricing within a city
- Portfolio piece showing per-city modeling, grouped-split evaluation, SHAP

NOT intended:
- Real rent setting (predictions are advertised price, not revenue)
- Cross-city raw currency comparison
- Live deployment without retraining (data ends 2021)
- Decisions requiring fairness across demographics

## 3. Training data

- Source: Airbnb Listings and Reviews Dataset (AnalystLab Africa)
- 10 cities: Bangkok, Cape Town, Hong Kong, Istanbul, Mexico City, New York, Paris, Rio de Janeiro, Rome, Sydney
- 279,599 listings after cleaning (113 zero-price rows dropped)
- 5,373,143 reviews (used for analysis, not pricing model)

## 4. Fairness and bias

- Geographic bias: Paris 23% vs Hong Kong 2.5% of listings
- Review-score bias: post-stay info, but sensitivity test shows minimal impact
- Currency bias: nominal prices, not purchasing-power adjusted
- Temporal bias: data ends 2021, market changed post-COVID

## 5. Performance

R2 range: 0.098 (Bangkok) to 0.508 (Cape Town). Mean R2 ~ 0.32.

## 6. Ethical recommendations

1. Do not present predictions as market truth
2. Retrain before new deployment
3. Be explicit about currency
4. Do not compare predictions across cities
5. Do not use for legal/financial decisions without human review

## 7. Attribution

Author: Daniel Thomas
Programme: AnalystLab Africa Experience Lab, Week 1
Dataset: Airbnb Listings and Reviews - AnalystLab Africa
