# Cleaning Report

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1
**Track:** Track C - Machine Learning

## 1. Input
- Raw rows: **279,712**
- Raw columns: **33**
- Encoding used: **latin-1**

## 2. Price cleaning
- Price dtype in raw file: **int64**
- Non-numeric prices: **0**
- Zero-price rows: **113**
- Negative-price rows: **0**
- Dropped for price ≤ 0 or non-numeric: **113**
- Rows after price cleaning: **279,599**

## 3. Extreme price clipping
- Strategy: clip `price_clean` at the **99.5th percentile within each city**.
  Motivation: per-city currency means we cannot pool prices; extremely
  high values in any one city would otherwise dominate MAE and RMSE in
  that city's model. Clipping at 99.5% removes the top 0.5% of prices
  per city, which is < 350 rows total in most cities. Those rows remain
  in the dataset — only the price value is capped.
  Rationale for 99.5 not 99: 99% would clip ~2,800 listings across the
  whole dataset. 99.5% balances outlier control against data loss.

| City | 99.5th pct cap |
|---|---|
| Bangkok | 31,755 |
| Cape Town | 30,451 |
| Hong Kong | 10,714 |
| Istanbul | 6,860 |
| Mexico City | 10,841 |
| New York | 1,200 |
| Paris | 856 |
| Rio de Janeiro | 10,000 |
| Rome | 999 |
| Sydney | 2,300 |

## 4. Categorical and boolean normalisation
- `host_response_time`: filled **128,701** missing values with `'unknown'`.
- `host_is_superhost`: mapped 't'/'f' → bool; **165** unmapped (NaN).
- `host_identity_verified`: mapped 't'/'f' → bool; **165** unmapped (NaN).
- `instant_bookable`: mapped 't'/'f' → bool; **0** unmapped (NaN).

## 5. Output column selection
- Kept columns (23): `['price', 'price_clean', 'host_id', 'accommodates', 'bedrooms', 'minimum_nights', 'maximum_nights', 'host_total_listings_count', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'room_type', 'property_type', 'city', 'neighbourhood', 'host_response_time', 'host_is_superhost', 'host_identity_verified', 'instant_bookable']`
- Dropped columns: 11

## 6. Output
- Output file: `data\processed\listings_clean.parquet`
- Final rows: **279,599**
- Final columns: **23**
- Target column present: `price`
- Model target: `price_clean`

## 7. Post-write validation
- ✅ Parquet round-trip: **279,599** rows
- ✅ `price_clean` has no NaN and no non-positive values
- ✅ Row count reconciliation: 279,712 → 279,599 → 279,599