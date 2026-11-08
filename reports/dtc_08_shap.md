# SHAP Explainability Report

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas

## Method
## How to read these plots

- **Beeswarm:** Each point is one listing in the sample. X-axis = SHAP value (contribution to price). Colour = feature value (red high, blue low). If red points cluster on the right of a feature, that feature *increases* price when its value is high.
- **Waterfall:** For one specific listing, starts at the model's base value (the average prediction). Each bar shows one feature pushing the prediction up (red) or down (blue) until reaching the final predicted price.

## ⚠️ Correlation is not causation

SHAP values show what the *model learned* from the data, not what causes prices in reality. A feature can show large SHAP values simply because it correlates with another unobserved driver. Do not interpret SHAP as causal evidence. For causal claims, an experiment or natural experiment would be required.


`shap.TreeExplainer` is used on the winning grouped-split model per city. Sample size: **2,000** rows per city (config: `explainability.max_samples_for_shap`). Larger samples produce cluttered visualisations and take hours; 2,000 rows preserves the signal while keeping runtime under a minute per city.

### Bangkok (winner: random_forest)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 585.938 |
| 2 | host_total_listings_count | 187.216 |
| 3 | accommodates | 171.938 |
| 4 | neighbourhood | 135.732 |
| 5 | review_scores_location | 123.418 |
| 6 | property_type_Room in hotel | 102.526 |
| 7 | host_response_time_within an hour | 92.712 |
| 8 | minimum_nights | 78.967 |
| 9 | host_response_time_unknown | 69.167 |
| 10 | review_scores_rating | 58.094 |

**Beeswarm plot:**

![beeswarm Bangkok](../visualizations/dtc_08_beeswarm_bangkok.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_bangkok_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_bangkok_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_bangkok_3.png)

### Cape Town (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | accommodates | 749.876 |
| 2 | bedrooms | 592.992 |
| 3 | neighbourhood | 496.976 |
| 4 | property_type_Entire villa | 275.430 |
| 5 | review_scores_rating | 231.276 |
| 6 | host_total_listings_count | 224.641 |
| 7 | minimum_nights | 174.744 |
| 8 | property_type_Entire house | 77.596 |
| 9 | host_response_time_unknown | 69.402 |
| 10 | host_response_time_within an hour | 68.120 |

**Beeswarm plot:**

![beeswarm Cape Town](../visualizations/dtc_08_beeswarm_cape_town.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_cape_town_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_cape_town_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_cape_town_3.png)

### Hong Kong (winner: random_forest)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 181.764 |
| 2 | host_total_listings_count | 131.420 |
| 3 | host_response_time_unknown | 82.079 |
| 4 | accommodates | 82.051 |
| 5 | room_type_Entire place | 62.911 |
| 6 | neighbourhood | 50.743 |
| 7 | minimum_nights | 48.354 |
| 8 | instant_bookable_True | 22.680 |
| 9 | property_type_Entire apartment | 19.855 |
| 10 | property_type_Private room in serviced apartment | 18.095 |

**Beeswarm plot:**

![beeswarm Hong Kong](../visualizations/dtc_08_beeswarm_hong_kong.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_hong_kong_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_hong_kong_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_hong_kong_3.png)

### Istanbul (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | accommodates | 79.937 |
| 2 | host_response_time_unknown | 63.125 |
| 3 | bedrooms | 49.993 |
| 4 | host_total_listings_count | 39.448 |
| 5 | room_type_Entire place | 30.061 |
| 6 | neighbourhood | 29.848 |
| 7 | property_type_Entire villa | 23.204 |
| 8 | room_type_Private room | 21.114 |
| 9 | review_scores_cleanliness | 20.920 |
| 10 | host_response_time_within an hour | 19.750 |

**Beeswarm plot:**

![beeswarm Istanbul](../visualizations/dtc_08_beeswarm_istanbul.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_istanbul_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_istanbul_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_istanbul_3.png)

### Mexico City (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 171.194 |
| 2 | room_type_Private room | 145.624 |
| 3 | accommodates | 126.261 |
| 4 | neighbourhood | 111.565 |
| 5 | host_total_listings_count | 105.761 |
| 6 | host_response_time_unknown | 98.389 |
| 7 | review_scores_rating | 85.277 |
| 8 | minimum_nights | 43.900 |
| 9 | maximum_nights | 42.190 |
| 10 | host_response_time_within an hour | 40.964 |

**Beeswarm plot:**

![beeswarm Mexico City](../visualizations/dtc_08_beeswarm_mexico_city.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_mexico_city_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_mexico_city_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_mexico_city_3.png)

### New York (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | accommodates | 27.154 |
| 2 | neighbourhood | 19.075 |
| 3 | room_type_Private room | 16.725 |
| 4 | bedrooms | 16.433 |
| 5 | host_response_time_unknown | 9.246 |
| 6 | room_type_Entire place | 8.204 |
| 7 | review_scores_rating | 5.241 |
| 8 | review_scores_location | 4.546 |
| 9 | maximum_nights | 4.494 |
| 10 | host_total_listings_count | 4.322 |

**Beeswarm plot:**

![beeswarm New York](../visualizations/dtc_08_beeswarm_new_york.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_new_york_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_new_york_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_new_york_3.png)

### Paris (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 19.777 |
| 2 | accommodates | 18.579 |
| 3 | neighbourhood | 12.817 |
| 4 | host_total_listings_count | 8.388 |
| 5 | property_type_Room in boutique hotel | 6.127 |
| 6 | review_scores_rating | 5.305 |
| 7 | minimum_nights | 2.582 |
| 8 | property_type_Entire apartment | 2.285 |
| 9 | review_scores_cleanliness | 2.118 |
| 10 | review_scores_location | 1.743 |

**Beeswarm plot:**

![beeswarm Paris](../visualizations/dtc_08_beeswarm_paris.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_paris_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_paris_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_paris_3.png)

### Rio de Janeiro (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 209.220 |
| 2 | host_response_time_unknown | 112.277 |
| 3 | review_scores_rating | 88.817 |
| 4 | accommodates | 76.883 |
| 5 | neighbourhood | 73.551 |
| 6 | host_total_listings_count | 60.558 |
| 7 | minimum_nights | 45.198 |
| 8 | room_type_Entire place | 44.963 |
| 9 | host_identity_verified_False | 17.990 |
| 10 | maximum_nights | 16.942 |

**Beeswarm plot:**

![beeswarm Rio de Janeiro](../visualizations/dtc_08_beeswarm_rio_de_janeiro.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_rio_de_janeiro_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_rio_de_janeiro_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_rio_de_janeiro_3.png)

### Rome (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 15.268 |
| 2 | neighbourhood | 12.744 |
| 3 | accommodates | 10.605 |
| 4 | review_scores_rating | 7.790 |
| 5 | host_total_listings_count | 7.179 |
| 6 | host_response_time_unknown | 6.226 |
| 7 | room_type_Entire place | 6.139 |
| 8 | host_response_time_within an hour | 4.067 |
| 9 | review_scores_location | 4.013 |
| 10 | minimum_nights | 3.256 |

**Beeswarm plot:**

![beeswarm Rome](../visualizations/dtc_08_beeswarm_rome.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_rome_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_rome_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_rome_3.png)

### Sydney (winner: hist_gradient_boosting)

**Global feature importance (mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | bedrooms | 54.376 |
| 2 | accommodates | 51.186 |
| 3 | neighbourhood | 20.056 |
| 4 | room_type_Entire place | 18.793 |
| 5 | host_total_listings_count | 12.273 |
| 6 | review_scores_rating | 11.111 |
| 7 | minimum_nights | 8.618 |
| 8 | property_type_Entire house | 8.252 |
| 9 | host_response_time_unknown | 4.940 |
| 10 | host_response_time_within a day | 4.618 |

**Beeswarm plot:**

![beeswarm Sydney](../visualizations/dtc_08_beeswarm_sydney.png)

**Example listings:**

- Median listing → ![waterfall](../visualizations/dtc_08_waterfall_sydney_1.png)
- Highest listing → ![waterfall](../visualizations/dtc_08_waterfall_sydney_2.png)
- Lowest listing → ![waterfall](../visualizations/dtc_08_waterfall_sydney_3.png)
