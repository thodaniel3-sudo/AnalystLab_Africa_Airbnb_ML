# Data Inspection Report

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1
**Track:** Track C - Machine Learning

## 1. Listings.csv - Shape and Schema

- Rows: **279,712**
- Columns: **33**
- File encoding used: **latin-1**

| Column | dtype | Non-null | Missing | Missing % |
|---|---|---|---|---|
| `listing_id` | int64 | 279,712 | 0 | 0.00% |
| `name` | object | 279,537 | 175 | 0.06% |
| `host_id` | int64 | 279,712 | 0 | 0.00% |
| `host_since` | object | 279,547 | 165 | 0.06% |
| `host_location` | object | 278,872 | 840 | 0.30% |
| `host_response_time` | object | 150,930 | 128,782 | 46.04% |
| `host_response_rate` | float64 | 150,930 | 128,782 | 46.04% |
| `host_acceptance_rate` | float64 | 166,625 | 113,087 | 40.43% |
| `host_is_superhost` | object | 279,547 | 165 | 0.06% |
| `host_total_listings_count` | float64 | 279,547 | 165 | 0.06% |
| `host_has_profile_pic` | object | 279,547 | 165 | 0.06% |
| `host_identity_verified` | object | 279,547 | 165 | 0.06% |
| `neighbourhood` | object | 279,712 | 0 | 0.00% |
| `district` | object | 37,012 | 242,700 | 86.77% |
| `city` | object | 279,712 | 0 | 0.00% |
| `latitude` | float64 | 279,712 | 0 | 0.00% |
| `longitude` | float64 | 279,712 | 0 | 0.00% |
| `property_type` | object | 279,712 | 0 | 0.00% |
| `room_type` | object | 279,712 | 0 | 0.00% |
| `accommodates` | int64 | 279,712 | 0 | 0.00% |
| `bedrooms` | float64 | 250,277 | 29,435 | 10.52% |
| `amenities` | object | 279,712 | 0 | 0.00% |
| `price` | int64 | 279,712 | 0 | 0.00% |
| `minimum_nights` | int64 | 279,712 | 0 | 0.00% |
| `maximum_nights` | int64 | 279,712 | 0 | 0.00% |
| `review_scores_rating` | float64 | 188,307 | 91,405 | 32.68% |
| `review_scores_accuracy` | float64 | 187,999 | 91,713 | 32.79% |
| `review_scores_cleanliness` | float64 | 188,047 | 91,665 | 32.77% |
| `review_scores_checkin` | float64 | 187,941 | 91,771 | 32.81% |
| `review_scores_communication` | float64 | 188,025 | 91,687 | 32.78% |
| `review_scores_location` | float64 | 187,937 | 91,775 | 32.81% |
| `review_scores_value` | float64 | 187,927 | 91,785 | 32.81% |
| `instant_bookable` | object | 279,712 | 0 | 0.00% |

## 2. Primary key - listing_id

- Distinct listing_id values: **279,712**
- Duplicate listing rows: **0**
- ✅ `listing_id` is a valid primary key (fully unique).

## 3. City distribution

- Distinct cities: **10**

| City | Listings | Share |
|---|---|---|
| Paris | 64,690 | 23.13% |
| New York | 37,012 | 13.23% |
| Sydney | 33,630 | 12.02% |
| Rome | 27,647 | 9.88% |
| Rio de Janeiro | 26,615 | 9.52% |
| Istanbul | 24,519 | 8.77% |
| Mexico City | 20,065 | 7.17% |
| Bangkok | 19,361 | 6.92% |
| Cape Town | 19,086 | 6.82% |
| Hong Kong | 7,087 | 2.53% |

## 4. Room type distribution


| Room type | Listings | Share |
|---|---|---|
| Entire place | 182,005 | 65.07% |
| Private room | 86,988 | 31.10% |
| Hotel room | 5,857 | 2.09% |
| Shared room | 4,862 | 1.74% |

## 5. Property type distribution (top 15)


| Property type | Listings | Share |
|---|---|---|
| Entire apartment | 138,989 | 49.69% |
| Private room in apartment | 47,322 | 16.92% |
| Private room in house | 13,292 | 4.75% |
| Entire house | 13,273 | 4.75% |
| Entire condominium | 11,250 | 4.02% |
| Room in boutique hotel | 5,771 | 2.06% |
| Entire loft | 4,587 | 1.64% |
| Private room in condominium | 4,462 | 1.60% |
| Private room in bed and breakfast | 4,238 | 1.52% |
| Entire serviced apartment | 3,973 | 1.42% |
| Room in hotel | 3,205 | 1.15% |
| Private room in townhouse | 2,959 | 1.06% |
| Shared room in apartment | 2,420 | 0.87% |
| Entire townhouse | 2,331 | 0.83% |
| Entire guest suite | 2,273 | 0.81% |

## 6. Price column - format and range

- dtype: `int64`
- Sample values: `['53', '120', '89', '58', '60', '95', '80', '59', '80', '90', '60', '80', '61', '145', '35', '39', '55', '55', '57', '55']`
- Non-numeric / parse-failed values: **0**
- Numeric parse succeeded on **279,712** rows
- Numeric min: **0**
- Numeric max: **625216**
- Numeric median: **150.0**
- Zero-price rows: **113**
- Negative-price rows: **0**

## 7. Host grouping - differentiator #1 evidence

- Distinct hosts: **182,024**
- Listings on average per host: **1.54**
- Hosts with more than 1 listing: **31,859** (17.50% of hosts)
- Largest single-host listing count: **627**
- **Implication:** A random train/test split will leak host information — listings by the same host will appear in both train and test, inflating the test score. **Stage 8 will use `GroupShuffleSplit` by `host_id`** and compare against a random split.

## 8. Reviews.csv - structure and integrity

- Total reviews: **5,373,143**
- Distinct review_id values: **5,372,983**
- Duplicate review_id rows: **160**
- Distinct (listing_id, review_id) pairs: **5,373,143**
- ✅ `(listing_id, review_id)` is unique across all rows.
- Distinct listing_ids with at least one review: **193,556**
- Total listings in Listings.csv: **279,712**
- Listings with zero reviews: **86,156**
- Orphan review listing_ids (not in Listings): **0**
- ✅ Foreign-key relationship `Reviews.listing_id → Listings.listing_id` is fully valid.
- Review date range: **2008-11-16** to **2021-03-01**

**Reviews per listing (over listings that have any reviews):**
- Min: **1**
- Max: **891**
- Mean: **27.76**
- Median: **8.0**

## 9. Summary of findings

- All raw files are present, readable, and match expected row counts.
- `listing_id` uniqueness validated.
- `(listing_id, review_id)` uniqueness validated.
- Foreign-key relationship validated.
- Price column format inspected; conversion strategy applied in Stage 5.
- Host grouping quantified — drives grouped-split decision in Stage 8.
- All findings carried forward to `reports/leakage_audit.md` and `reports/decision_log.md`.