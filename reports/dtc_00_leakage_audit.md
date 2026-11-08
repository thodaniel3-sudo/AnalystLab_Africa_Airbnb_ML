# Leakage Audit - Airbnb Listing Price Prediction

**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1
**Track:** Track C - Machine Learning
**Reference data:** `reports/data_inspection.md` (auto-generated from raw files)

---

## Purpose

This document records, for **every candidate feature** in `Listings.csv`,
the leakage risk it carries, the decision we made, and the reason. It
exists so that anyone reviewing the model — including the grader — can
see that feature selection was deliberate, not accidental.

"Leakage" means: information that would not be available at the moment
a price would realistically be predicted, or that encodes the target
variable in disguise. Including leaked features inflates test scores
during cross-validation but makes the model useless in production.

---

## 1. The target variable

`price` is an `int64` column with 279,712 values, **no missing values**,
min 0, max 625,216, median 150. **113 rows have price = 0** — a Stage 5
concern (likely bad data). It is the target and **must never** appear
as a feature.

**Currency note:** the dataset contains listings from 10 cities
(Paris, New York, Sydney, Rome, Rio de Janeiro, Istanbul, Mexico City,
Bangkok, Cape Town, Hong Kong). Each city's `price` is in its local
currency; the dataset has **no currency column**. See
`reports/decision_log.md` for the strategy (per-city models).

---

## 2. Complete column-by-column audit

Every column from the 33-column `Listings.csv` appears below. The "Decision"
column states whether it enters the model.

| # | Column | Dtype | Leakage risk | Decision | Reason |
|---|---|---|---|---|---|
| 1 | `listing_id` | int64 | **CRITICAL** | Excluded (identifier) | Not a feature - a key. Would let the model memorize listings. |
| 2 | `name` | object | LOW-MEDIUM | Excluded (out of scope) | Free text. Any signal would require NLP; excluded for scope and interpretability. |
| 3 | `host_id` | int64 | **CRITICAL (grouping)** | Excluded from features; used as **group key** in split | Used only in `GroupShuffleSplit(group=host_id)` to prevent host-level leakage. Never enters as a numeric feature. |
| 4 | `host_since` | object (date) | LOW | Excluded (Stage 5-6 candidate) | Date. Could be engineered into "host_tenure_days" in Stage 7; deferred. |
| 5 | `host_location` | object | LOW | Excluded | Free text, high cardinality (200+ unique strings). |
| 6 | `host_response_time` | object | LOW | **Included (categorical)** | Safe - describes host behaviour. 46.04% missing; Stage 5 adds "Unknown" category. |
| 7 | `host_response_rate` | float64 | LOW | Excluded (Stage 5 candidate) | 46.04% missing. Missingness too high to impute reliably. |
| 8 | `host_acceptance_rate` | float64 | LOW | Excluded (Stage 5 candidate) | 40.43% missing. Same reason. |
| 9 | `host_is_superhost` | object | LOW | **Included (categorical)** | Safe - host attribute. |
| 10 | `host_total_listings_count` | float64 | LOW-MEDIUM | **Included (numeric, cautious)** | Strong predictor of professionalism. Monitor for interaction with host grouping. |
| 11 | `host_has_profile_pic` | object | LOW (but low signal) | Excluded | Binary with near-zero variance. |
| 12 | `host_identity_verified` | object | LOW | **Included (categorical)** | Safe - host attribute. |
| 13 | `neighbourhood` | object | LOW | **Included (categorical)** | Location feature, safe. Zero missing. |
| 14 | `district` | object | LOW | Excluded | **86.77% missing (242,700 rows).** Unusable without imputation. |
| 15 | `city` | object | LOW | **Included (categorical AND grouping)** | Also used as the model-partition key in per-city modeling. |
| 16 | `latitude` | float64 | MEDIUM | Excluded | High-cardinality numeric. Strong proxy for `city` and `neighbourhood` (already features). |
| 17 | `longitude` | float64 | MEDIUM | Excluded | Same reason. |
| 18 | `property_type` | object | LOW | **Included (categorical)** | Descriptive, safe. |
| 19 | `room_type` | object | LOW | **Included (categorical)** | Descriptive, safe. |
| 20 | `accommodates` | int64 | LOW | **Included (numeric)** | Capacity - central pricing driver. |
| 21 | `bedrooms` | float64 | LOW | **Included (numeric)** | 10.52% missing (29,435 rows). Imputation planned in Stage 5. |
| 22 | `amenities` | object (JSON-ish) | MEDIUM | Excluded (Stage 7 stretch) | Per-row list of amenities. Could be engineered. Deferred - scope. |
| 23 | `price` | int64 | **TARGET** | Excluded | The target. |
| 24 | `minimum_nights` | int64 | LOW | **Included (numeric)** | Pricing-relevant constraint. |
| 25 | `maximum_nights` | int64 | LOW | **Included (numeric)** | Same. |
| 26 | `review_scores_rating` | float64 | MEDIUM | **Included (numeric, with sensitivity test)** | 32.68% missing. Aggregate review score. Soft leakage risk - do not exist before reviews exist. Stage 8 reports with/without. |
| 27 | `review_scores_accuracy` | float64 | MEDIUM | **Included (numeric, same sensitivity test)** | Same rationale. |
| 28 | `review_scores_cleanliness` | float64 | MEDIUM | **Included (numeric, same)** | Same. |
| 29 | `review_scores_checkin` | float64 | MEDIUM | **Included (numeric, same)** | Same. |
| 30 | `review_scores_communication` | float64 | MEDIUM | **Included (numeric, same)** | Same. |
| 31 | `review_scores_location` | float64 | MEDIUM | **Included (numeric, same)** | Same. |
| 32 | `review_scores_value` | float64 | **HIGH** | **Excluded** | Encodes guest-perceived value-for-money, which is price-dependent. Would unfairly inflate test R². |
| 33 | `instant_bookable` | object | LOW | **Included (categorical)** | Booking-policy flag. Safe. |

---

## 3. Summary

| Category | Count |
|---|---|
| **Target** | 1 (`price`) |
| **Excluded - identifier** | 4 (`listing_id`, `host_id` as feature, `name`, `host_location`) |
| **Excluded - too sparse** | 3 (`district`, `host_response_rate`, `host_acceptance_rate`) |
| **Excluded - geographic proxy** | 2 (`latitude`, `longitude`) |
| **Excluded - leakage** | 1 (`review_scores_value`) |
| **Excluded - low signal** | 1 (`host_has_profile_pic`) |
| **Excluded - deferred** | 2 (`host_since`, `amenities`) |
| **Included - numeric** | 12 |
| **Included - categorical** | 8 |

Every "included" count reconciles to the `features:` block in
`config.yaml`, which is guarded by the automated test
`test_configured_features_exist_in_dataset` in `tests/test_smoke.py`.

---

## 4. The host-grouping leakage risk (evidence)

From `reports/data_inspection.md`, Section 7:

- **182,024 distinct hosts** across 279,712 listings (1.54 listings/host average).
- **31,859 hosts (17.50%)** have more than one listing.
- **One single host has 627 listings.**

**Implication:** A random train/test split would put listings by the
same host into both training and test. A tree model can learn host-specific
patterns (e.g., "this host always prices at ~150") that do not generalise
to unseen hosts. The test score would be inflated relative to true
out-of-sample performance.

**Mitigation:** Stage 8 uses `GroupShuffleSplit` grouped by `host_id` and
reports **both** the random-split and grouped-split scores side by side.
The gap between them quantifies the leakage, and the grouped-split score
is the one we trust.

---

## 5. The duplicate review_id pattern (evidence)

From `reports/reviews_inspection.md` (automatically generated from
`Reviews.csv`):

- **160 duplicated `review_id` values** exist (matches the assignment brief).
- **320 total rows involved** (each duplicate appears exactly twice).
- **0 duplicate IDs appear within the same listing.**
- **160 duplicate IDs appear across different listings.**

**Implication:** The 160 repeated `review_id` values are not guest
re-reviews of the same property. They are cases where the **same
review_id value is attached to two different listings** — an artifact
of the data export rather than a genuine review duplication.

**Decision:** Do **not** delete any review rows. The composite key
`(listing_id, review_id)` is unique across all 5,373,143 rows (validated
in `data_inspection.md`, Section 8). For any per-review aggregation,
we treat the composite key as the true identifier. For per-listing
review counts, no change is needed — the `listing_id` side of the pair
is what's counted.

This decision is documented in `reports/decision_log.md`.

---

## 6. The review-score leakage risk (nuance)

Including `review_scores_*` features is a **debate**. Two positions:

1. **Include them:** they are stable, available, and highly predictive.
   In a real pricing model built *for hosts who already have listings*,
   their review scores are known. This is legitimate.
2. **Exclude them:** in real life, you would predict price *before* a
   listing has accumulated reviews. Including them creates a soft time-travel
   leak.

**Our decision:** include six of the seven review scores
(`review_scores_rating`, `_accuracy`, `_cleanliness`, `_checkin`,
`_communication`, `_location`) but **exclude `review_scores_value`**
because it encodes guest-perceived value for money — a direct proxy
for price.

**Sensitivity test in Stage 8:** report model performance with and
without the six review-score features. Both numbers go in the README.

---

## 7. Guardrail

The test `test_configured_features_exist_in_dataset` in
`tests/test_smoke.py` asserts that every feature name in `config.yaml`
exists as a column in `Listings.csv`. If a name is mistyped, or if a
column is ever removed from the dataset, `pytest` fails before the
model is trained. This ensures the leakage audit and the config never
drift apart silently.