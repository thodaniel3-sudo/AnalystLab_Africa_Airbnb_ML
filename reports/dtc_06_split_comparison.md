# Grouped vs Random Split Comparison

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas

**Differentiator #1 evidence.** For each city and model, compare R² on a random split (leaks host info) to R² on a grouped split (honest). A large positive Δ is the measured host leakage of a naive random split.

## Table: R² gap per (city, model)

| City | Model | R² (random) | R² (grouped) | Δ R² (leakage) |
|---|---|---|---|---|
| Mexico City | linear_regression | *diverged* | *diverged* | *see note* |
| Paris | linear_regression | *diverged* | *diverged* | *see note* |
| Rio de Janeiro | linear_regression | 0.231 | nan | +nan |
| Sydney | hist_gradient_boosting | 0.599 | 0.380 | +0.219 |
| Sydney | random_forest | 0.580 | 0.363 | +0.217 |
| New York | linear_regression | 0.352 | 0.164 | +0.188 |
| Bangkok | random_forest | 0.284 | 0.098 | +0.186 |
| Hong Kong | hist_gradient_boosting | 0.320 | 0.161 | +0.159 |
| Hong Kong | random_forest | 0.269 | 0.118 | +0.150 |
| Rio de Janeiro | random_forest | 0.382 | 0.241 | +0.141 |
| Cape Town | hist_gradient_boosting | 0.641 | 0.508 | +0.133 |
| Rio de Janeiro | hist_gradient_boosting | 0.387 | 0.258 | +0.129 |
| Cape Town | random_forest | 0.614 | 0.485 | +0.129 |
| Bangkok | hist_gradient_boosting | 0.269 | 0.165 | +0.104 |
| Rome | hist_gradient_boosting | 0.293 | 0.211 | +0.082 |
| Rome | random_forest | 0.276 | 0.194 | +0.081 |
| New York | hist_gradient_boosting | 0.477 | 0.410 | +0.066 |
| New York | random_forest | 0.442 | 0.381 | +0.061 |
| Mexico City | hist_gradient_boosting | 0.337 | 0.293 | +0.044 |
| Cape Town | linear_regression | 0.406 | 0.367 | +0.040 |
| Paris | random_forest | 0.491 | 0.456 | +0.036 |
| Mexico City | random_forest | 0.310 | 0.276 | +0.035 |
| Paris | hist_gradient_boosting | 0.503 | 0.470 | +0.033 |
| New York | dummy | -0.067 | -0.085 | +0.018 |
| Sydney | dummy | -0.098 | -0.111 | +0.013 |
| Hong Kong | linear_regression | 0.110 | 0.101 | +0.008 |
| Paris | dummy | -0.078 | -0.083 | +0.005 |
| Mexico City | dummy | -0.072 | -0.072 | +0.001 |
| Istanbul | dummy | -0.065 | -0.065 | +0.000 |
| Rio de Janeiro | dummy | -0.085 | -0.082 | -0.003 |
| Bangkok | dummy | -0.064 | -0.061 | -0.004 |
| Rome | dummy | -0.067 | -0.060 | -0.007 |
| Cape Town | dummy | -0.095 | -0.087 | -0.008 |
| Hong Kong | dummy | -0.063 | -0.047 | -0.016 |
| Bangkok | linear_regression | 0.142 | 0.158 | -0.017 |
| Istanbul | random_forest | 0.271 | 0.392 | -0.121 |
| Istanbul | hist_gradient_boosting | 0.281 | 0.411 | -0.130 |
| Istanbul | linear_regression | *diverged* | *diverged* | *see note* |
| Sydney | linear_regression | *diverged* | *diverged* | *see note* |
| Rome | linear_regression | *diverged* | *diverged* | *see note* |

## Interpretation

- Positive Δ means the random split **overstates** model quality — the model learns host-specific patterns that do not generalise.
- A large Δ is *expected*: 17.50% of hosts have multiple listings (largest single host: 627 listings). Random splits mix them across train and test; grouped splits prevent this by construction.
- The **grouped-split R² is the honest benchmark** and is the number used to select winners in `dtc_06_model_comparison.md`.

## Note on linear regression instability

**6 row(s)** show numerically unstable R² values (magnitude > 1) for **linear regression**. This is a known limitation of OLS on heavily right-skewed targets combined with high-cardinality one-hot encoded categorical features. The resulting coefficients become astronomically large and predictions explode at prediction time.

**Implication:** Linear regression is unusable for this dataset as-is. It is **not selected as the winning model in any city**. Its metrics are retained here for transparency but should not be quoted in comparative claims. A log-transformed target would fix the instability, but that is out of scope for this project and is documented as future work.

**Count of diverged linear_regression fits:** 6