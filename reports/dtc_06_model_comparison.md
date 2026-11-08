# Model Comparison Report

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas

Per-city, per-split, per-model test-set metrics.

## Grouped split (differentiator #1 — HONEST split)

Training and testing sets are disjoint at the **host level** — no host appears in both. This is the score we trust.

| City | Model | n_train | n_test | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| Bangkok | random_forest | 15,375 | 3,985 | 1022.16 | 2455.78 | 0.098 |
| Bangkok | dummy | 15,375 | 3,985 | 1043.48 | 2663.52 | -0.061 |
| Bangkok | hist_gradient_boosting | 15,375 | 3,985 | 1045.94 | 2363.45 | 0.165 |
| Bangkok | linear_regression | 15,375 | 3,985 | 1121.58 | 2372.50 | 0.158 |
| Cape Town | hist_gradient_boosting | 15,284 | 3,802 | 959.32 | 2195.70 | 0.508 |
| Cape Town | random_forest | 15,284 | 3,802 | 998.00 | 2246.10 | 0.485 |
| Cape Town | linear_regression | 15,284 | 3,802 | 1200.39 | 2491.38 | 0.367 |
| Cape Town | dummy | 15,284 | 3,802 | 1351.14 | 3263.56 | -0.087 |
| Hong Kong | random_forest | 5,490 | 1,597 | 451.72 | 1255.36 | 0.118 |
| Hong Kong | dummy | 5,490 | 1,597 | 465.50 | 1368.43 | -0.047 |
| Hong Kong | hist_gradient_boosting | 5,490 | 1,597 | 474.22 | 1224.63 | 0.161 |
| Hong Kong | linear_regression | 5,490 | 1,597 | 508.97 | 1267.58 | 0.101 |
| Istanbul | hist_gradient_boosting | 19,685 | 4,832 | 251.97 | 556.66 | 0.411 |
| Istanbul | random_forest | 19,685 | 4,832 | 257.22 | 565.74 | 0.392 |
| Istanbul | dummy | 19,685 | 4,832 | 286.72 | 748.70 | -0.065 |
| Istanbul | linear_regression | 19,685 | 4,832 | 88895964.85 | 3089684661.82 | -18138936307590.719 |
| Mexico City | hist_gradient_boosting | 16,137 | 3,921 | 496.77 | 1018.17 | 0.293 |
| Mexico City | random_forest | 16,137 | 3,921 | 510.59 | 1030.60 | 0.276 |
| Mexico City | dummy | 16,137 | 3,921 | 586.16 | 1253.92 | -0.072 |
| Mexico City | linear_regression | 16,137 | 3,921 | 59053263095.44 | 1740266151273.70 | -2065749224906851840.000 |
| New York | hist_gradient_boosting | 29,747 | 7,237 | 54.98 | 109.13 | 0.410 |
| New York | random_forest | 29,747 | 7,237 | 56.25 | 111.83 | 0.381 |
| New York | linear_regression | 29,747 | 7,237 | 62.81 | 129.94 | 0.164 |
| New York | dummy | 29,747 | 7,237 | 75.67 | 148.06 | -0.085 |
| Paris | hist_gradient_boosting | 51,748 | 12,880 | 38.47 | 74.46 | 0.470 |
| Paris | random_forest | 51,748 | 12,880 | 39.33 | 75.47 | 0.456 |
| Paris | dummy | 51,748 | 12,880 | 52.93 | 106.45 | -0.083 |
| Paris | linear_regression | 51,748 | 12,880 | 149003386.21 | 9444798441.33 | -8525252236580449.000 |
| Rio de Janeiro | hist_gradient_boosting | 21,354 | 5,254 | 378.30 | 991.46 | 0.258 |
| Rio de Janeiro | random_forest | 21,354 | 5,254 | 394.13 | 1002.41 | 0.241 |
| Rio de Janeiro | dummy | 21,354 | 5,254 | 448.67 | 1196.83 | -0.082 |
| Rio de Janeiro | linear_regression | 21,354 | 5,254 | 682662292.63 | 25712759448.41 | -499228117545467.688 |
| Rome | hist_gradient_boosting | 21,924 | 5,717 | 42.92 | 91.09 | 0.211 |
| Rome | random_forest | 21,924 | 5,717 | 44.01 | 92.05 | 0.194 |
| Rome | dummy | 21,924 | 5,717 | 48.16 | 105.60 | -0.060 |
| Rome | linear_regression | 21,924 | 5,717 | 166000367.70 | 7246568505.95 | -4992806943171249.000 |
| Sydney | hist_gradient_boosting | 26,788 | 6,842 | 101.44 | 250.64 | 0.380 |
| Sydney | random_forest | 26,788 | 6,842 | 104.57 | 254.08 | 0.363 |
| Sydney | dummy | 26,788 | 6,842 | 149.80 | 335.57 | -0.111 |
| Sydney | linear_regression | 26,788 | 6,842 | 70319405.00 | 3098846203.48 | -94766193810403.594 |

## Random split (comparison — leaks host information)

Training and testing sets are split randomly, so listings from the same host land in both. Test scores are **inflated** relative to true out-of-sample performance.

| City | Model | n_train | n_test | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| Bangkok | random_forest | 15,488 | 3,872 | 1020.80 | 2653.88 | 0.284 |
| Bangkok | hist_gradient_boosting | 15,488 | 3,872 | 1069.29 | 2682.47 | 0.269 |
| Bangkok | dummy | 15,488 | 3,872 | 1196.23 | 3236.32 | -0.064 |
| Bangkok | linear_regression | 15,488 | 3,872 | 1219.67 | 2905.79 | 0.142 |
| Cape Town | hist_gradient_boosting | 15,268 | 3,818 | 969.84 | 2144.42 | 0.641 |
| Cape Town | random_forest | 15,268 | 3,818 | 980.35 | 2223.81 | 0.614 |
| Cape Town | linear_regression | 15,268 | 3,818 | 1306.90 | 2757.68 | 0.406 |
| Cape Town | dummy | 15,268 | 3,818 | 1548.23 | 3745.70 | -0.095 |
| Hong Kong | random_forest | 5,669 | 1,418 | 388.81 | 1049.63 | 0.269 |
| Hong Kong | hist_gradient_boosting | 5,669 | 1,418 | 412.30 | 1011.93 | 0.320 |
| Hong Kong | dummy | 5,669 | 1,418 | 448.93 | 1265.54 | -0.063 |
| Hong Kong | linear_regression | 5,669 | 1,418 | 462.84 | 1158.02 | 0.110 |
| Istanbul | random_forest | 19,613 | 4,904 | 241.94 | 585.49 | 0.271 |
| Istanbul | hist_gradient_boosting | 19,613 | 4,904 | 242.26 | 581.21 | 0.281 |
| Istanbul | dummy | 19,613 | 4,904 | 274.03 | 707.36 | -0.065 |
| Istanbul | linear_regression | 19,613 | 4,904 | 97667039.18 | 3969682072.18 | -33533177930028.910 |
| Mexico City | hist_gradient_boosting | 16,046 | 4,012 | 491.05 | 1057.62 | 0.337 |
| Mexico City | random_forest | 16,046 | 4,012 | 501.11 | 1078.73 | 0.310 |
| Mexico City | dummy | 16,046 | 4,012 | 618.92 | 1344.50 | -0.072 |
| Mexico City | linear_regression | 16,046 | 4,012 | 535061865.13 | 28148152039.50 | -469780892536430.125 |
| New York | hist_gradient_boosting | 29,587 | 7,397 | 50.91 | 101.38 | 0.477 |
| New York | random_forest | 29,587 | 7,397 | 52.38 | 104.67 | 0.442 |
| New York | linear_regression | 29,587 | 7,397 | 59.70 | 112.77 | 0.352 |
| New York | dummy | 29,587 | 7,397 | 73.72 | 144.77 | -0.067 |
| Paris | hist_gradient_boosting | 51,702 | 12,926 | 36.26 | 69.70 | 0.503 |
| Paris | random_forest | 51,702 | 12,926 | 37.30 | 70.52 | 0.491 |
| Paris | dummy | 51,702 | 12,926 | 50.77 | 102.64 | -0.078 |
| Paris | linear_regression | 51,702 | 12,926 | 88743688.76 | 4531989653.44 | -2101602611013456.250 |
| Rio de Janeiro | hist_gradient_boosting | 21,286 | 5,322 | 364.26 | 905.74 | 0.387 |
| Rio de Janeiro | random_forest | 21,286 | 5,322 | 368.05 | 908.92 | 0.382 |
| Rio de Janeiro | linear_regression | 21,286 | 5,322 | 436.37 | 1014.48 | 0.231 |
| Rio de Janeiro | dummy | 21,286 | 5,322 | 456.00 | 1204.73 | -0.085 |
| Rome | hist_gradient_boosting | 22,112 | 5,529 | 42.26 | 92.26 | 0.293 |
| Rome | random_forest | 22,112 | 5,529 | 42.80 | 93.39 | 0.276 |
| Rome | dummy | 22,112 | 5,529 | 49.16 | 113.37 | -0.067 |
| Rome | linear_regression | 22,112 | 5,529 | 2079138418.40 | 58432930672.56 | -283496418907956064.000 |
| Sydney | hist_gradient_boosting | 26,904 | 6,726 | 84.60 | 182.87 | 0.599 |
| Sydney | random_forest | 26,904 | 6,726 | 88.34 | 187.06 | 0.580 |
| Sydney | dummy | 26,904 | 6,726 | 136.69 | 302.54 | -0.098 |
| Sydney | linear_regression | 26,904 | 6,726 | 720306924.70 | 21222363123.79 | -5405158801408818.000 |