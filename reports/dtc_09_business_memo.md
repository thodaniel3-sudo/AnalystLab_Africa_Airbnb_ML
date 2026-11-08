# Business Memo - Airbnb Price Analysis

**To:** AnalystLab Africa Leadership
**From:** Daniel Thomas (Track C - Machine Learning)
**Date:** Week 1
**Subject:** What we learned from 279,599 Airbnb listings

---

## Executive summary

We analyzed 279,599 Airbnb listings across 10 markets. Our per-city model predicts advertised prices with R2 up to 0.508. Three things matter for pricing; two things don't.

## Three things we learned

### 1. Predictability varies dramatically by city

Paris R2 = 0.470 (MAE 38 EUR). Cape Town R2 = 0.508 (MAE 959 ZAR). But Bangkok R2 = 0.098 and Hong Kong R2 = 0.118 — the model barely beats the median. In Paris and Cape Town, a pricing model is genuinely useful. In Bangkok and Hong Kong, price depends on factors the data doesn't capture.

### 2. A model can help individual hosts

Example: Paris Entire place, 2 guests, 1 bedroom predicted at 142 EUR. Comparable listings median 160 EUR. This specific listing is 12% below market — a specific, actionable number.

### 3. The model doesn't rely on reviews (good!)

We retrained without review scores. Average R2 drop: +0.018. The model works without any reviews — meaning a brand-new listing still gets a well-calibrated prediction.

## Three recommended actions

### 1. Deploy only in strong-signal cities

Paris, Cape Town, Istanbul, New York (R2 > 0.4). Hold Bangkok and Hong Kong.

Evaluation: track MAE against a hidden holdout of recent listings. If Paris MAE stays under 50 EUR for 3 months, expand.

### 2. Add occupancy data to enable revenue analysis

Price * bookings = revenue. Without booking data we can only say "in line with market," not "maximizes revenue."

Evaluation: A/B test with 100 hosts. Half follow model recommendation, half price themselves. Compare revenue over 3 months.

### 3. Communicate the uncertainty

Every prediction has a 10-90% range. A single point estimate invites over-trust. Any deployment should always show the range.

## Three things we cannot tell you

1. What rent you should actually charge (advertised price != revenue)
2. Whether your listing will book (no occupancy data)
3. What's happening right now (data ends March 2021)

## Recommendation

Pilot deployment in Paris and Cape Town with human oversight. Track real-world MAE. Expand only if accuracy holds.
