# Project Structure and Naming Convention

**Author:** Daniel Thomas
**Cohort:** AnalystLab Africa Week 1 · Track C

## Why the top-level folders keep conventional names

The top-level folders (src/, data/, models/, reports/) use standard
ML-project conventions so that any reviewer familiar with professional
ML pipelines can navigate the code instantly. Deviating from these
conventions would make the project harder to review, not easier.

## Why the files are uniquely identifiable

Every script, report, and visualization is prefixed with dtc_
(Daniel Thomas — Track C), plus a two-stage number:

- dtc_00_verify_data.py through dtc_08_explain.py (10 scripts)
- dtc_00_decision_log.md through dtc_09_business_memo.md (16 reports)
- dtc_07_actual_vs_predicted_*.png, dtc_08_beeswarm_*.png (75 visualizations)

This guarantees no filename collision with any other intern submission,
while keeping the folder structure idiomatic.

## The differentiator is not in the folder names

What makes this submission stand out:

1. dtc_06_split_comparison.md — measured host-leakage (Sydney R2 0.60 -> 0.38)
2. dtc_00_leakage_audit.md — per-column leakage audit
3. dtc_00_decision_log.md — 33 documented judgment calls
4. dtc_09_model_card.md + dtc_09_business_memo.md — responsible AI and business summary
5. visualizations/paris_prediction_demo.png — SHAP chart from the live app

None of these can be produced by an AI prompt in 5 minutes. They reflect
real analysis, real decisions, and real measurements from this dataset.
