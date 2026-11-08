"""
Stage 11 - Interactive pricing app (Streamlit).

Run with:
    streamlit run app/dtc_streamlit_app.py

Features:
  * City selector (10 cities)
  * Listing form (12 numeric + 6 categorical features)
  * Predict button → point estimate + 10-90% interval
  * SHAP breakdown — why this price?
  * Price fairness check — how the prediction compares to city median

Predictions are for listed advertised price only, not revenue or occupancy.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_").replace("/", "_")


@st.cache_data
def load_data() -> pd.DataFrame:
    cfg = load_config()
    return pd.read_parquet(ROOT / cfg["paths"]["listings_clean"])


@st.cache_resource
def load_model(city_slug_: str):
    cfg = load_config()
    path = ROOT / cfg["paths"]["models_dir"] / f"dtc_{city_slug_}_best_grouped.joblib"
    return joblib.load(path)


# ---- Currency map ----
CURRENCY = {
    "Bangkok": "THB (฿)",
    "Cape Town": "ZAR (R)",
    "Hong Kong": "HKD (HK$)",
    "Istanbul": "TRY (₺)",
    "Mexico City": "MXN (MX$)",
    "New York": "USD ($)",
    "Paris": "EUR (€)",
    "Rio de Janeiro": "BRL (R$)",
    "Rome": "EUR (€)",
    "Sydney": "AUD (A$)",
}


def main() -> None:
    st.set_page_config(page_title="Airbnb Listing Price Predictor", layout="wide")
    st.title("🏠 Airbnb Listing Price Predictor")
    st.caption(
        "AnalystLab Africa Week 1 · Track C · Daniel Thomas · "
        "Per-city ML models with grouped-split honest evaluation"
    )

    df = load_data()

    # ---- Sidebar: city selector ----
    st.sidebar.header("1. Choose a market")
    city = st.sidebar.selectbox("City", sorted(df["city"].dropna().unique()))
    cur = CURRENCY.get(city, "")
    st.sidebar.markdown(f"**Local currency:** `{cur}`")

    df_city = df[df["city"] == city].copy()

    # ---- Sidebar: form ----
    st.sidebar.header("2. Listing details")
    room_type = st.sidebar.selectbox(
        "Room type", sorted(df_city["room_type"].dropna().unique())
    )
    property_type = st.sidebar.selectbox(
        "Property type",
        sorted(df_city["property_type"].dropna().unique()),
    )
    neighbourhood = st.sidebar.selectbox(
        "Neighbourhood",
        sorted(df_city["neighbourhood"].dropna().unique()),
    )
    accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)
    bedrooms = st.sidebar.slider("Bedrooms", 0.0, 10.0, 1.0, step=0.5)
    minimum_nights = st.sidebar.slider("Minimum nights", 1, 365, 2)
    maximum_nights = st.sidebar.slider("Maximum nights", 1, 3650, 365, step=30)
    host_total_listings_count = st.sidebar.slider(
        "Host total listings", 1, 500, 1
    )
    host_response_time = st.sidebar.selectbox(
        "Host response time",
        ["within an hour", "within a few hours", "within a day",
         "a few days or more", "unknown"],
    )
    host_is_superhost = st.sidebar.checkbox("Superhost", value=False)
    host_identity_verified = st.sidebar.checkbox("Identity verified", value=True)
    instant_bookable = st.sidebar.checkbox("Instant bookable", value=False)

    # Review scores: median defaults, allow override
    st.sidebar.header("3. Review scores (optional)")
    use_default_scores = st.sidebar.checkbox("Use city medians", value=True)
    if use_default_scores:
        review_scores_rating = float(df_city["review_scores_rating"].median())
        review_scores_accuracy = float(df_city["review_scores_accuracy"].median())
        review_scores_cleanliness = float(df_city["review_scores_cleanliness"].median())
        review_scores_checkin = float(df_city["review_scores_checkin"].median())
        review_scores_communication = float(df_city["review_scores_communication"].median())
        review_scores_location = float(df_city["review_scores_location"].median())
        st.sidebar.caption(
            f"Using medians: rating={review_scores_rating:.0f}, "
            f"accuracy={review_scores_accuracy:.0f}, etc."
        )
    else:
        review_scores_rating = st.sidebar.slider("Rating", 20, 100, 95)
        review_scores_accuracy = st.sidebar.slider("Accuracy", 2, 10, 10)
        review_scores_cleanliness = st.sidebar.slider("Cleanliness", 2, 10, 10)
        review_scores_checkin = st.sidebar.slider("Checkin", 2, 10, 10)
        review_scores_communication = st.sidebar.slider("Communication", 2, 10, 10)
        review_scores_location = st.sidebar.slider("Location", 2, 10, 10)

    # ---- Build input row ----
    input_row = pd.DataFrame([{
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "minimum_nights": minimum_nights,
        "maximum_nights": maximum_nights,
        "host_total_listings_count": host_total_listings_count,
        "review_scores_rating": review_scores_rating,
        "review_scores_accuracy": review_scores_accuracy,
        "review_scores_cleanliness": review_scores_cleanliness,
        "review_scores_checkin": review_scores_checkin,
        "review_scores_communication": review_scores_communication,
        "review_scores_location": review_scores_location,
        "room_type": room_type,
        "property_type": property_type,
        "neighbourhood": neighbourhood,
        "host_response_time": host_response_time,
        "host_is_superhost": host_is_superhost,
        "host_identity_verified": host_identity_verified,
        "instant_bookable": instant_bookable,
    }])

    # ---- Predict button ----
    st.sidebar.header("4. Predict")
    predict = st.sidebar.button(
        "🔮 Predict price", type="primary", use_container_width=True
    )

    if not predict:
        st.info(
            "👈 Fill in the listing details on the left and click "
            "**Predict price** to see the model's estimate."
        )
        st.subheader(f"Market context — {city}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Listings in city", f"{len(df_city):,}")
        col2.metric("Median price", f"{df_city['price_clean'].median():,.0f}")
        col3.metric(
            "Median per guest",
            f"{(df_city['price_clean'] / df_city['accommodates'].clip(lower=1)).median():,.1f}",
        )
        return

    # ---- Do the prediction ----
    slug = city_slug(city)
    pipe = load_model(slug)

    # Models were trained on raw prices — no log transform. Simple float cast.
    pred = float(pipe.predict(input_row)[0])

    # ---- Compute price interval from test-set MAE ----
    try:
        mc = (ROOT / "reports" / "dtc_06_model_comparison.md").read_text(encoding="utf-8")
        in_grouped = False
        mae_city = None
        winner_model = None
        for line in mc.splitlines():
            if line.startswith("## Grouped split"):
                in_grouped = True
                continue
            if line.startswith("## Random split"):
                in_grouped = False
                continue
            if not in_grouped or not line.startswith("| "):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) != 7 or parts[0] != city:
                continue
            try:
                mae = float(parts[4])
            except ValueError:
                continue
            if mae_city is None or mae < mae_city:
                mae_city = mae
                winner_model = parts[1]
    except Exception:
        mae_city = None
        winner_model = None

    if mae_city is not None:
        lower = max(pred - 1.28 * mae_city, 0)
        upper = pred + 1.28 * mae_city
    else:
        lower, upper = pred * 0.75, pred * 1.25

    # ---- Display main prediction ----
    st.subheader(f"Prediction for {city}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted price", f"{pred:,.0f} {cur.split()[0] if cur else ''}")
    c2.metric("Likely range (10–90%)", f"{lower:,.0f} – {upper:,.0f}")
    c3.metric("Model", winner_model or "unknown")

    st.caption(
        f"Model: **{winner_model}** trained on {city} listings. "
        f"Grouped-split R² reported in `dtc_06_model_comparison.md`. "
        f"MAE ≈ {mae_city:,.1f} {cur.split()[0] if cur else ''}. "
        f"Range is approximate — derived from the test-set MAE."
    )

    # ---- Fairness check ----
    st.subheader("How does this listing compare?")
    similar = df_city[
        (df_city["room_type"] == room_type)
        & (df_city["bedrooms"].round() == round(bedrooms))
    ]
    if len(similar) >= 30:
        median_similar = float(similar["price_clean"].median())
        delta_pct = (pred - median_similar) / median_similar * 100
        if delta_pct > 15:
            verdict = f"🔴 **{delta_pct:+.0f}% above** comparable listings"
            color = "error"
        elif delta_pct < -15:
            verdict = f"🟢 **{delta_pct:+.0f}% below** comparable listings"
            color = "success"
        else:
            verdict = f"🟡 **In line** with comparable listings ({delta_pct:+.0f}%)"
            color = "warning"
        st.markdown(
            f"- Comparable listings (same room type + bedrooms in {city}): "
            f"**n = {len(similar):,}**"
        )
        st.markdown(f"- Their median price: **{median_similar:,.0f}**")
        getattr(st, color)(verdict)
    else:
        st.info(
            f"Not enough comparable listings (n={len(similar)}) for a reliable comparison."
        )

    # ---- SHAP explanation (bar chart — robust rendering) ----
    st.subheader("Why this price? (SHAP breakdown)")
    try:
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
        X_trans = pre.transform(input_row)
        feature_names = list(pre.get_feature_names_out())

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_trans)

        # Normalise to 1-D array of per-feature contributions
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 2:
            shap_row = shap_vals[0, :]
        else:
            shap_row = shap_vals.reshape(-1)
        shap_row = shap_row[:len(feature_names)]

        print(
            f"[SHAP DEBUG] shape={shap_row.shape} "
            f"min={shap_row.min():.4f} max={shap_row.max():.4f} "
            f"abs_sum={np.abs(shap_row).sum():.4f}"
        )

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).flatten()[0])
        else:
            base = float(base)

        # Pick top 12 by absolute value
        n_show = min(12, len(feature_names))
        top_idx = np.argsort(np.abs(shap_row))[::-1][:n_show]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = np.asarray(shap_row)[top_idx]

        print(f"[SHAP DEBUG] top_vals[:3]={top_vals[:3]}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["firebrick" if v > 0 else "steelblue" for v in top_vals]
        y_positions = np.arange(n_show)

        ax.barh(y_positions, top_vals, color=colors, edgecolor="white")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("SHAP value (contribution to predicted price)")
        ax.set_title(
            f"Why this price? Top {n_show} drivers\n"
            f"Base = {base:.1f}  →  Prediction = {base + float(np.sum(shap_row)):.1f}",
            fontsize=11, fontweight="bold",
        )
        ax.invert_yaxis()
        fig.tight_layout()



        # Save to disk and display via st.image (most reliable path)
        import os
        debug_dir = ROOT / "visualizations"
        debug_dir.mkdir(parents=True, exist_ok=True)
        shap_path = debug_dir / "_last_shap_prediction.png"
        fig.savefig(str(shap_path), format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[SHAP DEBUG] saved figure to {shap_path} "
              f"({os.path.getsize(shap_path)} bytes)")
        st.image(str(shap_path), use_container_width=True)

        st.caption(
            "🔴 Red = increases predicted price · "
            "🔵 Blue = decreases predicted price · "
            "Base + sum of bars = final prediction"
        )
    except Exception as e:
        st.error(f"SHAP explanation unavailable: {e}")
        import traceback
        st.code(traceback.format_exc())

    # ---- Footer: caveats ----
    st.divider()
    st.caption(
        "⚠️ **Important limitations**\n"
        "- Predictions are for **advertised listing price** only — "
        "not revenue, occupancy, or profit.\n"
        "- Review scores are optional; if you leave them at city medians, "
        "the model uses that average. Sensitivity test (Stage 9) showed the "
        "model does not rely heavily on review scores (mean Δ R² = +0.018).\n"
        "- The model was trained on historical listings from 2008–2021. "
        "Market conditions change; do not treat predictions as current market price."
    )


if __name__ == "__main__":
    main()