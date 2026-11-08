"""Pipeline integration tests - verify trained model, preprocessor, and app essentials."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_cleaned_dataset_exists():
    p = ROOT / CFG["paths"]["listings_clean"]
    assert p.exists(), f"missing: {p}"


def test_cleaned_dataset_has_no_null_target():
    df = pd.read_parquet(ROOT / CFG["paths"]["listings_clean"])
    assert df["price_clean"].notna().all()
    assert (df["price_clean"] > 0).all()


def test_preprocessor_exists():
    p = ROOT / CFG["paths"]["models_dir"] / "dtc_preprocessor.joblib"
    assert p.exists(), f"missing: {p}"
    joblib.load(p)  # raise if corrupt


def test_all_cities_have_a_best_model():
    models_dir = ROOT / CFG["paths"]["models_dir"]
    best = list(models_dir.glob("dtc_*_best_grouped.joblib"))
    assert len(best) >= 10, f"Only {len(best)} best models found - expected 10"


def test_sample_paris_prediction_is_plausible():
    models_dir = ROOT / CFG["paths"]["models_dir"]
    model_path = models_dir / "dtc_paris_best_grouped.joblib"
    if not model_path.exists():
        pytest.skip("Paris model not trained")

    pipe = joblib.load(model_path)
    X = pd.DataFrame([{
        "accommodates": 2, "bedrooms": 1.0, "minimum_nights": 2,
        "maximum_nights": 365, "host_total_listings_count": 1,
        "review_scores_rating": 95.0, "review_scores_accuracy": 10.0,
        "review_scores_cleanliness": 10.0, "review_scores_checkin": 10.0,
        "review_scores_communication": 10.0, "review_scores_location": 10.0,
        "room_type": "Entire place", "property_type": "Entire apartment",
        "neighbourhood": "Buttes-Montmartre",
        "host_response_time": "within an hour",
        "host_is_superhost": False, "host_identity_verified": True,
        "instant_bookable": False,
    }])
    pred = float(pipe.predict(X)[0])
    assert pred > 0, f"prediction must be positive, got {pred}"
    assert pred < 5000, f"prediction must be under 5000 EUR for Paris, got {pred}"
