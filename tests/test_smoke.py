"""Smoke tests - verify the project environment is functional."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def test_config_exists():
    assert CONFIG_PATH.exists(), "config.yaml is missing from project root"


def test_config_loads():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "project" in cfg
    assert "paths" in cfg
    assert "features" in cfg
    assert cfg["target"]["column"] == "price"
    assert cfg["split"]["group_by"] == "host_id"


def test_python_version():
    import sys
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_core_libraries_importable():
    import pandas     # noqa: F401
    import numpy      # noqa: F401
    import duckdb     # noqa: F401
    import sklearn    # noqa: F401
    import shap       # noqa: F401
    import yaml       # noqa: F401
    import joblib     # noqa: F401


def test_project_folders_exist():
    for folder in ["data/raw", "data/processed", "src", "models",
                   "reports", "visualizations", "tests", "app"]:
        assert (ROOT / folder).exists(), f"Missing folder: {folder}"


def test_configured_features_exist_in_dataset():
    """Guardrail: every feature listed in config.yaml must exist in Listings.csv."""
    import pandas as pd

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    listing_path = ROOT / cfg["paths"]["listings_raw"]
    assert listing_path.exists(), f"Listings.csv missing: {listing_path}"

    cols = set(pd.read_csv(listing_path, nrows=0).columns)

    declared = (
        cfg["features"]["numeric"]
        + cfg["features"]["categorical"]
        + cfg["features"]["excluded"]
    )
    missing = [f for f in declared if f not in cols]
    assert not missing, (
        f"config.yaml references columns not in Listings.csv: {missing}"
    )