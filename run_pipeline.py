"""
AnalystLab Africa - Airbnb Listing Price Prediction
Master pipeline runner.

File naming convention: every script and report is prefixed with
`dtc_` (Daniel Thomas - Track C) plus a two-digit stage number,
so the whole project is uniquely identifiable and reproducible.

Usage (from the project root, with .venv active):
    python run_pipeline.py status       # show project status and file check
    python run_pipeline.py inspect      # data inspection (Stage 4)
    python run_pipeline.py clean        # data cleaning (Stage 5)
    python run_pipeline.py eda          # exploratory data analysis (Stage 6)
    python run_pipeline.py train        # model training (Stage 8)
    python run_pipeline.py evaluate     # evaluation (Stage 9)
    python run_pipeline.py explain      # SHAP explanations (Stage 10)
    python run_pipeline.py test         # run pytest (Stage 12)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    """Read config.yaml and return it as a dictionary."""
    if not CONFIG_PATH.exists():
        sys.exit(f"ERROR: config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_module(module_path: str, *extra_args: str) -> None:
    """Run a Python module as a subprocess, forwarding any extra args."""
    cmd = [sys.executable, module_path, *extra_args]
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"ERROR: {module_path} exited with code {result.returncode}")


def run_pytest() -> None:
    cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT)


def show_status(cfg: dict) -> None:
    print("=" * 60)
    print("PROJECT STATUS")
    print("=" * 60)
    print(f"Project : {cfg['project']['name']}")
    print(f"Author  : {cfg['project']['author']}")
    print(f"Cohort  : {cfg['project']['cohort']}")
    print(f"Track   : {cfg['project']['track']}")
    print(f"Root    : {ROOT}")
    print("\nFile check:")
    for key, value in cfg["paths"].items():
        p = ROOT / value
        exists = "OK " if p.exists() else "-- "
        print(f"  [{exists}] {key:22s} -> {value}")
    print("\nTarget column   :", cfg["target"]["column"])
    print("Currency strategy:", cfg["target"]["strategy"])
    print("Split group_by  :", cfg["split"]["group_by"])
    print("Models          :", ", ".join(cfg["models"]["train_list"]))
    print("\nNumeric features:", len(cfg["features"]["numeric"]))
    print("Categorical feats:", len(cfg["features"]["categorical"]))
    print("Excluded cols   :", len(cfg["features"]["excluded"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Airbnb Listing Price Prediction - pipeline runner"
    )
    parser.add_argument(
        "stage",
        choices=[
            "status", "inspect", "clean", "eda",
            "train", "evaluate", "explain", "test",
        ],
        help="Which pipeline stage to run.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.stage == "status":
        show_status(cfg)
        return

    if args.stage == "inspect":
        run_module("src/dtc_01_data_inspection.py")
    elif args.stage == "clean":
        run_module("src/dtc_03_data_cleaning.py")
    elif args.stage == "eda":
        run_module("src/dtc_04_eda.py")
    elif args.stage == "train":
        run_module("src/dtc_06_train_model.py")
    elif args.stage == "evaluate":
        run_module("src/dtc_07_evaluate_model.py")
    elif args.stage == "explain":
        run_module("src/dtc_08_explain.py")
    elif args.stage == "test":
        run_pytest()


if __name__ == "__main__":
    main()