"""Read-only access helpers for dashboard artifacts.

The dashboard must not recompute the ML pipeline. All functions in this module
only read precomputed artifacts from ``results/`` and gracefully handle missing
files by returning ``None`` or empty structures.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _resolve(path: str | Path) -> Path:
    return Path(path)


@lru_cache(maxsize=1)
def load_json_file(path: str | Path) -> dict[str, Any]:
    path = _resolve(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_csv_file(path: str | Path) -> pd.DataFrame:
    path = _resolve(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=8)
def load_image_path(path: str | Path) -> Path | None:
    path = _resolve(path)
    return path if path.exists() else None


def metric_json_path(model_name: str) -> Path:
    return RESULTS_DIR / "metrics" / f"{model_name}_metrics.json"


def plot_path(plot_name: str) -> Path:
    return RESULTS_DIR / "plots" / plot_name


def load_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        metrics = load_metrics(model_name)
        if metrics:
            rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows)


def load_metrics(model_name: str) -> dict[str, Any]:
    return load_json_file(metric_json_path(model_name))


def available_models() -> list[str]:
    models = []
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        if metric_json_path(model_name).exists():
            models.append(model_name)
    return models
