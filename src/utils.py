"""Shared utilities for the EEG seizure detection project."""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """Configure project-wide logging.

    Parameters
    ----------
    level:
        Logging level name, for example ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to JSON with stable formatting."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON data from disk."""
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Persist a Python object with pickle."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle object from disk."""
    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def create_project_directories(paths: Any) -> None:
    """Create all path attributes ending with ``_dir`` on a config object."""
    for name in dir(paths):
        if name.endswith("_dir"):
            value = getattr(paths, name)
            if isinstance(value, Path):
                ensure_directory(value)


def validate_directory(path: str | Path) -> Path:
    """Validate that a directory exists.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a directory.
    """
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return directory
