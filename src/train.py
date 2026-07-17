"""Training utilities for classical EEG seizure detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ModelConfig
from src.utils import ensure_directory, save_pickle, load_pickle


def split_by_patient(X, y, metadata, test_patients: tuple[str, ...] = ()):  # noqa: ANN001
    """Perform a patient-wise train/test split with leakage checks."""
    if "patient_id" not in metadata.columns:
        raise ValueError("metadata must contain a patient_id column")

    unique_patients = sorted(metadata["patient_id"].unique().tolist())
    if not test_patients:
        n_test = max(1, int(round(0.2 * len(unique_patients))))
        test_patients = tuple(unique_patients[-n_test:])

    train_mask = ~metadata["patient_id"].isin(test_patients)
    test_mask = metadata["patient_id"].isin(test_patients)

    train_patients = set(metadata.loc[train_mask, "patient_id"].unique())
    heldout_patients = set(metadata.loc[test_mask, "patient_id"].unique())
    overlap = train_patients.intersection(heldout_patients)
    if overlap:
        raise AssertionError(f"Patient leakage detected: {overlap}")
    if not heldout_patients:
        raise ValueError("No test patients found. Check ModelConfig.test_patients.")
    if not train_patients:
        raise ValueError("No training patients remain after split.")

    return (
        X.loc[train_mask].reset_index(drop=True),
        X.loc[test_mask].reset_index(drop=True),
        y.loc[train_mask].reset_index(drop=True),
        y.loc[test_mask].reset_index(drop=True),
        metadata.loc[train_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True),
    )


def build_logistic_regression_model(config: ModelConfig):
    """Build a scaled Logistic Regression pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.logistic_regression_max_iter,
                    class_weight=config.class_weight,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def build_random_forest_model(config: ModelConfig):
    """Build a Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=config.random_forest_n_estimators,
        max_depth=config.random_forest_max_depth,
        class_weight=config.class_weight,
        random_state=config.random_seed,
        n_jobs=-1,
    )


MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression_model,
    "random_forest": build_random_forest_model,
}


def get_model(model_name: str, config: ModelConfig):
    """Build a model by registry name."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](config)


def train_model(model, X_train, y_train):  # noqa: ANN001
    """Fit a scikit-learn compatible model."""
    return model.fit(X_train, y_train)


def train_all_models(X_train, y_train, config: ModelConfig) -> dict[str, Any]:  # noqa: ANN001
    """Train all configured models."""
    trained = {}
    for model_name in config.model_names:
        model = get_model(model_name, config)
        trained[model_name] = train_model(model, X_train, y_train)
    return trained


def save_model(model, output_path: Path) -> None:  # noqa: ANN001
    """Persist a trained model to disk."""
    ensure_directory(Path(output_path).parent)
    save_pickle(model, output_path)


def load_model(model_path: Path):
    """Load a persisted model from disk."""
    return load_pickle(model_path)


def save_all_models(models: dict[str, Any], model_dir: Path) -> None:
    """Persist all trained models using ``<name>.pkl`` filenames."""
    ensure_directory(model_dir)
    for name, model in models.items():
        save_model(model, Path(model_dir) / f"{name}.pkl")
