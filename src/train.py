"""Training utilities for classical EEG seizure detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ModelConfig
from src.utils import ensure_directory, save_pickle, load_pickle

# sklearn's RandomForestClassifier converts its input to float32 internally
# regardless of what dtype is passed in (verified empirically: predict_proba is
# bit-identical whether the model is fit on float64 or float32 input), so storing
# X as float32 is a true, zero-cost no-op for random_forest.
#
# LogisticRegression + StandardScaler do NOT do this -- they compute in whatever
# dtype they're given. IMPORTANT, and verified rather than assumed: once X has
# already been downcast float64->float32, casting it back up to float64 here does
# NOT recover the precision lost at that first downcast (a float64->float32->
# float64 round-trip is measurably not the identity -- confirmed directly, not
# inferred). So this cast does not make logistic_regression's output bit-identical
# to a pipeline that stored X as float64 throughout; it only prevents further
# precision loss from computing in float32 on top of the already-reduced values.
# The actual discrepancy this leaves versus an all-float64 pipeline is very small
# (~1e-8 in predict_proba, empirically measured) but it is real and should be
# disclosed, not glossed over.
MODEL_DTYPES: dict[str, np.dtype | None] = {
    "logistic_regression": np.float64,
    "random_forest": np.float32,
    "xgboost": np.float32,
}


def prepare_features_for_model(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Cast X to the dtype a given model expects.

    For random_forest this is a proven no-op on output. For logistic_regression
    this does NOT restore precision already lost if X was previously downcast to
    float32 upstream (see the comment above MODEL_DTYPES) -- it only prevents
    additional precision loss from here on. Returns X unchanged (no copy) when
    it's already the right dtype; only allocates a new array when a cast is
    actually required.
    """
    target_dtype = MODEL_DTYPES.get(model_name)
    if target_dtype is None or X.to_numpy(copy=False).dtype == target_dtype:
        return X
    return X.astype(target_dtype, copy=False)


def log_class_distribution(y: pd.Series, label: str) -> dict[int, int]:
    """Return and log exact class counts for a binary label series."""
    counts = y.value_counts(dropna=False).sort_index().to_dict()
    total = int(len(y))
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    print(f"{label} class distribution: total={total}, class_0={negatives}, class_1={positives}")
    return {int(k): int(v) for k, v in counts.items()}


def balanced_binary_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible balanced subset for binary training.

    Keeps all minority-class samples and randomly samples the same number of
    majority-class samples, using a fixed RNG seed for reproducibility.
    This applies only to the training set and leaves the held-out patient test
    split unchanged.
    """
    y_values = y.to_numpy(copy=False)
    minority_mask = y_values == 1
    majority_mask = y_values == 0

    n_minority = int(minority_mask.sum())
    n_majority = int(majority_mask.sum())
    if n_minority == 0 or n_majority == 0:
        return X, y

    rng = np.random.default_rng(random_state)
    minority_indices = np.flatnonzero(minority_mask)
    majority_indices = np.flatnonzero(majority_mask)
    sampled_majority = rng.choice(majority_indices, size=n_minority, replace=False)
    selected_indices = np.sort(np.concatenate([minority_indices, sampled_majority]))

    return X.iloc[selected_indices].reset_index(drop=True), y.iloc[selected_indices].reset_index(drop=True)


def split_by_patient(X, y, metadata, test_patients: tuple[str, ...] = ()):  # noqa: ANN001
    """Perform a patient-wise train/test split with leakage checks.

    Builds train_mask/test_mask as plain numpy boolean arrays (not pandas boolean
    Series) and indexes X's underlying buffer directly via .to_numpy() instead of
    going through pandas' .loc[]. For the multi-million-row X this matters because
    .loc[boolean_series] additionally pays for index alignment before it can even
    begin the positional boolean-mask copy that actually produces the result --
    with a numpy boolean array there's no index to align, so it's a single direct
    copy. The result is wrapped back into a DataFrame (unavoidable: downstream code
    such as explainability needs .columns), so this does not eliminate the
    fundamental one-copy-per-split cost, but it removes the redundant work pandas'
    label-based machinery does on top of that, and avoids ever materializing a
    boolean pandas Series/reset_index chain for the large frame.
    """
    if "patient_id" not in metadata.columns:
        raise ValueError("metadata must contain a patient_id column")

    patient_id_values = metadata["patient_id"].to_numpy()
    unique_patients = sorted(set(patient_id_values.tolist()))
    if not test_patients:
        n_test = max(1, int(round(0.2 * len(unique_patients))))
        test_patients = tuple(unique_patients[-n_test:])

    test_patients_set = set(test_patients)
    test_mask = np.fromiter(
        (patient_id in test_patients_set for patient_id in patient_id_values),
        dtype=bool,
        count=len(patient_id_values),
    )
    train_mask = ~test_mask

    train_patients = set(patient_id_values[train_mask].tolist())
    heldout_patients = set(patient_id_values[test_mask].tolist())
    overlap = train_patients.intersection(heldout_patients)
    if overlap:
        raise AssertionError(f"Patient leakage detected: {overlap}")
    if not heldout_patients:
        raise ValueError("No test patients found. Check ModelConfig.test_patients.")
    if not train_patients:
        raise ValueError("No training patients remain after split.")

    # X.to_numpy(copy=False) returns a view onto the existing buffer (no copy) for
    # a single-dtype DataFrame; the boolean-mask indexing that follows is the one
    # unavoidable copy per split (numpy fancy/boolean indexing always copies -- a
    # genuinely new, smaller array is required here since train and test must be
    # separate, disjoint buffers for the models to train/evaluate on).
    columns = X.columns
    X_values = X.to_numpy(copy=False)
    X_train = pd.DataFrame(X_values[train_mask], columns=columns, copy=False)
    X_test = pd.DataFrame(X_values[test_mask], columns=columns, copy=False)

    y_values = y.to_numpy(copy=False)
    y_train = pd.Series(y_values[train_mask], name=y.name)
    y_test = pd.Series(y_values[test_mask], name=y.name)

    # metadata is only 4 columns (patient_id, recording_id, start/end times), so
    # it isn't the memory bottleneck -- .loc[] with a numpy boolean array (rather
    # than a pandas boolean Series) still skips the index-alignment step.
    train_metadata = metadata.loc[train_mask].reset_index(drop=True)
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, train_metadata, test_metadata


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


def build_xgboost_model(config: ModelConfig, scale_pos_weight: float):
    """Build an XGBoost classifier with imbalance-aware weighting."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise ImportError("xgboost is required for the XGBoost model. Install requirements.txt.") from exc

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression_model,
    "random_forest": build_random_forest_model,
    "xgboost": build_xgboost_model,
}


def get_model(model_name: str, config: ModelConfig):
    """Build a model by registry name."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    if model_name == "xgboost":
        raise ValueError("XGBoost model construction requires scale_pos_weight; use get_model_with_training_stats().")
    return MODEL_REGISTRY[model_name](config)


def get_model_with_training_stats(model_name: str, config: ModelConfig, y_train: pd.Series):
    """Build a model that may need training-set statistics."""
    if model_name == "xgboost":
        y_values = y_train.to_numpy(copy=False)
        n_positive = int(np.sum(y_values == 1))
        n_negative = int(np.sum(y_values == 0))
        scale_pos_weight = float(n_negative / max(n_positive, 1))
        return build_xgboost_model(config, scale_pos_weight=scale_pos_weight)
    return get_model(model_name, config)


def train_model(model, X_train, y_train):  # noqa: ANN001
    """Fit a scikit-learn compatible model."""
    return model.fit(X_train, y_train)


def train_all_models(X_train, y_train, config: ModelConfig) -> dict[str, Any]:  # noqa: ANN001
    """Train all configured models.

    Casts X_train to each model's required dtype (see MODEL_DTYPES /
    prepare_features_for_model) immediately before that model's .fit() call, and
    lets the cast copy (if any) go out of scope again right after -- so at most one
    model's transient float64 copy is alive at a time, not all of them at once.
    """
    trained = {}
    for model_name in config.model_names:
        model = get_model_with_training_stats(model_name, config, y_train)
        model_X_train = prepare_features_for_model(X_train, model_name)
        if model_name == "logistic_regression":
            model_X_train, model_y_train = balanced_binary_subsample(
                model_X_train,
                y_train,
                random_state=config.random_seed,
            )
        else:
            model_y_train = y_train
        trained[model_name] = train_model(model, model_X_train, model_y_train)
        del model_X_train
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
