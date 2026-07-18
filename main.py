"""Command-line entry point for the EEG seizure detection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CONFIG, PipelineConfig
from src.data_loader import get_seizure_intervals_for_recording, load_dataset
from src.evaluate import evaluate_all_models
from src.explain import run_random_forest_explainability
from src.feature_extraction import extract_feature_matrix
from src.segmentation import segment_recording
from src.train import save_all_models, split_by_patient, train_all_models
from src.utils import create_project_directories, ensure_directory, set_random_seed, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureShardPaths:
    """Paths for per-recording persisted feature artifacts."""

    features: Path
    labels: Path
    metadata: Path


def get_feature_shard_paths(features_dir: Path, patient_id: str, recording_id: str) -> FeatureShardPaths:
    """Build stable file paths for one recording's extracted features."""
    shard_dir = ensure_directory(Path(features_dir) / "shards")
    shard_name = f"patient_{patient_id}_recording_{recording_id}"
    return FeatureShardPaths(
        features=shard_dir / f"{shard_name}_features.parquet",
        labels=shard_dir / f"{shard_name}_labels.csv",
        metadata=shard_dir / f"{shard_name}_metadata.csv",
    )


def persist_feature_artifacts(X, y, metadata, paths: FeatureShardPaths) -> None:  # noqa: ANN001
    """Persist one recording's feature artifacts to disk."""
    ensure_directory(paths.features.parent)
    try:
        X.to_parquet(paths.features, index=False)
    except Exception:  # pragma: no cover - parquet backend availability varies
        X.to_csv(paths.features.with_suffix(".csv"), index=False)
    y.to_csv(paths.labels, index=False)
    metadata.to_csv(paths.metadata, index=False)


def _load_feature_frame(path: Path) -> pd.DataFrame:
    """Load a persisted feature shard from parquet or CSV."""
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv") if path.suffix == ".parquet" else path
    return pd.read_csv(csv_path)


def load_persisted_feature_dataset(features_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load all persisted feature shards for downstream training."""
    shard_dir = Path(features_dir) / "shards"
    feature_paths = sorted(shard_dir.glob("*_features.parquet"))
    if not feature_paths:
        feature_paths = sorted(shard_dir.glob("*_features.csv"))
    if not feature_paths:
        raise RuntimeError("No persisted feature shards found. Run feature extraction first.")

    feature_frames: list[pd.DataFrame] = []
    label_frames: list[pd.Series] = []
    metadata_frames: list[pd.DataFrame] = []
    
    for i, feature_path in enumerate(feature_paths):
        base = feature_path.name.removesuffix("_features.parquet").removesuffix("_features.csv")
        label_path = shard_dir / f"{base}_labels.csv"
        metadata_path = shard_dir / f"{base}_metadata.csv"

        df = _load_feature_frame(feature_path)

        if i < 5:
            print(feature_path.name, "shape=", df.shape, "n_cols=", len(df.columns))
            print("  first10cols=", list(df.columns[:10]))

        feature_frames.append(df)
        label_frames.append(pd.read_csv(label_path).iloc[:, 0].rename("label"))
        metadata_frames.append(pd.read_csv(metadata_path))

    col_counts = [len(df.columns) for df in feature_frames]
    col_sets = [set(df.columns) for df in feature_frames]
    union_cols = set().union(*col_sets)
    print("=== PRE-CONCAT DIAGNOSTICS ===")
    print("n_shards:", len(feature_frames))
    print("per-shard col counts: min=", min(col_counts), "max=", max(col_counts), "mean=", sum(col_counts) / len(col_counts))
    print("sum of per-shard col counts (axis=1-style bug bound):", sum(col_counts))
    print("size of UNION of column names (axis=0 outer-join bound):", len(union_cols))
    print("total rows across shards:", sum(df.shape[0] for df in feature_frames))
    X = pd.concat(feature_frames, ignore_index=True)
    print("=== POST-CONCAT ===")
    print("X.shape:", X.shape)
    y = pd.concat(label_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    return X, y, metadata


def save_feature_artifacts(X, y, metadata, features_dir: Path) -> None:  # noqa: ANN001
    """Save extracted feature matrix, labels, and metadata."""
    ensure_directory(features_dir)
    X.to_csv(features_dir / "features.csv", index=False)
    y.to_csv(features_dir / "labels.csv", index=False)
    metadata.to_csv(features_dir / "metadata.csv", index=False)


def run_pipeline(config: PipelineConfig = CONFIG) -> dict[str, dict[str, float]]:
    """Run the full EEG seizure detection pipeline."""
    setup_logging(config.log_level)
    set_random_seed(config.models.random_seed)
    create_project_directories(config.paths)

    LOGGER.info("Loading CHB-MIT dataset from %s", config.paths.raw_data_dir)
    recordings, annotations = load_dataset(config.paths.raw_data_dir, preload=False)

    LOGGER.info("Processing %s recordings one at a time", len(recordings))
    for index, recording in enumerate(recordings, start=1):
        shard_paths = get_feature_shard_paths(
            config.paths.features_dir,
            recording.patient_id,
            recording.recording_id,
        )

        feature_exists = (
            shard_paths.features.exists()
            or shard_paths.features.with_suffix(".csv").exists()
        )

        if (
            feature_exists
            and shard_paths.labels.exists()
            and shard_paths.metadata.exists()
        ):
            LOGGER.info(
                "[%s/%s] Skipping recording %s for patient %s because persisted shards already exist",
                index,
                len(recordings),
                recording.recording_id,
                recording.patient_id,
            )
            continue

        seizure_intervals = get_seizure_intervals_for_recording(
            recording_id=recording.recording_id,
            annotations=annotations,
            patient_id=recording.patient_id,
        )

        LOGGER.info(
            "[%s/%s] Segmenting recording %s for patient %s",
            index,
            len(recordings),
            recording.recording_id,
            recording.patient_id,
        )
        windows = segment_recording(
            recording=recording,
            seizure_intervals=seizure_intervals,
            segmentation_config=config.segmentation,
            signal_config=config.signal,
        )
        if not windows:
            LOGGER.warning("No windows generated for recording %s", recording.recording_id)
            continue

        LOGGER.info(
            "[%s/%s] Extracting features from %s windows",
            index,
            len(recordings),
            len(windows),
        )
        X_recording, y_recording, metadata_recording = extract_feature_matrix(windows, config.features)
        persist_feature_artifacts(X_recording, y_recording, metadata_recording, shard_paths)
        del windows, X_recording, y_recording, metadata_recording

    LOGGER.info("Loading persisted feature dataset from disk")
    X, y, metadata = load_persisted_feature_dataset(config.paths.features_dir)

    print("=== FINAL DATASET ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("metadata shape:", metadata.shape)

    save_feature_artifacts(X, y, metadata, config.paths.features_dir)

    LOGGER.info("Performing patient-wise train/test split")
    X_train, X_test, y_train, y_test, train_metadata, test_metadata = split_by_patient(
        X=X,
        y=y,
        metadata=metadata,
        test_patients=config.models.test_patients,
    )
    train_metadata.to_csv(config.paths.features_dir / "train_metadata.csv", index=False)
    test_metadata.to_csv(config.paths.features_dir / "test_metadata.csv", index=False)

    LOGGER.info("Training models: %s", ", ".join(config.models.model_names))
    models = train_all_models(X_train, y_train, config.models)
    save_all_models(models, config.paths.model_dir)

    LOGGER.info("Evaluating models")
    metrics = evaluate_all_models(models, X_test, y_test, config.paths.results_dir)

    if "random_forest" in models:
        LOGGER.info("Running SHAP explainability for Random Forest")
        run_random_forest_explainability(
            model=models["random_forest"],
            X_train=X_train,
            X_test=X_test,
            output_dir=config.paths.results_dir,
            config=config.explainability,
        )

    LOGGER.info("Pipeline complete")
    return metrics


def main() -> None:
    """CLI entry point."""
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()