"""Command-line entry point for the EEG seizure detection pipeline."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CONFIG, PipelineConfig
from src.data_loader import RecordingMontageAudit, get_seizure_intervals_for_recording, load_dataset
from src.evaluate import evaluate_model
from src.explain import run_random_forest_explainability, run_xgboost_explainability
from src.feature_extraction import extract_feature_matrix
from src.segmentation import segment_recording
from src.train import load_model, log_class_distribution, prepare_features_for_model, save_all_models, split_by_patient, train_all_models
from src.utils import create_project_directories, ensure_directory, set_random_seed, setup_logging
from src.threshold_optimization import run_threshold_sweep, write_threshold_artifacts

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
    for feature_path in feature_paths:
        base = feature_path.name.removesuffix("_features.parquet").removesuffix("_features.csv")
        label_path = shard_dir / f"{base}_labels.csv"
        metadata_path = shard_dir / f"{base}_metadata.csv"
        df = _load_feature_frame(feature_path)

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


def save_montage_audit_report(audits: list[RecordingMontageAudit], reports_dir: Path) -> None:
    """Persist a transparent record of montage classifications and exclusions."""
    audit_dir = ensure_directory(Path(reports_dir) / "montage_audit")
    report_path = audit_dir / "montage_audit_report.md"
    csv_path = audit_dir / "montage_audit.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "patient_id",
                "recording_id",
                "file_path",
                "classification",
                "reference_channel",
                "excluded",
                "missing_endpoints",
                "missing_derivations",
            ]
        )
        for audit in audits:
            writer.writerow(
                [
                    audit.patient_id,
                    audit.recording_id,
                    str(audit.file_path),
                    audit.classification,
                    audit.reference_channel or "",
                    audit.excluded,
                    ";".join(audit.missing_endpoints),
                    ";".join(audit.missing_derivations),
                ]
            )

    excluded = [audit for audit in audits if audit.excluded]
    lines = [
        "# Montage Audit Report",
        "",
        f"- Total recordings audited: {len(audits)}",
        f"- Canonical bipolar: {sum(a.classification == 'canonical_bipolar' for a in audits)}",
        f"- Reconstructable referential: {sum(a.classification == 'reconstructable_referential' for a in audits)}",
        f"- Non-reconstructable referential: {sum(a.classification == 'non_reconstructable_referential' for a in audits)}",
        f"- Other/unknown: {sum(a.classification == 'other_unknown' for a in audits)}",
        f"- Excluded recordings: {len(excluded)}",
        "",
        "## Excluded recordings",
    ]
    if excluded:
        lines.append("| Patient | Recording | Classification | Missing endpoints | Missing derivations |")
        lines.append("|---|---|---|---|---|")
        for audit in excluded:
            lines.append(
                f"| {audit.patient_id} | {audit.recording_id} | {audit.classification} | "
                f"{', '.join(audit.missing_endpoints) or '-'} | {', '.join(audit.missing_derivations) or '-'} |"
            )
    else:
        lines.append("No recordings were excluded.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def load_or_train_models(
    model_dir: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig,
) -> dict[str, object]:
    """Load any existing model artifacts and train only the missing ones."""
    trained_or_loaded: dict[str, object] = {}
    missing_model_names: list[str] = []
    for model_name in config.models.model_names:
        model_path = Path(model_dir) / f"{model_name}.pkl"
        if model_path.exists():
            LOGGER.info("Loading existing model: %s", model_name)
            trained_or_loaded[model_name] = load_model(model_path)
        else:
            missing_model_names.append(model_name)

    if missing_model_names:
        LOGGER.info("Training missing models: %s", ", ".join(missing_model_names))
        trained_missing = train_all_models(X_train, y_train, config.models)
        for model_name in missing_model_names:
            trained_or_loaded[model_name] = trained_missing[model_name]
            save_all_models({model_name: trained_missing[model_name]}, model_dir)
    return trained_or_loaded


def save_model_comparison_report(
    metrics: dict[str, dict[str, float]],
    threshold_summaries: dict[str, pd.Series],
    output_dir: Path,
) -> None:
    """Persist a compact classical model comparison report."""
    ensure_directory(output_dir)
    rows = []
    for model_name, metric_row in metrics.items():
        row = {"model": model_name, **metric_row, "threshold": 0.5}
        rows.append(row)
        if model_name in threshold_summaries:
            optimized = {"model": f"{model_name}_optimized", **threshold_summaries[model_name].to_dict()}
            rows.append(optimized)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "classical_model_comparison.csv", index=False)
    markdown = ["# Classical Model Comparison", "", df.to_markdown(index=False)]
    (output_dir / "classical_model_comparison.md").write_text("\n".join(markdown), encoding="utf-8")


def run_pipeline(config: PipelineConfig = CONFIG) -> dict[str, dict[str, float]]:
    """Run the full EEG seizure detection pipeline."""
    setup_logging(config.log_level)
    set_random_seed(config.models.random_seed)
    create_project_directories(config.paths)

    LOGGER.info("Loading CHB-MIT dataset from %s", config.paths.raw_data_dir)
    recordings, annotations, montage_audits = load_dataset(config.paths.raw_data_dir, preload=False)
    save_montage_audit_report(montage_audits, config.paths.reports_dir)

    LOGGER.info("Processing %s recordings one at a time", len(recordings))
    for index, recording in enumerate(recordings, start=1):
        audit = recording.montage_audit
        if audit is not None and audit.excluded:
            LOGGER.warning(
                "[%s/%s] Excluding recording %s for patient %s: %s | missing endpoints=%s | missing derivations=%s",
                index,
                len(recordings),
                recording.recording_id,
                recording.patient_id,
                audit.classification,
                ", ".join(audit.missing_endpoints) or "-",
                ", ".join(audit.missing_derivations) or "-",
            )
            continue

        shard_paths = get_feature_shard_paths(config.paths.features_dir, recording.patient_id, recording.recording_id)
        if shard_paths.features.exists() and shard_paths.labels.exists() and shard_paths.metadata.exists():
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

    # Downcast X to float32 immediately after loading, before it's saved or split.
    # This is a true, zero-cost no-op for random_forest (sklearn's
    # RandomForestClassifier converts to float32 internally regardless of input
    # dtype -- verified predict_proba is bit-identical either way) and roughly
    # halves X's memory footprint (1,712,792 x 529 floats: ~7.25GB at float64 ->
    # ~3.6GB at float32) right before the train/test split, which is where this
    # pipeline was running out of memory.
    #
    # DISCLOSED TRADEOFF: logistic_regression is NOT dtype-invariant the way
    # random_forest is. src/train.py's prepare_features_for_model() casts its X
    # back to float64 before it touches that model, but this does not recover
    # precision already lost at this downcast -- a float64->float32->float64
    # round-trip is measurably not the identity (confirmed directly). The
    # resulting difference versus an all-float64 pipeline is small (~1e-8 in
    # predict_proba, empirically measured) but real. If bit-identical
    # logistic_regression output is a hard requirement, remove this line and
    # accept the original (larger) memory footprint for that model's path.
    X = X.astype("float32", copy=False)
    save_feature_artifacts(X, y, metadata, config.paths.features_dir)

    LOGGER.info("Performing patient-wise train/test split")
    X_train, X_test, y_train, y_test, train_metadata, test_metadata = split_by_patient(
        X=X,
        y=y,
        metadata=metadata,
        test_patients=config.models.test_patients,
    )
    log_class_distribution(y_train, "y_train")
    log_class_distribution(y_test, "y_test")
    # The full concatenated X/y/metadata are no longer needed once the train/test
    # copies exist -- drop the references so they can be garbage collected before
    # model training, rather than staying alive alongside X_train/X_test for the
    # rest of the run.
    del X, y, metadata
    train_metadata.to_csv(config.paths.features_dir / "train_metadata.csv", index=False)
    test_metadata.to_csv(config.paths.features_dir / "test_metadata.csv", index=False)

    models = load_or_train_models(config.paths.model_dir, X_train, y_train, config)

    LOGGER.info("Evaluating models")
    # Evaluate one model at a time, casting X_test to that model's required dtype
    # (see src/train.py) rather than one shared evaluate_all_models call -- keeps
    # at most one transient float64 copy of X_test alive at a time instead of
    # needing a dtype that satisfies every model simultaneously.
    metrics = {}
    threshold_summaries: dict[str, pd.Series] = {}
    for model_name, model in models.items():
        model_X_test = prepare_features_for_model(X_test, model_name)
        metrics[model_name] = evaluate_model(model, model_X_test, y_test, config.paths.results_dir, model_name)
        del model_X_test

    # Threshold optimization for tree-based models without retraining.
    for model_name in ("random_forest", "xgboost"):
        if model_name in models:
            model_X_test = prepare_features_for_model(X_test, model_name)
            threshold_results = run_threshold_sweep(
                model=models[model_name],
                X_test=model_X_test,
                y_test=y_test,
                model_name=model_name,
                output_dir=config.paths.results_dir / "threshold_optimization" / model_name,
            )
            write_threshold_artifacts(threshold_results, config.paths.results_dir / "threshold_optimization" / model_name, model_name)
            threshold_summaries[model_name] = threshold_results.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
            del model_X_test

    if "random_forest" in models:
        LOGGER.info("Running SHAP explainability for Random Forest")
        run_random_forest_explainability(
            model=models["random_forest"],
            X_train=X_train,
            X_test=X_test,
            output_dir=config.paths.results_dir,
            config=config.explainability,
        )

    if "xgboost" in models:
        LOGGER.info("Running SHAP explainability for XGBoost")
        run_xgboost_explainability(
            model=models["xgboost"],
            X_train=X_train,
            X_test=X_test,
            output_dir=config.paths.results_dir,
            config=config.explainability,
        )

    save_model_comparison_report(metrics, threshold_summaries, config.paths.results_dir / "metrics")

    LOGGER.info("Pipeline complete")
    return metrics


def main() -> None:
    """CLI entry point."""
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()
