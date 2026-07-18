"""Frozen configuration for the EEG seizure detection pipeline.

These values are selected after the dataset-analysis/design-review phase. They
remain centralized here so pipeline modules avoid hardcoded paths and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = Path("D:/CHBMIT")
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    features_dir = Path(r"D:\CHBMIT\features")
    model_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    metrics_dir: Path = PROJECT_ROOT / "results" / "metrics"
    plots_dir: Path = PROJECT_ROOT / "results" / "plots"
    predictions_dir: Path = PROJECT_ROOT / "results" / "predictions"
    reports_dir: Path = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class SignalConfig:
    """EEG preprocessing parameters."""

    low_freq_hz: float = 0.5
    high_freq_hz: float = 40.0
    notch_freq_hz: float = 50.0
    target_sampling_frequency_hz: float | None = None
    normalize: bool = True
    channel_strategy: str = "common_subset"
    selected_channels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SegmentationConfig:
    """Windowing and labeling parameters."""

    window_size_seconds: float = 4.0
    overlap_fraction: float = 0.5
    labeling_strategy: str = "overlap_any_seizure_interval"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction parameters."""

    frequency_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 40.0),
        }
    )


@dataclass(frozen=True)
class ModelConfig:
    """Classical ML model parameters."""

    model_names: tuple[str, ...] = ("logistic_regression", "random_forest")
    random_seed: int = 42
    test_patients: tuple[str, ...] = ()
    logistic_regression_max_iter: int = 1000
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = None
    class_weight: str = "balanced"


@dataclass(frozen=True)
class ExplainabilityConfig:
    """SHAP explainability parameters."""

    shap_background_samples: int = 100
    shap_explain_samples: int = 200
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    signal: SignalConfig = field(default_factory=SignalConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    log_level: str = "INFO"


CONFIG = PipelineConfig()
