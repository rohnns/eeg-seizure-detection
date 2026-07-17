"""Dataset analysis for the CHB-MIT EEG seizure detection project.

This module runs before final pipeline implementation decisions are locked in.
It analyzes EDF metadata and CHB-MIT summary annotations to justify choices such
as window size, overlap, labeling strategy, preprocessing frequencies, and model
validation design.

Example
-------
python -m src.dataset_analysis --data-dir data/raw/chbmit --output-dir reports/dataset_analysis
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.utils import ensure_directory, save_json, setup_logging, validate_directory

LOGGER = logging.getLogger(__name__)


def require_pandas():
    """Import pandas lazily for report generation.

    This keeps lightweight parsing and window-estimation utilities testable even
    in environments where project dependencies have not yet been installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for dataset analysis reports. Install requirements.txt."
        ) from exc
    return pd


def require_matplotlib():
    """Import matplotlib lazily for optional plot generation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for dataset analysis plots. Install requirements.txt."
        ) from exc
    return plt


@dataclass(frozen=True)
class RecordingMetadata:
    """Metadata extracted from one EDF recording."""

    patient_id: str
    recording_id: str
    file_path: str
    duration_seconds: float
    sampling_frequency_hz: float
    n_channels: int
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class SeizureInterval:
    """One seizure interval parsed from a CHB-MIT summary file."""

    patient_id: str
    recording_id: str
    seizure_start_seconds: float
    seizure_end_seconds: float

    @property
    def duration_seconds(self) -> float:
        """Return seizure duration in seconds."""
        return max(0.0, self.seizure_end_seconds - self.seizure_start_seconds)


@dataclass(frozen=True)
class ImbalanceEstimate:
    """Class imbalance estimate for one window/overlap setting."""

    window_size_seconds: float
    overlap_fraction: float
    total_windows: int
    seizure_windows: int
    non_seizure_windows: int
    seizure_window_percent: float
    non_seizure_to_seizure_ratio: float | None


def discover_patient_dirs(data_dir: Path) -> list[Path]:
    """Return CHB-MIT patient directories containing EDF files."""
    patient_dirs = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and any(path.glob("*.edf"))
    ]
    return sorted(patient_dirs, key=lambda path: path.name.lower())


def discover_edf_files(data_dir: Path) -> list[Path]:
    """Return all EDF files below the dataset directory."""
    return sorted(data_dir.rglob("*.edf"), key=lambda path: str(path).lower())


def infer_patient_id(file_path: Path, data_dir: Path) -> str:
    """Infer the CHB-MIT patient ID from an EDF path."""
    relative_parts = file_path.relative_to(data_dir).parts
    if len(relative_parts) > 1:
        return relative_parts[0]
    match = re.match(r"(chb\d+)", file_path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def infer_recording_id(file_path: Path) -> str:
    """Infer recording ID from EDF file name."""
    return file_path.stem


def read_edf_metadata(file_path: Path, data_dir: Path) -> RecordingMetadata:
    """Read EDF header metadata without preloading signal samples.

    Parameters
    ----------
    file_path:
        Path to an EDF file.
    data_dir:
        Dataset root used to infer patient IDs.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for EDF metadata analysis. Install requirements.txt."
        ) from exc

    LOGGER.debug("Reading EDF metadata: %s", file_path)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
    sampling_frequency = float(raw.info["sfreq"])
    n_samples = raw.n_times
    duration_seconds = float(n_samples / sampling_frequency) if sampling_frequency else 0.0
    channel_names = tuple(str(channel) for channel in raw.ch_names)

    return RecordingMetadata(
        patient_id=infer_patient_id(file_path, data_dir),
        recording_id=infer_recording_id(file_path),
        file_path=str(file_path),
        duration_seconds=duration_seconds,
        sampling_frequency_hz=sampling_frequency,
        n_channels=len(channel_names),
        channel_names=channel_names,
    )


def find_summary_files(data_dir: Path) -> list[Path]:
    """Return CHB-MIT summary annotation text files."""
    return sorted(data_dir.rglob("*summary*.txt"), key=lambda path: str(path).lower())


def parse_chbmit_summary(summary_file: Path) -> list[SeizureInterval]:
    """Parse seizure intervals from a CHB-MIT summary file.

    The summary format varies slightly across mirrors. This parser handles the
    common PhysioNet text format with lines such as::

        File Name: chb01_03.edf
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    """
    patient_match = re.search(r"(chb\d+)", summary_file.name, flags=re.IGNORECASE)
    default_patient_id = patient_match.group(1).lower() if patient_match else summary_file.parent.name

    intervals: list[SeizureInterval] = []
    current_recording_id: str | None = None
    pending_starts: list[float] = []

    with summary_file.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            file_match = re.search(r"File Name:\s*(\S+)", line, flags=re.IGNORECASE)
            if file_match:
                current_recording_id = Path(file_match.group(1)).stem
                pending_starts = []
                continue

            start_match = re.search(
                r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9.]+)",
                line,
                flags=re.IGNORECASE,
            )
            if start_match:
                pending_starts.append(float(start_match.group(1)))
                continue

            end_match = re.search(
                r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9.]+)",
                line,
                flags=re.IGNORECASE,
            )
            if end_match and current_recording_id and pending_starts:
                start_time = pending_starts.pop(0)
                end_time = float(end_match.group(1))
                intervals.append(
                    SeizureInterval(
                        patient_id=default_patient_id,
                        recording_id=current_recording_id,
                        seizure_start_seconds=start_time,
                        seizure_end_seconds=end_time,
                    )
                )

    return intervals


def load_all_recording_metadata(data_dir: Path) -> list[RecordingMetadata]:
    """Load metadata for every EDF file in the dataset."""
    edf_files = discover_edf_files(data_dir)
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found below {data_dir}")

    metadata: list[RecordingMetadata] = []
    for index, edf_file in enumerate(edf_files, start=1):
        LOGGER.info("Reading EDF metadata %s/%s: %s", index, len(edf_files), edf_file.name)
        metadata.append(read_edf_metadata(edf_file, data_dir))
    return metadata


def load_all_seizure_intervals(data_dir: Path) -> list[SeizureInterval]:
    """Load all seizure annotations from CHB-MIT summary files."""
    summary_files = find_summary_files(data_dir)
    intervals: list[SeizureInterval] = []
    for summary_file in summary_files:
        LOGGER.info("Parsing summary file: %s", summary_file)
        intervals.extend(parse_chbmit_summary(summary_file))
    return intervals


def compute_channel_consistency(recordings: Iterable[RecordingMetadata]) -> dict[str, object]:
    """Summarize channel consistency across recordings and patients."""
    recordings = list(recordings)
    channel_sets = [frozenset(recording.channel_names) for recording in recordings]
    exact_montage_counts = Counter(tuple(recording.channel_names) for recording in recordings)
    channel_frequency = Counter(
        channel for recording in recordings for channel in recording.channel_names
    )
    common_channels = sorted(set.intersection(*map(set, channel_sets))) if channel_sets else []
    union_channels = sorted(set.union(*map(set, channel_sets))) if channel_sets else []

    patient_to_channels: dict[str, set[str]] = defaultdict(set)
    for recording in recordings:
        patient_to_channels[recording.patient_id].update(recording.channel_names)

    return {
        "n_unique_exact_channel_orders": len(exact_montage_counts),
        "most_common_channel_orders": [
            {"count": count, "channels": list(channels)}
            for channels, count in exact_montage_counts.most_common(5)
        ],
        "n_common_channels_all_recordings": len(common_channels),
        "common_channels_all_recordings": common_channels,
        "n_union_channels": len(union_channels),
        "union_channels": union_channels,
        "channel_presence_counts": dict(sorted(channel_frequency.items())),
        "channels_by_patient": {
            patient: sorted(channels) for patient, channels in sorted(patient_to_channels.items())
        },
    }


def compute_annotation_coverage(
    recordings: list[RecordingMetadata],
    intervals: list[SeizureInterval],
) -> dict[str, object]:
    """Check whether parsed seizure intervals map cleanly to discovered EDF files."""
    recording_keys = {(recording.patient_id, recording.recording_id) for recording in recordings}
    interval_keys = {(interval.patient_id, interval.recording_id) for interval in intervals}
    unmatched_interval_keys = sorted(interval_keys - recording_keys)
    seizure_recording_keys = sorted(interval_keys & recording_keys)

    invalid_intervals = [
        asdict(interval)
        for interval in intervals
        if interval.seizure_end_seconds <= interval.seizure_start_seconds
    ]

    return {
        "n_recordings_with_parsed_seizures": len(seizure_recording_keys),
        "n_annotation_recordings_not_found_as_edf": len(unmatched_interval_keys),
        "annotation_recordings_not_found_as_edf": [
            {"patient_id": patient_id, "recording_id": recording_id}
            for patient_id, recording_id in unmatched_interval_keys
        ],
        "n_invalid_intervals": len(invalid_intervals),
        "invalid_intervals": invalid_intervals,
    }


def interval_overlaps_window(
    window_start: float,
    window_end: float,
    intervals: Iterable[SeizureInterval],
) -> bool:
    """Return True if a window overlaps any seizure interval."""
    return any(
        window_start < interval.seizure_end_seconds
        and window_end > interval.seizure_start_seconds
        for interval in intervals
    )


def estimate_windows_for_recording(
    duration_seconds: float,
    seizure_intervals: list[SeizureInterval],
    window_size_seconds: float,
    overlap_fraction: float,
) -> tuple[int, int]:
    """Estimate total and seizure windows for one recording."""
    if duration_seconds <= 0 or window_size_seconds <= 0:
        return 0, 0

    step_seconds = window_size_seconds * (1.0 - overlap_fraction)
    if step_seconds <= 0:
        raise ValueError("overlap_fraction must be less than 1.0")

    if duration_seconds < window_size_seconds:
        return 0, 0

    total_windows = int(math.floor((duration_seconds - window_size_seconds) / step_seconds)) + 1
    seizure_windows = 0
    for window_index in range(total_windows):
        start = window_index * step_seconds
        end = start + window_size_seconds
        if interval_overlaps_window(start, end, seizure_intervals):
            seizure_windows += 1
    return total_windows, seizure_windows


def estimate_class_imbalance(
    recordings: list[RecordingMetadata],
    intervals: list[SeizureInterval],
    window_sizes: list[float],
    overlaps: list[float],
) -> list[ImbalanceEstimate]:
    """Estimate class imbalance after segmentation for candidate settings."""
    intervals_by_recording: dict[tuple[str, str], list[SeizureInterval]] = defaultdict(list)
    for interval in intervals:
        intervals_by_recording[(interval.patient_id, interval.recording_id)].append(interval)

    estimates: list[ImbalanceEstimate] = []
    for window_size in window_sizes:
        for overlap in overlaps:
            total_windows = 0
            seizure_windows = 0
            for recording in recordings:
                recording_intervals = intervals_by_recording.get(
                    (recording.patient_id, recording.recording_id), []
                )
                recording_total, recording_seizure = estimate_windows_for_recording(
                    duration_seconds=recording.duration_seconds,
                    seizure_intervals=recording_intervals,
                    window_size_seconds=window_size,
                    overlap_fraction=overlap,
                )
                total_windows += recording_total
                seizure_windows += recording_seizure

            non_seizure_windows = total_windows - seizure_windows
            seizure_percent = (
                100.0 * seizure_windows / total_windows if total_windows else 0.0
            )
            ratio = (
                non_seizure_windows / seizure_windows if seizure_windows else None
            )
            estimates.append(
                ImbalanceEstimate(
                    window_size_seconds=window_size,
                    overlap_fraction=overlap,
                    total_windows=total_windows,
                    seizure_windows=seizure_windows,
                    non_seizure_windows=non_seizure_windows,
                    seizure_window_percent=seizure_percent,
                    non_seizure_to_seizure_ratio=ratio,
                )
            )
    return estimates


def build_summary_report(
    recordings: list[RecordingMetadata],
    intervals: list[SeizureInterval],
    imbalance_estimates: list[ImbalanceEstimate],
    channel_consistency: dict[str, object],
    annotation_coverage: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable dataset summary report."""
    pd = require_pandas()
    recordings_df = pd.DataFrame(asdict(recording) for recording in recordings)
    seizure_durations = [interval.duration_seconds for interval in intervals]

    patient_ids = sorted(recordings_df["patient_id"].unique().tolist())
    seizure_patients = sorted({interval.patient_id for interval in intervals})

    design_recommendations = build_design_recommendations(
        recordings_df=recordings_df,
        intervals=intervals,
        imbalance_estimates=imbalance_estimates,
        channel_consistency=channel_consistency,
        annotation_coverage=annotation_coverage,
    )

    return {
        "dataset_overview": {
            "n_patients": len(patient_ids),
            "patient_ids": patient_ids,
            "n_recordings": len(recordings),
            "n_patients_with_seizures": len(seizure_patients),
            "patients_with_seizures": seizure_patients,
            "n_seizure_intervals": len(intervals),
        },
        "recording_duration_seconds": recordings_df["duration_seconds"].describe().to_dict(),
        "sampling_frequency_hz_counts": recordings_df["sampling_frequency_hz"].value_counts().sort_index().to_dict(),
        "channels_per_recording_counts": recordings_df["n_channels"].value_counts().sort_index().to_dict(),
        "seizure_duration_seconds": pd.Series(seizure_durations).describe().to_dict()
        if seizure_durations
        else {},
        "class_imbalance_estimates": [asdict(estimate) for estimate in imbalance_estimates],
        "channel_consistency": channel_consistency,
        "annotation_coverage": annotation_coverage,
        "design_recommendations": design_recommendations,
        "design_implications": infer_design_implications(
            recordings_df=recordings_df,
            intervals=intervals,
            imbalance_estimates=imbalance_estimates,
            channel_consistency=channel_consistency,
        ),
    }


def build_design_recommendations(
    recordings_df,
    intervals: list[SeizureInterval],
    imbalance_estimates: list[ImbalanceEstimate],
    channel_consistency: dict[str, object],
    annotation_coverage: dict[str, object],
) -> dict[str, object]:
    """Recommend config values from dataset analysis results.

    The output is intentionally reviewable instead of automatically editing
    ``config.py``. The team should inspect these recommendations, then freeze
    project configuration explicitly.
    """
    sampling_counts = recordings_df["sampling_frequency_hz"].value_counts().sort_values(ascending=False)
    dominant_sampling_frequency = float(sampling_counts.index[0]) if not sampling_counts.empty else None

    best_imbalance = choose_candidate_window(imbalance_estimates)
    common_channels = channel_consistency.get("common_channels_all_recordings", [])
    seizure_durations = [interval.duration_seconds for interval in intervals]

    if best_imbalance is None:
        recommended_window = None
        recommended_overlap = None
    else:
        recommended_window = best_imbalance.window_size_seconds
        recommended_overlap = best_imbalance.overlap_fraction

    return {
        "status": "review_required_before_freezing_config",
        "recommended_values": {
            "sampling_frequency_hz": dominant_sampling_frequency,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 40.0,
            "notch_frequency_hz": 50.0,
            "window_size_seconds": recommended_window,
            "window_overlap_fraction": recommended_overlap,
            "labeling_strategy": "overlap_any_seizure_interval",
            "channel_strategy": "common_channel_subset" if common_channels else "harmonize_or_drop_inconsistent_channels",
            "selected_channels": common_channels,
            "split_strategy": "patient_wise_holdout",
            "primary_metrics": ["recall", "sensitivity", "specificity", "f1", "roc_auc"],
        },
        "rationale": {
            "sampling_frequency": (
                "Use the dominant sampling frequency if all/most recordings match; otherwise keep frequency-aware feature extraction or resample."
            ),
            "bandpass": (
                "0.5-40 Hz preserves delta through low-gamma features while reducing drift and high-frequency noise."
            ),
            "notch": (
                "50 Hz notch is included because the requested pipeline targets 50 Hz line-noise suppression."
            ),
            "windowing": (
                "Recommendation favors a candidate with manageable class imbalance while keeping windows short enough for seizure localization."
            ),
            "labeling": (
                "Any-overlap labeling captures seizure boundary windows and matches the clinical goal of detecting seizure presence."
            ),
            "channels": (
                "A common-channel subset makes feature columns consistent for classical ML and patient-wise testing."
            ),
            "validation": (
                "Patient-wise holdout avoids leakage from correlated windows belonging to the same subject."
            ),
        },
        "review_warnings": build_review_warnings(
            annotation_coverage=annotation_coverage,
            seizure_durations=seizure_durations,
            common_channel_count=len(common_channels),
            sampling_frequency_count=len(sampling_counts),
        ),
    }


def choose_candidate_window(
    imbalance_estimates: list[ImbalanceEstimate],
) -> ImbalanceEstimate | None:
    """Choose a defensible default window candidate from class imbalance estimates."""
    valid_estimates = [
        estimate
        for estimate in imbalance_estimates
        if estimate.total_windows > 0 and estimate.seizure_windows > 0
    ]
    if not valid_estimates:
        return None

    # Prefer 4-5 seconds when available because these windows are long enough
    # for stable spectral estimates but still short for event localization. Use
    # class imbalance as the secondary criterion.
    preferred = [
        estimate
        for estimate in valid_estimates
        if 4.0 <= estimate.window_size_seconds <= 5.0
    ]
    candidates = preferred or valid_estimates
    return min(
        candidates,
        key=lambda estimate: (
            estimate.non_seizure_to_seizure_ratio
            if estimate.non_seizure_to_seizure_ratio is not None
            else float("inf"),
            estimate.window_size_seconds,
            estimate.overlap_fraction,
        ),
    )


def build_review_warnings(
    annotation_coverage: dict[str, object],
    seizure_durations: list[float],
    common_channel_count: int,
    sampling_frequency_count: int,
) -> list[str]:
    """Build warnings that should be reviewed before freezing configuration."""
    warnings: list[str] = []
    if annotation_coverage["n_annotation_recordings_not_found_as_edf"]:
        warnings.append(
            "Some parsed annotation recording IDs were not found among EDF files; inspect summary parsing and dataset layout."
        )
    if annotation_coverage["n_invalid_intervals"]:
        warnings.append("Some seizure intervals have non-positive duration and should be corrected or excluded.")
    if not seizure_durations:
        warnings.append("No seizure intervals were parsed; class imbalance and labels cannot be trusted yet.")
    if common_channel_count == 0:
        warnings.append("No channels are common across every recording; channel harmonization must be designed before feature extraction.")
    if sampling_frequency_count > 1:
        warnings.append("Multiple sampling frequencies are present; decide whether to resample before segmentation/features.")
    return warnings


def infer_design_implications(
    recordings_df,
    intervals: list[SeizureInterval],
    imbalance_estimates: list[ImbalanceEstimate],
    channel_consistency: dict[str, object],
) -> list[str]:
    """Generate concise, defensible design implications from analysis results."""
    implications: list[str] = []

    sampling_counts = recordings_df["sampling_frequency_hz"].value_counts()
    if len(sampling_counts) == 1:
        sfreq = sampling_counts.index[0]
        implications.append(
            f"All analyzed recordings use {sfreq:g} Hz sampling, so resampling is not required initially."
        )
    else:
        implications.append(
            "Sampling frequencies vary, so preprocessing should include a resampling decision or frequency-aware feature extraction."
        )

    common_channel_count = int(channel_consistency["n_common_channels_all_recordings"])
    if common_channel_count > 0:
        implications.append(
            f"There are {common_channel_count} channels common to all recordings; using a common-channel subset can improve patient-wise comparability."
        )
    else:
        implications.append(
            "No single channel set appears in every recording; channel harmonization or per-channel missing handling is required."
        )

    if intervals:
        pd = require_pandas()
        durations = pd.Series([interval.duration_seconds for interval in intervals])
        implications.append(
            f"Median seizure duration is {durations.median():.1f}s, supporting short windows that can localize seizure transitions."
        )

    if imbalance_estimates:
        best = min(
            imbalance_estimates,
            key=lambda estimate: estimate.non_seizure_to_seizure_ratio
            if estimate.non_seizure_to_seizure_ratio is not None
            else float("inf"),
        )
        implications.append(
            "Window-level labels are highly imbalanced; evaluation should report precision, recall, F1, sensitivity, specificity, and AUC rather than accuracy alone."
        )
        implications.append(
            f"Among tested settings, {best.window_size_seconds:g}s windows with {best.overlap_fraction:.0%} overlap produced the least severe non-seizure/seizure ratio."
        )

    implications.append(
        "Patient-wise splitting remains mandatory because windows from the same subject are strongly correlated."
    )
    implications.append(
        "A 0.5-40 Hz bandpass is defensible for classical EEG seizure features because the candidate spectral bands stop at low gamma and high-frequency noise is reduced."
    )
    return implications


def write_analysis_outputs(
    recordings: list[RecordingMetadata],
    intervals: list[SeizureInterval],
    imbalance_estimates: list[ImbalanceEstimate],
    report: dict[str, object],
    output_dir: Path,
) -> None:
    """Write analysis tables and summary report to disk."""
    pd = require_pandas()
    ensure_directory(output_dir)
    pd.DataFrame(asdict(recording) for recording in recordings).to_csv(
        output_dir / "recordings.csv", index=False
    )

    # Materialize duration_seconds explicitly: it is a @property on
    # SeizureInterval, so asdict() does not include it. The CSV should still
    # surface seizure duration as an explicit column.
    seizure_intervals_df = pd.DataFrame(asdict(interval) for interval in intervals)
    if not seizure_intervals_df.empty:
        seizure_intervals_df["duration_seconds"] = (
            seizure_intervals_df["seizure_end_seconds"]
            - seizure_intervals_df["seizure_start_seconds"]
        )
    seizure_intervals_df.to_csv(output_dir / "seizure_intervals.csv", index=False)

    pd.DataFrame(asdict(estimate) for estimate in imbalance_estimates).to_csv(
        output_dir / "class_imbalance_estimates.csv", index=False
    )
    save_json(report, output_dir / "dataset_summary.json")
    write_markdown_report(report, output_dir / "dataset_analysis_report.md")
    write_recommendation_report(report, output_dir / "design_recommendations.md")
    write_analysis_plots(recordings, intervals, imbalance_estimates, output_dir / "plots")


def write_markdown_report(report: dict[str, object], output_path: Path) -> None:
    """Write a human-readable Markdown analysis report."""
    overview = report["dataset_overview"]
    implications = report["design_implications"]
    imbalance = report["class_imbalance_estimates"]
    coverage = report["annotation_coverage"]

    lines = [
        "# CHB-MIT Dataset Analysis",
        "",
        "## Dataset Overview",
        f"- Patients: {overview['n_patients']}",
        f"- Recordings: {overview['n_recordings']}",
        f"- Seizure intervals: {overview['n_seizure_intervals']}",
        f"- Patients with seizures: {overview['n_patients_with_seizures']}",
        f"- Recordings with parsed seizures: {coverage['n_recordings_with_parsed_seizures']}",
        f"- Annotation recording IDs not found as EDF: {coverage['n_annotation_recordings_not_found_as_edf']}",
        f"- Invalid seizure intervals: {coverage['n_invalid_intervals']}",
        "",
        "## Recording Duration Summary Seconds",
        "```json",
        str(report["recording_duration_seconds"]),
        "```",
        "",
        "## Sampling Frequency Counts",
        "```json",
        str(report["sampling_frequency_hz_counts"]),
        "```",
        "",
        "## Channels Per Recording Counts",
        "```json",
        str(report["channels_per_recording_counts"]),
        "```",
        "",
        "## Seizure Duration Summary Seconds",
        "```json",
        str(report["seizure_duration_seconds"]),
        "```",
        "",
        "## Class Imbalance Estimates",
        "| Window s | Overlap | Total windows | Seizure windows | Seizure % | Non-seizure:seizure |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in imbalance:
        ratio = row["non_seizure_to_seizure_ratio"]
        ratio_text = "NA" if ratio is None else f"{ratio:.1f}:1"
        lines.append(
            f"| {row['window_size_seconds']} | {row['overlap_fraction']:.0%} | "
            f"{row['total_windows']} | {row['seizure_windows']} | "
            f"{row['seizure_window_percent']:.3f}% | {ratio_text} |"
        )

    lines.extend([
        "",
        "## Design Implications",
    ])
    lines.extend(f"- {item}" for item in implications)
    lines.extend([
        "",
        "## Review Outputs",
        "- `design_recommendations.md`: human-readable recommendation report for freezing `config.py`.",
        "- `dataset_summary.json`: machine-readable full analysis summary.",
        "- `plots/`: visual summaries for duration, seizure duration, sampling frequency, and class imbalance.",
    ])
    lines.append("")

    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_recommendation_report(report: dict[str, object], output_path: Path) -> None:
    """Write a review-ready recommendation report for freezing config.py."""
    recommendations = report["design_recommendations"]
    values = recommendations["recommended_values"]
    rationale = recommendations["rationale"]
    warnings = recommendations["review_warnings"]

    selected_channels = values.get("selected_channels") or []
    channel_preview = ", ".join(selected_channels[:20])
    if len(selected_channels) > 20:
        channel_preview += f", ... ({len(selected_channels)} total)"

    lines = [
        "# Design Recommendation Report",
        "",
        "This report is generated by `src.dataset_analysis` and should be reviewed before freezing `config.py`.",
        "",
        "## Recommended Config Values",
        "",
        "| Parameter | Recommended value |",
        "|---|---|",
        f"| Sampling frequency | {values['sampling_frequency_hz']} Hz |",
        f"| Bandpass low | {values['bandpass_low_hz']} Hz |",
        f"| Bandpass high | {values['bandpass_high_hz']} Hz |",
        f"| Notch frequency | {values['notch_frequency_hz']} Hz |",
        f"| Window size | {values['window_size_seconds']} seconds |",
        f"| Window overlap | {values['window_overlap_fraction']} |",
        f"| Labeling strategy | `{values['labeling_strategy']}` |",
        f"| Channel strategy | `{values['channel_strategy']}` |",
        f"| Split strategy | `{values['split_strategy']}` |",
        f"| Primary metrics | {', '.join(values['primary_metrics'])} |",
        "",
        "## Selected Common Channels",
        channel_preview if channel_preview else "No universal common channel subset found yet.",
        "",
        "## Rationale",
    ]
    lines.extend(f"- **{key.replace('_', ' ').title()}**: {value}" for key, value in rationale.items())

    lines.extend([
        "",
        "## Review Warnings",
    ])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No blocking warnings detected in the analyzed metadata.")

    lines.extend([
        "",
        "## Freeze Checklist",
        "- [ ] Confirm all expected CHB-MIT patients and EDF files were analyzed.",
        "- [ ] Confirm seizure annotations map to EDF recording IDs.",
        "- [ ] Confirm channel strategy is acceptable for classical ML feature tables.",
        "- [ ] Confirm candidate window and overlap are defensible from imbalance estimates.",
        "- [ ] Update `config.py` with frozen values only after this review.",
        "- [ ] Proceed to core pipeline modules after `config.py` is frozen.",
        "",
    ])

    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis_plots(
    recordings: list[RecordingMetadata],
    intervals: list[SeizureInterval],
    imbalance_estimates: list[ImbalanceEstimate],
    output_dir: Path,
) -> None:
    """Generate lightweight dataset analysis plots."""
    try:
        plt = require_matplotlib()
        pd = require_pandas()
    except ImportError as exc:
        LOGGER.warning("Skipping dataset plots: %s", exc)
        return

    ensure_directory(output_dir)
    recordings_df = pd.DataFrame(asdict(recording) for recording in recordings)
    imbalance_df = pd.DataFrame(asdict(estimate) for estimate in imbalance_estimates)

    plot_histogram(
        plt=plt,
        values=recordings_df["duration_seconds"],
        title="Recording Duration Distribution",
        xlabel="Duration seconds",
        output_path=output_dir / "recording_duration_distribution.png",
    )

    if intervals:
        seizure_df = pd.DataFrame(asdict(interval) for interval in intervals)

        # duration_seconds is a @property on SeizureInterval and is therefore
        # dropped by asdict(). Materialize it explicitly before any column
        # access, and validate the source columns are present first so a
        # future upstream rename fails loudly here instead of as a bare
        # KeyError deeper in plotting code.
        required = {"seizure_start_seconds", "seizure_end_seconds"}
        missing = required - set(seizure_df.columns)
        if missing:
            raise ValueError(f"Missing required seizure columns: {sorted(missing)}")

        seizure_df["duration_seconds"] = (
            seizure_df["seizure_end_seconds"] - seizure_df["seizure_start_seconds"]
        )

        plot_histogram(
            plt=plt,
            values=seizure_df["duration_seconds"],
            title="Seizure Duration Distribution",
            xlabel="Duration seconds",
            output_path=output_dir / "seizure_duration_distribution.png",
        )

    plot_bar_counts(
        plt=plt,
        counts=recordings_df["sampling_frequency_hz"].value_counts().sort_index(),
        title="Sampling Frequency Counts",
        xlabel="Sampling frequency Hz",
        ylabel="Recordings",
        output_path=output_dir / "sampling_frequency_counts.png",
    )

    if not imbalance_df.empty:
        imbalance_df = imbalance_df.copy()
        imbalance_df["setting"] = imbalance_df.apply(
            lambda row: f"{row['window_size_seconds']:g}s/{row['overlap_fraction']:.0%}",
            axis=1,
        )
        plot_bar_counts(
            plt=plt,
            counts=imbalance_df.set_index("setting")["seizure_window_percent"],
            title="Estimated Seizure Window Percentage",
            xlabel="Window / overlap",
            ylabel="Seizure windows percent",
            output_path=output_dir / "class_imbalance_by_window.png",
        )


def plot_histogram(plt, values, title: str, xlabel: str, output_path: Path) -> None:
    """Save a histogram plot."""
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bar_counts(plt, counts, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    """Save a bar chart from a pandas Series-like count object."""
    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_dataset_analysis(
    data_dir: Path,
    output_dir: Path,
    window_sizes: list[float],
    overlaps: list[float],
) -> dict[str, object]:
    """Run complete CHB-MIT dataset analysis and save outputs."""
    data_dir = validate_directory(data_dir)
    output_dir = ensure_directory(output_dir)

    recordings = load_all_recording_metadata(data_dir)
    intervals = load_all_seizure_intervals(data_dir)
    channel_consistency = compute_channel_consistency(recordings)
    annotation_coverage = compute_annotation_coverage(recordings, intervals)
    imbalance_estimates = estimate_class_imbalance(
        recordings=recordings,
        intervals=intervals,
        window_sizes=window_sizes,
        overlaps=overlaps,
    )
    report = build_summary_report(
        recordings=recordings,
        intervals=intervals,
        imbalance_estimates=imbalance_estimates,
        channel_consistency=channel_consistency,
        annotation_coverage=annotation_coverage,
    )
    write_analysis_outputs(recordings, intervals, imbalance_estimates, report, output_dir)
    LOGGER.info("Dataset analysis complete. Outputs saved to %s", output_dir)
    return report


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats from CLI arguments."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Analyze CHB-MIT EDF dataset metadata.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/chbmit"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dataset_analysis"))
    parser.add_argument("--window-sizes", type=parse_float_list, default=[2.0, 4.0, 5.0])
    parser.add_argument("--overlaps", type=parse_float_list, default=[0.0, 0.5])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_dataset_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_sizes=args.window_sizes,
        overlaps=args.overlaps,
    )


if __name__ == "__main__":
    main()
