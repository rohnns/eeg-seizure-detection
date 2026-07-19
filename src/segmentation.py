"""Window segmentation and labeling for EEG recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SegmentationConfig
from src.data_loader import EEGRecording, SeizureAnnotation
from src.preprocessing import preprocess_raw_recording, raw_to_numpy


@dataclass(frozen=True)
class SegmentedWindow:
    """One fixed-length EEG window with metadata and label."""

    patient_id: str
    recording_id: str
    start_time_seconds: float
    end_time_seconds: float
    data: np.ndarray
    label: int
    sampling_frequency: float
    channel_names: list[str]


def create_windows(
    data: np.ndarray,
    sampling_frequency: float,
    window_size_seconds: float,
    overlap_fraction: float,
) -> np.ndarray:
    """Create overlapping windows from ``(channels, samples)`` data."""
    window_samples = int(round(window_size_seconds * sampling_frequency))
    step_samples = int(round(window_samples * (1.0 - overlap_fraction)))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Invalid window size or overlap configuration")
    if data.shape[1] < window_samples:
        return np.empty((0, data.shape[0], window_samples))

    starts = range(0, data.shape[1] - window_samples + 1, step_samples)
    return np.stack([data[:, start : start + window_samples] for start in starts])


def compute_window_times(
    n_samples: int,
    sampling_frequency: float,
    window_size_seconds: float,
    overlap_fraction: float,
) -> list[tuple[float, float]]:
    """Compute start/end seconds for each generated window."""
    window_samples = int(round(window_size_seconds * sampling_frequency))
    step_samples = int(round(window_samples * (1.0 - overlap_fraction)))
    if n_samples < window_samples:
        return []
    times = []
    for start_sample in range(0, n_samples - window_samples + 1, step_samples):
        start = start_sample / sampling_frequency
        times.append((start, start + window_size_seconds))
    return times


def window_overlaps_seizure(
    window_start: float,
    window_end: float,
    seizure_intervals: list[SeizureAnnotation],
) -> bool:
    """Return True if a half-open window overlaps a seizure interval."""
    return any(
        window_start < interval.seizure_end_seconds
        and window_end > interval.seizure_start_seconds
        for interval in seizure_intervals
    )


def label_windows(
    window_times: list[tuple[float, float]],
    seizure_intervals: list[SeizureAnnotation],
) -> list[int]:
    """Assign seizure labels to window times using any-overlap labeling."""
    return [
        int(window_overlaps_seizure(start, end, seizure_intervals))
        for start, end in window_times
    ]


def segment_recording_from_array(
    data: np.ndarray,
    sampling_frequency: float,
    channel_names: list[str],
    patient_id: str,
    recording_id: str,
    seizure_intervals: list[SeizureAnnotation],
    config: SegmentationConfig,
) -> list[SegmentedWindow]:
    """Segment preprocessed NumPy data into labeled windows."""
    windows = create_windows(
        data=data,
        sampling_frequency=sampling_frequency,
        window_size_seconds=config.window_size_seconds,
        overlap_fraction=config.overlap_fraction,
    )
    window_times = compute_window_times(
        n_samples=data.shape[1],
        sampling_frequency=sampling_frequency,
        window_size_seconds=config.window_size_seconds,
        overlap_fraction=config.overlap_fraction,
    )
    labels = label_windows(window_times, seizure_intervals)
    return [
        SegmentedWindow(
            patient_id=patient_id,
            recording_id=recording_id,
            start_time_seconds=start,
            end_time_seconds=end,
            data=window,
            label=label,
            sampling_frequency=sampling_frequency,
            channel_names=channel_names,
        )
        for window, (start, end), label in zip(windows, window_times, labels)
    ]


def segment_recording(
    recording: EEGRecording,
    seizure_intervals: list[SeizureAnnotation],
    segmentation_config: SegmentationConfig,
    signal_config,
) -> list[SegmentedWindow]:
    """Preprocess and segment one EEG recording."""
    processed_raw = preprocess_raw_recording(recording.raw, signal_config, montage_audit=recording.montage_audit)
    data, sfreq, channel_names = raw_to_numpy(processed_raw, normalize=signal_config.normalize)
    return segment_recording_from_array(
        data=data,
        sampling_frequency=sfreq,
        channel_names=channel_names,
        patient_id=recording.patient_id,
        recording_id=recording.recording_id,
        seizure_intervals=seizure_intervals,
        config=segmentation_config,
    )


def segment_dataset(
    recordings: list[EEGRecording],
    annotations: list[SeizureAnnotation],
    segmentation_config: SegmentationConfig,
    signal_config,
) -> list[SegmentedWindow]:
    """Preprocess and segment all recordings."""
    windows: list[SegmentedWindow] = []
    for recording in recordings:
        intervals = [
            annotation
            for annotation in annotations
            if annotation.patient_id == recording.patient_id
            and annotation.recording_id == recording.recording_id
        ]
        windows.extend(segment_recording(recording, intervals, segmentation_config, signal_config))
    return windows
