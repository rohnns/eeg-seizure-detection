"""EEG preprocessing functions."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from config import SignalConfig
from src.data_loader import RecordingMontageAudit

LOGGER = logging.getLogger(__name__)


def _import_mne():
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG preprocessing. Install requirements.txt.") from exc
    return mne

def apply_bandpass_filter(raw, low_freq: float, high_freq: float):
    """Apply an in-place MNE bandpass filter and return the raw object."""
    LOGGER.debug("Applying bandpass filter %.2f-%.2f Hz", low_freq, high_freq)
    return raw.filter(l_freq=low_freq, h_freq=high_freq, verbose="ERROR")


def apply_notch_filter(raw, notch_freq: float):
    """Apply an in-place MNE notch filter and return the raw object."""
    LOGGER.debug("Applying notch filter %.2f Hz", notch_freq)
    return raw.notch_filter(freqs=[notch_freq], verbose="ERROR")


def select_channels(raw, channels: Sequence[str] | None, montage_audit: RecordingMontageAudit | None = None):
    """Pick selected channels if provided, preserving order.

    The selection policy is driven by the dataset audit. Only recordings classified
    as canonical bipolar are retained here. Other classes are explicitly rejected
    so they can be reported upstream as exclusions.
    """
    if not channels:
        return raw
    if montage_audit is not None and montage_audit.excluded:
        raise ValueError(
            f"Recording {montage_audit.patient_id}/{montage_audit.recording_id} is excluded by montage audit: "
            f"{montage_audit.classification}"
        )

    if montage_audit is not None and montage_audit.classification != "canonical_bipolar":
        raise ValueError(
            f"Recording {montage_audit.patient_id}/{montage_audit.recording_id} is not canonical bipolar: "
            f"{montage_audit.classification}"
        )

    available = set(raw.ch_names)

    missing = [channel for channel in channels if channel not in available]
    if not missing:
        return raw.copy().pick(list(channels))

    raise ValueError(f"Requested channels missing from recording: {missing}")


def maybe_resample(raw, target_sampling_frequency_hz: float | None):
    """Resample a raw object if a target sampling rate is configured."""
    if target_sampling_frequency_hz is None:
        return raw
    current = float(raw.info["sfreq"])
    if np.isclose(current, target_sampling_frequency_hz):
        return raw
    LOGGER.info("Resampling from %.2f Hz to %.2f Hz", current, target_sampling_frequency_hz)
    return raw.resample(target_sampling_frequency_hz, verbose="ERROR")


def normalize_eeg_data(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize EEG data channel-wise.

    Parameters
    ----------
    data:
        Array with shape ``(n_channels, n_samples)``.
    eps:
        Small value preventing division by zero.
    """
    means = data.mean(axis=1, keepdims=True)
    stds = np.maximum(data.std(axis=1, keepdims=True), eps)

    data -= means
    data /= stds

    return data


def raw_to_numpy(raw, normalize: bool = True) -> tuple[np.ndarray, float, list[str]]:
    """Convert an MNE Raw object to NumPy data and metadata."""
    data = raw.get_data()
    if normalize:
        data = normalize_eeg_data(data)
    return data, float(raw.info["sfreq"]), list(raw.ch_names)


def preprocess_raw_recording(raw, config: SignalConfig, montage_audit: RecordingMontageAudit | None = None):
    """Run the full preprocessing chain on one MNE Raw recording."""
    processed = raw.copy()
    processed.load_data()     
    processed = select_channels(processed, config.selected_channels, montage_audit=montage_audit)
    processed = maybe_resample(processed, config.target_sampling_frequency_hz)
    processed = apply_bandpass_filter(processed, config.low_freq_hz, config.high_freq_hz)
    processed = apply_notch_filter(processed, config.notch_freq_hz)
    return processed
