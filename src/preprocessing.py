"""EEG preprocessing functions."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from config import SignalConfig

LOGGER = logging.getLogger(__name__)

def apply_bandpass_filter(raw, low_freq: float, high_freq: float):
    """Apply an in-place MNE bandpass filter and return the raw object."""
    LOGGER.debug("Applying bandpass filter %.2f-%.2f Hz", low_freq, high_freq)
    return raw.filter(l_freq=low_freq, h_freq=high_freq, verbose="ERROR")


def apply_notch_filter(raw, notch_freq: float):
    """Apply an in-place MNE notch filter and return the raw object."""
    LOGGER.debug("Applying notch filter %.2f Hz", notch_freq)
    return raw.notch_filter(freqs=[notch_freq], verbose="ERROR")


def select_channels(raw, channels: Sequence[str] | None):
    """Pick selected channels if provided, preserving order."""
    if not channels:
        return raw
    available = set(raw.ch_names)
    missing = [channel for channel in channels if channel not in available]
    if missing:
        raise ValueError(f"Requested channels missing from recording: {missing}")
    return raw.copy().pick(list(channels))


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


def preprocess_raw_recording(raw, config: SignalConfig):
    """Run the full preprocessing chain on one MNE Raw recording."""
    processed = raw.copy()
    processed.load_data()     
    processed = select_channels(processed, config.selected_channels)
    processed = maybe_resample(processed, config.target_sampling_frequency_hz)
    processed = apply_bandpass_filter(processed, config.low_freq_hz, config.high_freq_hz)
    processed = apply_notch_filter(processed, config.notch_freq_hz)
    return processed
