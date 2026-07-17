"""Time-domain and frequency-domain EEG feature extraction."""

from __future__ import annotations

import numpy as np

from config import FeatureConfig
from src.segmentation import SegmentedWindow


def compute_mean(signal: np.ndarray) -> float:
    """Return signal mean."""
    return float(np.mean(signal))


def compute_std(signal: np.ndarray) -> float:
    """Return signal standard deviation."""
    return float(np.std(signal))


def compute_rms(signal: np.ndarray) -> float:
    """Return root mean square."""
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_variance(signal: np.ndarray) -> float:
    """Return signal variance."""
    return float(np.var(signal))


def compute_skewness(signal: np.ndarray, eps: float = 1e-8) -> float:
    """Return signal skewness."""
    centered = signal - np.mean(signal)
    std = np.std(signal)
    return float(np.mean(centered**3) / max(std**3, eps))


def compute_kurtosis(signal: np.ndarray, eps: float = 1e-8) -> float:
    """Return excess kurtosis."""
    centered = signal - np.mean(signal)
    std = np.std(signal)
    return float(np.mean(centered**4) / max(std**4, eps) - 3.0)


def compute_zero_crossings(signal: np.ndarray) -> int:
    """Count zero crossings."""
    return int(np.sum(np.diff(np.signbit(signal)) != 0))


def compute_line_length(signal: np.ndarray) -> float:
    """Return EEG line length."""
    return float(np.sum(np.abs(np.diff(signal))))


def compute_hjorth_parameters(signal: np.ndarray, eps: float = 1e-8) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, and complexity."""
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(first_derivative) / max(activity, eps))
    derivative_mobility = np.sqrt(
        np.var(second_derivative) / max(np.var(first_derivative), eps)
    )
    complexity = derivative_mobility / max(mobility, eps)
    return float(activity), float(mobility), float(complexity)


def extract_time_domain_features(signal: np.ndarray) -> dict[str, float]:
    """Extract all time-domain features for one channel."""
    hjorth_activity, hjorth_mobility, hjorth_complexity = compute_hjorth_parameters(signal)
    return {
        "mean": compute_mean(signal),
        "std": compute_std(signal),
        "rms": compute_rms(signal),
        "variance": compute_variance(signal),
        "skewness": compute_skewness(signal),
        "kurtosis": compute_kurtosis(signal),
        "zero_crossings": float(compute_zero_crossings(signal)),
        "line_length": compute_line_length(signal),
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }


def compute_power_spectral_density(signal: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT power spectral density approximation."""
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_frequency)
    fft_values = np.fft.rfft(signal)
    psd = (np.abs(fft_values) ** 2) / max(signal.size, 1)
    return freqs, psd


def compute_band_power(freqs: np.ndarray, psd: np.ndarray, low_freq: float, high_freq: float) -> float:
    """Integrate PSD within a frequency band."""
    mask = (freqs >= low_freq) & (freqs < high_freq)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_relative_band_powers(band_powers: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    """Compute relative power per band."""
    total_power = sum(band_powers.values())
    return {f"relative_{band}_power": power / max(total_power, eps) for band, power in band_powers.items()}


def compute_spectral_entropy(psd: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized spectral entropy."""
    psd_sum = float(np.sum(psd))
    if psd_sum <= eps:
        return 0.0
    probabilities = psd / psd_sum
    entropy = -np.sum(probabilities * np.log2(probabilities + eps))
    return float(entropy / np.log2(len(probabilities))) if len(probabilities) > 1 else 0.0


def compute_dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Return frequency with maximum spectral power."""
    if psd.size == 0:
        return 0.0
    return float(freqs[int(np.argmax(psd))])


def extract_frequency_domain_features(
    signal: np.ndarray,
    sampling_frequency: float,
    config: FeatureConfig,
) -> dict[str, float]:
    """Extract spectral features for one channel."""
    freqs, psd = compute_power_spectral_density(signal, sampling_frequency)
    band_powers = {
        f"{band}_power": compute_band_power(freqs, psd, low, high)
        for band, (low, high) in config.frequency_bands.items()
    }
    relative = compute_relative_band_powers(
        {band.replace("_power", ""): power for band, power in band_powers.items()}
    )
    return {
        **band_powers,
        **relative,
        "spectral_entropy": compute_spectral_entropy(psd),
        "dominant_frequency": compute_dominant_frequency(freqs, psd),
    }


def sanitize_channel_name(channel_name: str) -> str:
    """Make channel names safe for tabular feature columns."""
    return channel_name.replace(" ", "_").replace("-", "_").replace(".", "_")


def extract_features_from_window(window: SegmentedWindow, config: FeatureConfig) -> dict[str, float]:
    """Extract a flat feature dictionary from one segmented window."""
    features: dict[str, float] = {}
    for channel_index, channel_name in enumerate(window.channel_names):
        prefix = sanitize_channel_name(channel_name)
        signal = window.data[channel_index]
        channel_features = {
            **extract_time_domain_features(signal),
            **extract_frequency_domain_features(signal, window.sampling_frequency, config),
        }
        for feature_name, value in channel_features.items():
            features[f"{prefix}_{feature_name}"] = float(value)
    return features


def extract_feature_matrix(windows: list[SegmentedWindow], config: FeatureConfig):
    """Convert windows to ``X``, ``y``, and metadata DataFrames/Series."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for feature matrix creation.") from exc

    feature_rows = [extract_features_from_window(window, config) for window in windows]
    metadata_rows = [
        {
            "patient_id": window.patient_id,
            "recording_id": window.recording_id,
            "start_time_seconds": window.start_time_seconds,
            "end_time_seconds": window.end_time_seconds,
        }
        for window in windows
    ]
    labels = [window.label for window in windows]
    return pd.DataFrame(feature_rows), pd.Series(labels, name="label"), pd.DataFrame(metadata_rows)
