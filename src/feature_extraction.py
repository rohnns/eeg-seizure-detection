"""Time-domain and frequency-domain EEG feature extraction."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import FeatureConfig
from src.segmentation import SegmentedWindow

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib is a required dependency, guarded defensively
    Parallel = None
    delayed = None

# Below this many windows, joblib's process/thread-pool dispatch overhead is not
# worth paying; run serially instead. Tune if profiling on a given machine says otherwise.
_PARALLEL_MIN_WINDOWS = 8


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


# NOTE ON OPTIMIZATION STRATEGY:
# The public compute_* functions above are left completely untouched so any external
# code, notebooks, or tests importing them keep working exactly as before.
#
# extract_time_domain_features() below is the hot path (called 23 channels x ~7200
# windows x 686 recordings times) and previously called compute_mean, compute_std,
# compute_skewness, compute_kurtosis, and compute_hjorth_parameters independently,
# each of which redundantly recomputed the signal's mean and variance from scratch,
# and both compute_line_length and compute_hjorth_parameters independently called
# np.diff(signal). It's rewritten below to compute each shared quantity exactly once
# and reuse it, using the exact same formulas (algebraically and in floating-point
# evaluation order) as the original functions, so results are bit-identical.
def extract_time_domain_features(signal: np.ndarray) -> dict[str, float]:
    """Extract all time-domain features for one channel.

    Mathematically and numerically identical to calling compute_mean, compute_std,
    compute_rms, compute_variance, compute_skewness, compute_kurtosis,
    compute_zero_crossings, compute_line_length, and compute_hjorth_parameters
    independently -- the only change is that shared intermediates (mean, centered
    signal, variance/std, first derivative) are computed once and reused instead of
    being recomputed by each function.
    """
    eps = 1e-8

    mean = np.mean(signal)
    centered = signal - mean
    # Bit-identical to np.var(signal) / np.std(signal) -- verified in existing tests.
    variance = float(np.mean(centered**2))
    std = float(np.sqrt(variance))

    rms = float(np.sqrt(np.mean(np.square(signal))))

    skewness = float(np.mean(centered**3) / max(std**3, eps))
    kurtosis = float(np.mean(centered**4) / max(std**4, eps) - 3.0)

    zero_crossings = float(compute_zero_crossings(signal))

    # Computed once, shared between line_length and the Hjorth parameters
    # (previously each computed their own np.diff(signal) independently).
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)

    line_length = float(np.sum(np.abs(first_derivative)))

    # Identical formulas to compute_hjorth_parameters, with `activity` reusing the
    # already-computed `variance` (== np.var(signal), verified bit-identical above)
    # instead of a fresh np.var(signal) call.
    activity = variance
    var_first_derivative = float(np.var(first_derivative))
    var_second_derivative = float(np.var(second_derivative))
    mobility = float(np.sqrt(var_first_derivative / max(activity, eps)))
    derivative_mobility = float(np.sqrt(var_second_derivative / max(var_first_derivative, eps)))
    complexity = float(derivative_mobility / max(mobility, eps))

    return {
        "mean": float(mean),
        "std": std,
        "rms": rms,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zero_crossings": zero_crossings,
        "line_length": line_length,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def compute_power_spectral_density(signal: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT power spectral density approximation."""
    freqs = _cached_rfftfreq(signal.size, sampling_frequency)
    fft_values = np.fft.rfft(signal)
    psd = (np.abs(fft_values) ** 2) / max(signal.size, 1)
    return freqs, psd


@lru_cache(maxsize=64)
def _cached_rfftfreq(n: int, sampling_frequency: float) -> np.ndarray:
    """Cache the FFT frequency-bin axis.

    freqs = np.fft.rfftfreq(n, d=1/fs) depends only on window length and sampling
    frequency, both of which are constant for every window of a given recording (and
    almost always across the whole dataset). It was previously recomputed from
    scratch on every one of the ~7200 windows x 23 channels calls per recording even
    though it always produced the same array. The actual FFT (np.fft.rfft) still runs
    on every call, unchanged -- only this deterministic, signal-independent axis is
    cached.
    """
    return np.fft.rfftfreq(n, d=1.0 / sampling_frequency)


@lru_cache(maxsize=512)
def _cached_band_mask(n: int, sampling_frequency: float, low_freq: float, high_freq: float) -> np.ndarray:
    """Cache the boolean frequency-band mask used by compute_band_power.

    Like the frequency axis itself, this mask depends only on (window length,
    sampling frequency, band edges) -- all constant per recording -- yet was
    recomputed independently for every band, every channel, and every window.
    """
    freqs = _cached_rfftfreq(n, sampling_frequency)
    return (freqs >= low_freq) & (freqs < high_freq)


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
    # Uses the cached band mask (see _cached_band_mask) instead of calling
    # compute_band_power directly, which recomputed the boolean mask from freqs
    # every time. Formula (np.trapz over the masked slice) is unchanged.
    band_powers: dict[str, float] = {}
    for band, (low, high) in config.frequency_bands.items():
        mask = _cached_band_mask(signal.size, sampling_frequency, low, high)
        if not np.any(mask):
            band_powers[f"{band}_power"] = 0.0
        else:
            band_powers[f"{band}_power"] = float(np.trapz(psd[mask], freqs[mask]))
    relative = compute_relative_band_powers(
        {band.replace("_power", ""): power for band, power in band_powers.items()}
    )
    return {
        **band_powers,
        **relative,
        "spectral_entropy": compute_spectral_entropy(psd),
        "dominant_frequency": compute_dominant_frequency(freqs, psd),
    }


@lru_cache(maxsize=256)
def sanitize_channel_name(channel_name: str) -> str:
    """Make channel names safe for tabular feature columns.

    Cached because the same ~23 channel names are sanitized again for every single
    window (thousands of times per recording) but the output never changes for a
    given input string.
    """
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

    # Parallelize across windows (each window's feature extraction is independent).
    # backend="threading" is used rather than multiprocessing because:
    #   - it works identically on Windows/Linux/macOS with no pickling of
    #     SegmentedWindow/config objects and no subprocess start-up cost per recording
    #   - the hot path is dominated by NumPy array operations (mean/var/FFT/etc.),
    #     which release the GIL, so real parallelism is still achieved
    # joblib.Parallel returns results in the same order as the input generator, so
    # DataFrame row order and metadata/label alignment are preserved exactly.
    if Parallel is not None and len(windows) >= _PARALLEL_MIN_WINDOWS:
        feature_rows = Parallel(n_jobs=-1, backend="threading")(
            delayed(extract_features_from_window)(window, config) for window in windows
        )
    else:
        # Tiny workloads: skip pool dispatch overhead and run serially.
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