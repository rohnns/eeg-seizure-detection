"""Data loading utilities for CHB-MIT EDF recordings and annotations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeizureAnnotation:
    """Seizure interval for one EDF recording."""

    patient_id: str
    recording_id: str
    seizure_start_seconds: float
    seizure_end_seconds: float

    @property
    def duration_seconds(self) -> float:
        """Return seizure duration in seconds."""
        return max(0.0, self.seizure_end_seconds - self.seizure_start_seconds)


@dataclass
class EEGRecording:
    """Container for an EDF recording and its metadata."""

    patient_id: str
    recording_id: str
    file_path: Path
    raw: object
    sampling_frequency: float
    channel_names: list[str]
    duration_seconds: float


def find_edf_files(data_dir: Path) -> list[Path]:
    """Recursively discover EDF files below ``data_dir``."""
    return sorted(Path(data_dir).rglob("*.edf"), key=lambda path: str(path).lower())


def get_patient_id_from_path(file_path: Path, data_dir: Path | None = None) -> str:
    """Infer CHB-MIT patient ID from path or filename."""
    file_path = Path(file_path)
    if data_dir is not None:
        try:
            parts = file_path.relative_to(data_dir).parts
            if len(parts) > 1:
                return parts[0].lower()
        except ValueError:
            pass
    for part in file_path.parts:
        if re.fullmatch(r"chb\d+", part, flags=re.IGNORECASE):
            return part.lower()
    match = re.match(r"(chb\d+)", file_path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def get_recording_id_from_path(file_path: Path) -> str:
    """Return EDF recording ID from filename stem."""
    return Path(file_path).stem


def load_edf_file(file_path: Path, data_dir: Path | None = None, preload: bool = True) -> EEGRecording:
    """Load one EDF file using MNE and return an ``EEGRecording``."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required to load EDF files. Install requirements.txt.") from exc

    file_path = Path(file_path)
    LOGGER.info("Loading EDF: %s", file_path)
    raw = mne.io.read_raw_edf(file_path, preload=preload, verbose="ERROR")
    sampling_frequency = float(raw.info["sfreq"])
    duration_seconds = float(raw.n_times / sampling_frequency) if sampling_frequency else 0.0
    return EEGRecording(
        patient_id=get_patient_id_from_path(file_path, data_dir),
        recording_id=get_recording_id_from_path(file_path),
        file_path=file_path,
        raw=raw,
        sampling_frequency=sampling_frequency,
        channel_names=list(raw.ch_names),
        duration_seconds=duration_seconds,
    )


def load_patient_recordings(patient_dir: Path, preload: bool = True) -> list[EEGRecording]:
    """Load all EDF recordings in one patient directory."""
    patient_dir = Path(patient_dir)
    return [load_edf_file(path, patient_dir.parent, preload) for path in sorted(patient_dir.glob("*.edf"))]


def find_summary_file(patient_dir: Path) -> Path | None:
    """Find a CHB-MIT summary annotation file inside a patient directory."""
    matches = sorted(Path(patient_dir).glob("*summary*.txt"), key=lambda path: path.name.lower())
    return matches[0] if matches else None


def parse_chbmit_summary(summary_file: Path) -> list[SeizureAnnotation]:
    """Parse seizure intervals from a CHB-MIT summary text file."""
    summary_file = Path(summary_file)
    patient_match = re.search(r"(chb\d+)", summary_file.name, flags=re.IGNORECASE)
    patient_id = patient_match.group(1).lower() if patient_match else summary_file.parent.name.lower()

    annotations: list[SeizureAnnotation] = []
    current_recording_id: str | None = None
    starts: list[float] = []

    with summary_file.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            file_match = re.search(r"File Name:\s*(\S+)", line, flags=re.IGNORECASE)
            if file_match:
                current_recording_id = Path(file_match.group(1)).stem
                starts = []
                continue

            start_match = re.search(
                r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9.]+)",
                line,
                flags=re.IGNORECASE,
            )
            if start_match:
                starts.append(float(start_match.group(1)))
                continue

            end_match = re.search(
                r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9.]+)",
                line,
                flags=re.IGNORECASE,
            )
            if end_match and current_recording_id and starts:
                start = starts.pop(0)
                end = float(end_match.group(1))
                annotations.append(SeizureAnnotation(patient_id, current_recording_id, start, end))
    return annotations


def load_annotations(data_dir: Path) -> list[SeizureAnnotation]:
    """Load all CHB-MIT seizure annotations below ``data_dir``."""
    annotations: list[SeizureAnnotation] = []
    for summary_file in sorted(Path(data_dir).rglob("*summary*.txt")):
        LOGGER.info("Parsing annotations: %s", summary_file)
        annotations.extend(parse_chbmit_summary(summary_file))
    return annotations


def get_seizure_intervals_for_recording(
    recording_id: str,
    annotations: list[SeizureAnnotation],
    patient_id: str | None = None,
) -> list[SeizureAnnotation]:
    """Return annotations matching one recording, optionally constrained by patient."""
    return [
        annotation
        for annotation in annotations
        if annotation.recording_id == recording_id
        and (patient_id is None or annotation.patient_id == patient_id)
    ]


def load_dataset(data_dir: Path, preload: bool = True) -> tuple[list[EEGRecording], list[SeizureAnnotation]]:
    """Load all EDF recordings and all seizure annotations."""
    data_dir = Path(data_dir)
    edf_files = find_edf_files(data_dir)
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found below {data_dir}")
    recordings = [load_edf_file(path, data_dir, preload=preload) for path in edf_files]
    annotations = load_annotations(data_dir)
    return recordings, annotations
