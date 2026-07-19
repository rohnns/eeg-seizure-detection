"""Data loading utilities for CHB-MIT EDF recordings and annotations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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
    montage_audit: "RecordingMontageAudit | None" = None


@dataclass(frozen=True)
class RecordingMontageAudit:
    """Per-recording montage classification and exclusion details."""

    patient_id: str
    recording_id: str
    file_path: Path
    channel_names: tuple[str, ...]
    classification: str
    reference_channel: str | None = None
    missing_endpoints: tuple[str, ...] = ()
    missing_derivations: tuple[str, ...] = ()
    excluded: bool = False


CANONICAL_BIPOLAR_CHANNELS: tuple[str, ...] = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-0",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8-1",
)


def _parse_bipolar_endpoints(channel_name: str) -> tuple[str, str] | None:
    parts = channel_name.split("-")
    if len(parts) < 2:
        return None
    if len(parts) > 2 and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _available_referential_electrodes(channel_names: list[str], reference_channel: str) -> set[str]:
    electrodes: set[str] = set()
    suffix = f"-{reference_channel}"
    for channel in channel_names:
        if channel.endswith(suffix):
            electrodes.add(channel[: -len(suffix)])
    return electrodes


def audit_recording_montage(recording: EEGRecording) -> RecordingMontageAudit:
    """Classify one recording's montage and reconstructability."""
    channel_names = tuple(recording.channel_names)
    available = set(channel_names)
    canonical_present = all(channel in available for channel in CANONICAL_BIPOLAR_CHANNELS)
    if canonical_present:
        return RecordingMontageAudit(
            patient_id=recording.patient_id,
            recording_id=recording.recording_id,
            file_path=recording.file_path,
            channel_names=channel_names,
            classification="canonical_bipolar",
        )

    for reference_channel in ("CS2",):
        referential_electrodes = _available_referential_electrodes(recording.channel_names, reference_channel)
        if not referential_electrodes:
            continue

        missing_endpoints: set[str] = set()
        missing_derivations: list[str] = []
        for derivation in CANONICAL_BIPOLAR_CHANNELS:
            endpoints = _parse_bipolar_endpoints(derivation)
            if endpoints is None:
                missing_derivations.append(derivation)
                continue
            left, right = endpoints
            left_ref = f"{left}-{reference_channel}"
            right_ref = f"{right}-{reference_channel}"
            if left not in referential_electrodes:
                missing_endpoints.add(left)
            if right not in referential_electrodes:
                missing_endpoints.add(right)
            if left not in referential_electrodes or right not in referential_electrodes:
                missing_derivations.append(derivation)

        if missing_derivations:
            return RecordingMontageAudit(
                patient_id=recording.patient_id,
                recording_id=recording.recording_id,
                file_path=recording.file_path,
                channel_names=channel_names,
                classification="non_reconstructable_referential",
                reference_channel=reference_channel,
                missing_endpoints=tuple(sorted(missing_endpoints)),
                missing_derivations=tuple(sorted(set(missing_derivations))),
                excluded=True,
            )

        return RecordingMontageAudit(
            patient_id=recording.patient_id,
            recording_id=recording.recording_id,
            file_path=recording.file_path,
            channel_names=channel_names,
            classification="reconstructable_referential",
            reference_channel=reference_channel,
        )

    return RecordingMontageAudit(
        patient_id=recording.patient_id,
        recording_id=recording.recording_id,
        file_path=recording.file_path,
        channel_names=channel_names,
        classification="other_unknown",
        excluded=True,
    )


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
    recording = EEGRecording(
        patient_id=get_patient_id_from_path(file_path, data_dir),
        recording_id=get_recording_id_from_path(file_path),
        file_path=file_path,
        raw=raw,
        sampling_frequency=sampling_frequency,
        channel_names=list(raw.ch_names),
        duration_seconds=duration_seconds,
    )
    return EEGRecording(**{**recording.__dict__, "montage_audit": audit_recording_montage(recording)})


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


def load_dataset(data_dir: Path, preload: bool = True) -> tuple[list[EEGRecording], list[SeizureAnnotation], list[RecordingMontageAudit]]:
    """Load all EDF recordings, annotations, and montage audit results."""
    data_dir = Path(data_dir)
    edf_files = find_edf_files(data_dir)
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found below {data_dir}")
    recordings = [load_edf_file(path, data_dir, preload=preload) for path in edf_files]
    annotations = load_annotations(data_dir)
    audits = [recording.montage_audit for recording in recordings if recording.montage_audit is not None]
    return recordings, annotations, audits
