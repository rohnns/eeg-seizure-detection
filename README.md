# EEG Seizure Detection Pipeline

Production-style EEG seizure detection pipeline for the CHB-MIT Scalp EEG Database.

## Current Phase: Dataset Analysis First

Before finalizing preprocessing and segmentation parameters, run dataset analysis to inspect:

- Number of patients and recordings
- Recording duration distribution
- Sampling frequencies
- Channel consistency across patients and recordings
- Seizure duration distribution
- Estimated class imbalance for candidate windows such as 2 s, 4 s, and 5 s
- Design implications for preprocessing, segmentation, evaluation, and modeling

## Expected Dataset Layout

Place CHB-MIT files under:

```text
data/raw/chbmit/
  chb01/
    chb01_01.edf
    chb01-summary.txt
  chb02/
    ...
```

The analyzer also supports recursive EDF and `*summary*.txt` discovery if the files are nested differently.

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run Dataset Analysis

```bash
python -m src.dataset_analysis \
  --data-dir data/raw/chbmit \
  --output-dir reports/dataset_analysis \
  --window-sizes 2,4,5 \
  --overlaps 0,0.5
```

Outputs:

```text
reports/dataset_analysis/
  recordings.csv
  seizure_intervals.csv
  class_imbalance_estimates.csv
  dataset_summary.json
  dataset_analysis_report.md
  design_recommendations.md
  plots/
    recording_duration_distribution.png
    seizure_duration_distribution.png
    sampling_frequency_counts.png
    class_imbalance_by_window.png
```

## Review Before Freezing `config.py`

After running dataset analysis, review:

1. `dataset_analysis_report.md` for dataset coverage and summary statistics.
2. `design_recommendations.md` for recommended config values and rationale.
3. `dataset_summary.json` for machine-readable details.
4. `plots/` for visual justification of segmentation and preprocessing choices.

Do not proceed to the core training pipeline until:

- All expected EDF files are discovered.
- Parsed seizure annotations map to EDF recording IDs.
- Window size and overlap are selected from class imbalance estimates.
- Channel handling strategy is selected.
- Patient-wise split strategy is confirmed.

## Why This Phase Exists

The final `config.py` values should be justified by the dataset rather than hardcoded upfront. After reviewing `dataset_analysis_report.md`, we will finalize window size, overlap, channel strategy, labeling strategy, and preprocessing settings.

## Planned Pipeline Modules

```text
src/data_loader.py
src/preprocessing.py
src/segmentation.py
src/feature_extraction.py
src/train.py
src/evaluate.py
src/explain.py
```

## Run Full Pipeline

After `config.py` is frozen and dependencies are installed:

```bash
python main.py
```

The pipeline performs:

1. EDF loading with MNE
2. CHB-MIT summary annotation parsing
3. Bandpass, notch filtering, optional resampling, and normalization
4. Overlapping window segmentation with seizure-overlap labels
5. Time-domain and frequency-domain feature extraction
6. Patient-wise train/test split
7. Logistic Regression and Random Forest training
8. Metrics, ROC curves, confusion matrices
9. SHAP explainability for Random Forest

Expected outputs:

```text
data/features/features.csv
data/features/labels.csv
data/features/metadata.csv
models/logistic_regression.pkl
models/random_forest.pkl
results/metrics/*_metrics.json
results/plots/*_confusion_matrix.png
results/plots/*_roc_curve.png
results/plots/shap_*_random_forest.png
results/predictions/local_explanation_example.json
```
