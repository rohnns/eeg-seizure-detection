# CHB-MIT Dataset Analysis

## Dataset Overview
- Patients: 24
- Recordings: 686
- Seizure intervals: 198
- Patients with seizures: 24
- Recordings with parsed seizures: 141
- Annotation recording IDs not found as EDF: 0
- Invalid seizure intervals: 0

## Recording Duration Summary Seconds
```json
{'count': 686.0, 'mean': 5158.260932944607, 'std': 3660.2498136381296, 'min': 600.0, '25%': 3600.0, '50%': 3600.0, '75%': 3600.0, 'max': 14427.0}
```

## Sampling Frequency Counts
```json
{256.0: 686}
```

## Channels Per Recording Counts
```json
{22: 26, 23: 276, 24: 54, 25: 1, 28: 275, 29: 14, 31: 1, 38: 39}
```

## Seizure Duration Summary Seconds
```json
{'count': 198.0, 'mean': 58.64141414141414, 'std': 65.02531878637151, 'min': 6.0, '25%': 25.0, '50%': 45.5, '75%': 71.0, 'max': 752.0}
```

## Class Imbalance Estimates
| Window s | Overlap | Total windows | Seizure windows | Seizure % | Non-seizure:seizure |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0% | 1769265 | 5907 | 0.334% | 298.5:1 |
| 2.0 | 50% | 3537881 | 11809 | 0.334% | 298.6:1 |
| 4.0 | 0% | 884609 | 3056 | 0.345% | 288.5:1 |
| 4.0 | 50% | 1768579 | 6105 | 0.345% | 288.7:1 |
| 5.0 | 0% | 707683 | 2480 | 0.350% | 284.4:1 |
| 5.0 | 50% | 1414710 | 4999 | 0.353% | 282.0:1 |

## Design Implications
- All analyzed recordings use 256 Hz sampling, so resampling is not required initially.
- No single channel set appears in every recording; channel harmonization or per-channel missing handling is required.
- Median seizure duration is 45.5s, supporting short windows that can localize seizure transitions.
- Window-level labels are highly imbalanced; evaluation should report precision, recall, F1, sensitivity, specificity, and AUC rather than accuracy alone.
- Among tested settings, 5s windows with 50% overlap produced the least severe non-seizure/seizure ratio.
- Patient-wise splitting remains mandatory because windows from the same subject are strongly correlated.
- A 0.5-40 Hz bandpass is defensible for classical EEG seizure features because the candidate spectral bands stop at low gamma and high-frequency noise is reduced.

## Review Outputs
- `design_recommendations.md`: human-readable recommendation report for freezing `config.py`.
- `dataset_summary.json`: machine-readable full analysis summary.
- `plots/`: visual summaries for duration, seizure duration, sampling frequency, and class imbalance.
