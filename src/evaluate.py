"""Model evaluation: standardized metrics, curves, and confusion matrices.

Every model in this pipeline (logistic_regression, random_forest, xgboost) is
evaluated through exactly the same code path here, which guarantees a fixed
metric schema defined in STANDARD_METRIC_KEYS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import ensure_directory, save_json

# The fixed, standardized metric schema every model reports, in report order.
# Any code that assembles a per-model results row should build it from a dict
# with exactly these keys (compute_classification_metrics guarantees this),
# so comparison tables never end up with the ragged, model-dependent column
# sets (some rows had "sensitivity"/"auc", others "balanced_accuracy"/
# "roc_auc", with NaNs in between) that the previous ad hoc reporting produced.
STANDARD_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "threshold",
)


def predict_model(model, X_test):  # noqa: ANN001
    """Return class predictions using the model's own default decision rule.

    For every classifier used in this pipeline (LogisticRegression,
    RandomForestClassifier, XGBClassifier), .predict() applies a fixed 0.5
    threshold on the predicted probability -- so metrics computed from this
    are reported at threshold=0.5, matching compute_classification_metrics's
    default.
    """
    return model.predict(X_test)


def predict_probabilities(model, X_test):  # noqa: ANN001
    """Return positive-class probabilities when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities[:, 0]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def predictions_at_threshold(y_prob, threshold: float) -> np.ndarray:  # noqa: ANN001
    """Binarize probabilities at an arbitrary decision threshold.

    Shared by any code path that needs a consistent thresholded prediction.
    >= threshold counts as the positive class.
    """
    return (np.asarray(y_prob) >= threshold).astype(int)


def compute_specificity(y_true, y_pred) -> float:  # noqa: ANN001
    """Compute true negative rate."""
    from sklearn.metrics import confusion_matrix

    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def compute_classification_metrics(y_true, y_pred, y_prob=None, threshold: float = 0.5) -> dict[str, float]:  # noqa: ANN001
    """Compute the standardized metric set for one set of predictions.

    Always returns exactly the keys in STANDARD_METRIC_KEYS, in that order --
    this is the single function both evaluate_model (default threshold) and
    run_threshold_sweep (every swept threshold) call, so every row in every
    report this pipeline produces has the same schema regardless of model or
    threshold.

    roc_auc and average_precision require probability scores and at least
    one example of each class; both are set to NaN otherwise (e.g. a
    probability-free model, or a degenerate all-one-class y_true) rather than
    raising, so a report can still be generated for the metrics that are
    well-defined.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": compute_specificity(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }

    has_both_classes = len(set(y_true)) > 1
    if y_prob is not None and has_both_classes:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return {key: metrics[key] for key in STANDARD_METRIC_KEYS}


def plot_confusion_matrix(y_true, y_pred, output_path: Path, threshold: float | None = None) -> None:  # noqa: ANN001
    """Save a confusion matrix plot.

    When `threshold` is given, it's included in the title.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    ensure_directory(Path(output_path).parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["non-seizure", "seizure"])
    display.plot(cmap="Blues", values_format="d")
    title = "Confusion Matrix"
    if threshold is not None:
        title += f" (threshold={threshold:.2f})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, output_path: Path) -> None:  # noqa: ANN001
    """Save a ROC curve plot if probabilities are available."""
    if y_prob is None or len(set(y_true)) < 2:
        return
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    ensure_directory(Path(output_path).parent)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC Curve AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, output_path: Path) -> float:  # noqa: ANN001
    """Save a Precision-Recall curve plot and return the Average Precision.

    Under the severe class imbalance in this dataset (~0.35% positive rate),
    ROC curves look deceptively strong even for models with poor real-world
    precision (see logistic_regression: ROC-AUC 0.874 but 2.3% precision at
    its default threshold) -- the PR curve and AP score make that visible in
    a way ROC-AUC alone does not, which is the whole reason this function
    exists as a first-class part of the evaluation output rather than an
    afterthought.
    """
    if y_prob is None or len(set(y_true)) < 2:
        return float("nan")
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

    ensure_directory(Path(output_path).parent)
    average_precision = float(average_precision_score(y_true, y_prob))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"Precision-Recall Curve AP={average_precision:.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return average_precision


@dataclass
class ModelEvaluation:
    """Everything produced by evaluating one model at its default threshold.

    Carries y_prob forward so callers can reuse the same probability array
    without calling predict_proba a second time.
    """

    model_name: str
    metrics: dict[str, float]
    y_prob: np.ndarray | None


def evaluate_model(model, X_test, y_test, output_dir: Path, model_name: str) -> ModelEvaluation:  # noqa: ANN001
    """Evaluate a trained model at its default threshold and save all plots/metrics.

    Writes, under output_dir:
      metrics/{model_name}_metrics.json      -- standardized schema, threshold=0.5
      plots/{model_name}_confusion_matrix.png
      plots/{model_name}_roc_curve.png
      plots/{model_name}_pr_curve.png        -- new: Precision-Recall curve + AP
    """
    output_dir = ensure_directory(output_dir)
    y_pred = predict_model(model, X_test)
    y_prob = predict_probabilities(model, X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob, threshold=0.5)

    save_metrics(metrics, output_dir / "metrics" / f"{model_name}_metrics.json")
    plot_confusion_matrix(y_test, y_pred, output_dir / "plots" / f"{model_name}_confusion_matrix.png", threshold=0.5)
    plot_roc_curve(y_test, y_prob, output_dir / "plots" / f"{model_name}_roc_curve.png")
    plot_precision_recall_curve(y_test, y_prob, output_dir / "plots" / f"{model_name}_pr_curve.png")

    return ModelEvaluation(model_name=model_name, metrics=metrics, y_prob=y_prob)


def evaluate_all_models(models: dict, X_test, y_test, output_dir: Path) -> dict[str, ModelEvaluation]:  # noqa: ANN001
    """Evaluate all trained models with identical treatment.

    Convenience entry point for callers that don't need per-model dtype
    handling (see src/train.py's prepare_features_for_model for callers,
    like the main pipeline, that do).
    """
    return {
        model_name: evaluate_model(model, X_test, y_test, output_dir, model_name)
        for model_name, model in models.items()
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON, with NaNs written as JSON null."""
    serializable = {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in metrics.items()}
    save_json(serializable, output_path)