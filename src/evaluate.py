"""Model evaluation metrics and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import ensure_directory, save_json


def predict_model(model, X_test):  # noqa: ANN001
    """Return class predictions."""
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


def compute_specificity(y_true, y_pred) -> float:  # noqa: ANN001
    """Compute true negative rate."""
    from sklearn.metrics import confusion_matrix

    labels = [0, 1]
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:  # noqa: ANN001
    """Compute required binary classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": compute_specificity(y_true, y_pred),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:  # noqa: ANN001
    """Save a confusion matrix plot."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    ensure_directory(Path(output_path).parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["non-seizure", "seizure"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
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


def evaluate_model(model, X_test, y_test, output_dir: Path, model_name: str) -> dict[str, float]:  # noqa: ANN001
    """Evaluate a trained model and save metrics/plots."""
    output_dir = ensure_directory(output_dir)
    y_pred = predict_model(model, X_test)
    y_prob = predict_probabilities(model, X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    save_metrics(metrics, output_dir / "metrics" / f"{model_name}_metrics.json")
    plot_confusion_matrix(y_test, y_pred, output_dir / "plots" / f"{model_name}_confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, output_dir / "plots" / f"{model_name}_roc_curve.png")
    return metrics


def evaluate_all_models(models: dict, X_test, y_test, output_dir: Path) -> dict[str, dict[str, float]]:  # noqa: ANN001
    """Evaluate all trained models."""
    return {
        model_name: evaluate_model(model, X_test, y_test, output_dir, model_name)
        for model_name, model in models.items()
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON."""
    serializable = {key: (None if np.isnan(value) else value) for key, value in metrics.items()}
    save_json(serializable, output_path)
