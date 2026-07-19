"""SHAP explainability for tree-based seizure predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ExplainabilityConfig
from src.utils import ensure_directory, save_json


def sample_frame(X, n_samples: int, random_seed: int):  # noqa: ANN001
    """Return a reproducible sample from a pandas DataFrame."""
    if len(X) <= n_samples:
        return X
    return X.sample(n=n_samples, random_state=random_seed)


def create_shap_explainer(model, X_background):  # noqa: ANN001
    """Create a SHAP TreeExplainer for a tree model."""
    import shap

    # XGBoost + newer SHAP versions can fail when TreeExplainer tries to parse
    # the sklearn wrapper's serialized base_score (e.g. "[5E-1]"). Using the
    # underlying booster is the smallest compatible fix and leaves the trained
    # model and evaluation pipeline unchanged.
    if hasattr(model, "get_booster"):
        try:
            return shap.TreeExplainer(model.get_booster(), data=X_background)
        except Exception:
            pass
    return shap.TreeExplainer(model, data=X_background)


def compute_shap_values(explainer, X):  # noqa: ANN001
    """Compute SHAP values for samples."""
    return explainer.shap_values(X)


def _positive_class_shap_values(shap_values):  # noqa: ANN001
    """Extract positive-class SHAP values across SHAP versions."""
    if isinstance(shap_values, list) and len(shap_values) > 1:
        return shap_values[1]
    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, 1]
        return values
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def plot_shap_summary(shap_values, X, output_path: Path) -> None:  # noqa: ANN001
    """Save a SHAP beeswarm-style summary plot."""
    import matplotlib.pyplot as plt
    import shap

    ensure_directory(Path(output_path).parent)
    values = _positive_class_shap_values(shap_values)
    shap.summary_plot(values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values, X, output_path: Path) -> None:  # noqa: ANN001
    """Save a mean absolute SHAP importance bar plot."""
    import matplotlib.pyplot as plt
    import shap

    ensure_directory(Path(output_path).parent)
    values = _positive_class_shap_values(shap_values)
    shap.summary_plot(values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_single_prediction(model, explainer, X_row, feature_names: list[str]) -> dict[str, object]:  # noqa: ANN001
    """Generate local SHAP explanation data for one prediction."""
    shap_values = compute_shap_values(explainer, X_row)
    values = _positive_class_shap_values(shap_values)[0]
    prediction = int(model.predict(X_row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_row)[0, 1])
    ranked = sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)
    top_features = [
        {"feature": feature, "shap_value": float(value)} for feature, value in ranked[:10]
    ]
    return {
        "prediction": prediction,
        "prediction_label": "seizure" if prediction == 1 else "non-seizure",
        "seizure_probability": probability,
        "top_features": top_features,
        "text_explanation": generate_text_explanation(top_features, prediction),
    }


def generate_text_explanation(top_features: list[dict[str, float]], prediction: int) -> str:
    """Create a readable explanation from top SHAP features."""
    direction = "seizure" if prediction == 1 else "non-seizure"
    positive_features = [item["feature"] for item in top_features if item["shap_value"] > 0][:3]
    negative_features = [item["feature"] for item in top_features if item["shap_value"] < 0][:3]
    if prediction == 1 and positive_features:
        return f"The model predicted seizure mainly because {', '.join(positive_features)} increased seizure evidence."
    if prediction == 0 and negative_features:
        return f"The model predicted non-seizure mainly because {', '.join(negative_features)} reduced seizure evidence."
    return f"The model predicted {direction}; inspect the listed SHAP features for the strongest local contributors."


def run_random_forest_explainability(
    model,
    X_train,
    X_test,
    output_dir: Path,
    config: ExplainabilityConfig,
) -> dict[str, object]:  # noqa: ANN001
    """Run global and local SHAP explanations for Random Forest."""
    background = sample_frame(X_train, config.shap_background_samples, config.random_seed)
    explain_sample = sample_frame(X_test, config.shap_explain_samples, config.random_seed)
    explainer = create_shap_explainer(model, background)
    shap_values = compute_shap_values(explainer, explain_sample)

    plots_dir = ensure_directory(Path(output_dir) / "plots")
    predictions_dir = ensure_directory(Path(output_dir) / "predictions")
    plot_shap_summary(shap_values, explain_sample, plots_dir / "shap_summary_random_forest.png")
    plot_shap_bar(shap_values, explain_sample, plots_dir / "shap_bar_random_forest.png")

    local = explain_single_prediction(model, explainer, explain_sample.iloc[[0]], list(X_test.columns))
    save_json(local, predictions_dir / "local_explanation_example.json")
    return local


def run_xgboost_explainability(
    model,
    X_train,
    X_test,
    output_dir: Path,
    config: ExplainabilityConfig,
) -> dict[str, object]:  # noqa: ANN001
    """Run global and local SHAP explanations for XGBoost."""
    background = sample_frame(X_train, config.shap_background_samples, config.random_seed)
    explain_sample = sample_frame(X_test, config.shap_explain_samples, config.random_seed)
    explainer = create_shap_explainer(model, background)
    shap_values = compute_shap_values(explainer, explain_sample)

    plots_dir = ensure_directory(Path(output_dir) / "plots")
    predictions_dir = ensure_directory(Path(output_dir) / "predictions")
    plot_shap_summary(shap_values, explain_sample, plots_dir / "shap_summary_xgboost.png")
    plot_shap_bar(shap_values, explain_sample, plots_dir / "shap_bar_xgboost.png")

    local = explain_single_prediction(model, explainer, explain_sample.iloc[[0]], list(X_test.columns))
    save_json(local, predictions_dir / "local_explanation_xgboost_example.json")
    return local
