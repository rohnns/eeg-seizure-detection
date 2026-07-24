"""Streamlit dashboard entrypoint.

Visualization-only dashboard. It reads saved artifacts from ``results/`` and
never reruns feature extraction, training, evaluation, or SHAP computation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import display_image, highlight_best_values, render_callout, render_key_value_table, render_pipeline_diagram
from data_access import available_models, load_comparison_table, plot_path
from styles import inject_global_styles

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def _fmt(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


@st.cache_data(show_spinner=False)
def _comparison_df() -> pd.DataFrame:
    return load_comparison_table()


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_overview() -> None:
    st.header("Overview")
    st.markdown(
        "Detecting epileptic seizures from scalp EEG using classical, interpretable "
        "machine learning \u2014 trained on patient recordings from the CHB-MIT Scalp "
        "EEG Database and evaluated patient-wise, so no patient appears in both the "
        "training and test sets."
    )

    st.subheader("Dataset")
    st.markdown("**CHB-MIT Scalp EEG Database**")
    st.caption("Public benchmark dataset for EEG seizure detection.")
    render_key_value_table(
        [
            ("Patients", "23"),
            ("EEG Recordings", "664 EDF files"),
            ("Annotated Seizures", "198"),
            ("Sampling Rate", "256 Hz"),
            ("Resolution", "16-bit"),
            ("Channels", "23 bipolar EEG channels"),
        ]
    )
    render_callout(
        "Long-term scalp EEG recordings from pediatric epilepsy patients. Raw EEG "
        "signals are preprocessed, segmented into fixed-length windows, transformed "
        "into handcrafted features, and evaluated using patient-wise train/test "
        "splits to measure generalization to previously unseen patients."
    )

    st.divider()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Pipeline")
        render_pipeline_diagram(
            [
                "CHB-MIT EEG",
                "Preprocessing",
                "Feature Extraction",
                "Patient-wise Split",
                "Model Training",
                "Evaluation",
                "SHAP Explainability",
            ]
        )
    with right:
        st.subheader("Best Model (Highest F1)")
        comparison = _comparison_df()
        best = None
        if not comparison.empty and "f1" in comparison.columns:
            best = comparison.sort_values(["f1", "roc_auc"], ascending=[False, False]).iloc[0]
        if best is not None:
            st.metric(_label(str(best["model"])), f"F1 {_fmt(best.get('f1'))}")
            st.caption(
                f"Accuracy {_fmt(best.get('accuracy'))} | Precision {_fmt(best.get('precision'))} | "
                f"Recall {_fmt(best.get('recall'))} | ROC-AUC {_fmt(best.get('roc_auc'))}"
            )
            st.caption("Full model comparison and curves are on the Performance page.")
        else:
            st.info("Comparison table unavailable. Run evaluation to generate `results/metrics/`.")


def render_performance() -> None:
    st.header("Performance")
    df = _comparison_df()
    if df.empty:
        st.info("Performance table is unavailable. Run evaluation to generate `results/metrics/`.")
        return

    display_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "average_precision"] if c in df.columns]
    metric_cols = [c for c in display_cols if c != "model"]
    st.dataframe(highlight_best_values(df[display_cols], metric_cols), use_container_width=True, hide_index=True)
    st.caption("Highlighted cells mark the best value per metric across models.")

    st.divider()
    st.subheader("ROC and Precision-Recall Curves")
    models = [m for m in ["logistic_regression", "random_forest", "xgboost"] if m in available_models()]
    for model_name in models:
        st.markdown(f"**{_label(model_name)}**")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            display_image(plot_path(f"{model_name}_roc_curve.png"), f"{_label(model_name)} ROC curve")
        with c2:
            display_image(plot_path(f"{model_name}_pr_curve.png"), f"{_label(model_name)} PR curve")

    st.divider()
    st.subheader("Confusion Matrices")
    if models:
        cm_cols = st.columns(len(models))
        for col, model_name in zip(cm_cols, models):
            with col:
                display_image(plot_path(f"{model_name}_confusion_matrix.png"), f"{_label(model_name)} confusion matrix")


def render_explainability() -> None:
    st.header("Explainability")
    models = [m for m in ["random_forest", "xgboost"] if m in available_models()]
    if not models:
        st.info("Explainability artifacts are unavailable.")
        return

    model = st.selectbox("Model", models, format_func=_label)
    st.markdown(
        "SHAP highlights which features push predictions toward seizure or non-seizure. The summary plot shows distribution and direction, while the bar plot shows mean absolute importance."
    )
    left, right = st.columns(2, gap="large")
    with left:
        display_image(plot_path(f"shap_summary_{model}.png"), f"{_label(model)} SHAP summary")
    with right:
        display_image(plot_path(f"shap_bar_{model}.png"), f"{_label(model)} SHAP importance")


def main() -> None:
    st.set_page_config(page_title="EEG Seizure Detection Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    inject_global_styles()
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.9rem; padding-bottom: 0.75rem; max-width: 1600px; }
        h1, h2, h3, h4 { letter-spacing: -0.02em; }
        [data-testid="stMetric"] { border-radius: 0.9rem; padding: 0.8rem 0.9rem; background: rgba(127,127,127,0.08); }
        img { width: 100%; height: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Sections")
    section = st.sidebar.radio("", ["Overview", "Performance", "Explainability"], label_visibility="collapsed")

    if section == "Overview":
        render_overview()
    elif section == "Performance":
        render_performance()
    else:
        render_explainability()


if __name__ == "__main__":
    main()