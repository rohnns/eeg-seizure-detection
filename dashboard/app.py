"""Streamlit dashboard entrypoint.

Visualization-only dashboard. It reads saved artifacts from ``results/`` and
never reruns feature extraction, training, evaluation, or SHAP computation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import display_image, highlight_best_values, render_pipeline_diagram
from data_access import RESULTS_DIR, available_models, load_comparison_table, load_json_file, plot_path
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


@st.cache_data(show_spinner=False)
def _dataset_summary() -> dict:
    return load_json_file(RESULTS_DIR / "dataset_analysis" / "dataset_summary.json")


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
    summary = _dataset_summary()
    if not summary:
        st.warning(
            "Dataset summary not yet generated. Run dataset analysis to produce "
            "`results/dataset_analysis/dataset_summary.json`."
        )
    else:
        total_windows = summary.get("total_windows")
        seizure_windows = summary.get("seizure_windows")
        num_features = summary.get("num_features")
        non_seizure_windows = None
        prevalence = None
        if total_windows is not None and seizure_windows is not None:
            non_seizure_windows = int(total_windows) - int(seizure_windows)
            prevalence = (seizure_windows / total_windows) if total_windows else None

        cols = st.columns(5)
        cols[0].metric("Total windows", f"{int(total_windows):,}" if total_windows is not None else "N/A")
        cols[1].metric("Seizure windows", f"{int(seizure_windows):,}" if seizure_windows is not None else "N/A")
        cols[2].metric(
            "Non-seizure windows",
            f"{non_seizure_windows:,}" if non_seizure_windows is not None else "N/A",
        )
        cols[3].metric(
            "Seizure prevalence",
            f"{prevalence * 100:.3f}%" if prevalence is not None else "N/A",
        )
        cols[4].metric("Features", f"{int(num_features):,}" if num_features is not None else "N/A")

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