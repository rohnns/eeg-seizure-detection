"""Streamlit dashboard entrypoint.

Visualization-only dashboard. It reads saved artifacts from ``results/`` and
never reruns feature extraction, training, evaluation, or SHAP computation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import display_image, highlight_best_values
from data_access import RESULTS_DIR, available_models, load_comparison_table, load_json_file, load_metrics, plot_path
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


@st.cache_data(show_spinner=False)
def _metrics(model_name: str) -> dict:
    return load_metrics(model_name)


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_overview() -> None:
    st.header("Overview")
    st.markdown("EEG Seizure Detection using Classical Machine Learning on the CHB-MIT dataset.")

    summary = _dataset_summary()
    total_windows = int(summary.get("total_windows", 250594))
    seizure_windows = int(summary.get("seizure_windows", 7900))
    non_seizure_windows = int(summary.get("non_seizure_windows", max(0, total_windows - seizure_windows)))
    prevalence = float(summary.get("seizure_prevalence", seizure_windows / total_windows if total_windows else 0.0))
    num_features = int(summary.get("num_features", 529))

    cols = st.columns(5)
    cols[0].metric("Total windows", f"{total_windows:,}")
    cols[1].metric("Seizure windows", f"{seizure_windows:,}")
    cols[2].metric("Non-seizure windows", f"{non_seizure_windows:,}")
    cols[3].metric("Seizure prevalence", f"{prevalence * 100:.3f}%")
    cols[4].metric("Features", f"{num_features:,}")

    best = None
    comparison = _comparison_df()
    if not comparison.empty and "f1" in comparison.columns:
        best = comparison.sort_values(["f1", "roc_auc"], ascending=[False, False]).iloc[0]

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Pipeline diagram")
        st.code(
        "Dataset analysis → preprocessing → segmentation → feature extraction → patient-wise split → classical models → evaluation → SHAP",
        language="text",
    )
    with right:
        st.subheader("Best model")
        if best is not None:
            st.metric(_label(str(best["model"])), f"F1 {_fmt(best.get('f1'))}")
            st.caption(
                f"Accuracy {_fmt(best.get('accuracy'))} | Precision {_fmt(best.get('precision'))} | "
                f"Recall {_fmt(best.get('recall'))} | ROC-AUC {_fmt(best.get('roc_auc'))}"
            )
        else:
            st.info("Comparison table unavailable.")

    st.subheader("Key metrics")
    xgboost_metrics = _metrics("xgboost") if "xgboost" in available_models() else {}
    if xgboost_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", _fmt(xgboost_metrics.get("roc_auc")))
        c2.metric("F1", _fmt(xgboost_metrics.get("f1")))
        c3.metric("Balanced accuracy", _fmt(xgboost_metrics.get("balanced_accuracy")))


def render_performance() -> None:
    st.header("Performance")
    df = _comparison_df()
    if df.empty:
        st.info("Performance table is unavailable.")
        return

    display_cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "average_precision"] if c in df.columns]
    metric_cols = [c for c in display_cols if c != "model"]
    st.dataframe(highlight_best_values(df[display_cols], metric_cols), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    if "f1" in df.columns:
        c1.metric("Best F1", _fmt(df["f1"].max()))
    if "roc_auc" in df.columns:
        c2.metric("Best ROC-AUC", _fmt(df["roc_auc"].max()))
    if "balanced_accuracy" in df.columns:
        c3.metric("Best Balanced accuracy", _fmt(df["balanced_accuracy"].max()))

    models = [m for m in ["logistic_regression", "random_forest", "xgboost"] if m in available_models()]
    if models:
        cards = st.columns(len(models))
        for col, model_name in zip(cards, models):
            metrics = _metrics(model_name)
            with col:
                st.markdown(f"**{_label(model_name)}**")
                st.metric("F1", _fmt(metrics.get("f1")))
                st.metric("Recall", _fmt(metrics.get("recall")))
                st.metric("ROC-AUC", _fmt(metrics.get("roc_auc")))

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("ROC curves")
        for model_name in models:
            display_image(plot_path(f"{model_name}_roc_curve.png"), f"{_label(model_name)} ROC curve")
    with right:
        st.subheader("Precision-Recall curves")
        for model_name in models:
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
