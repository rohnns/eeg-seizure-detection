"""Reusable dashboard components."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def display_image(path: Path | None, caption: str) -> None:
    if path is None or not path.exists():
        st.info(f"Missing artifact: {caption}")
        return
    st.image(str(path), caption=caption, use_container_width=True)


def highlight_best_values(df: pd.DataFrame, metric_columns: list[str]) -> pd.io.formats.style.Styler:
    def style_max(s: pd.Series):
        is_max = s == s.max()
        return ["background-color: rgba(46, 204, 113, 0.25); font-weight: 600" if v else "" for v in is_max]

    return df.style.apply(style_max, subset=metric_columns)

