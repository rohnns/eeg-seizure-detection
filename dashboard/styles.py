"""Shared Streamlit dashboard styling helpers."""

from __future__ import annotations


def inject_global_styles() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 1rem; }
        .stMetric { background: rgba(127,127,127,0.08); padding: 0.75rem; border-radius: 0.75rem; }
        .dashboard-subtitle { color: rgba(128,128,128,0.95); font-size: 0.95rem; }
        .small-note { color: rgba(128,128,128,0.85); font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

