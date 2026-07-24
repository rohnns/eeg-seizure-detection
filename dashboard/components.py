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


def render_pipeline_diagram(steps: list[str]) -> None:
    """Render a vertical flow diagram as SVG (replaces the text pipeline string).

    Uses ``currentColor`` for strokes/text so it adapts to Streamlit's light and
    dark themes automatically, without hardcoding a palette.
    """
    node_h, gap, node_w, pad = 46, 30, 240, 16
    svg_w = node_w + pad * 2
    svg_h = len(steps) * node_h + (len(steps) - 1) * gap + pad * 2

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;" role="img" aria-label="Pipeline flow diagram">',
        '<defs><marker id="pipeArrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="currentColor" opacity="0.6"/></marker></defs>',
    ]
    y = pad
    x = pad
    for i, step in enumerate(steps):
        cy = y + node_h / 2
        parts.append(
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="10" '
            f'fill="rgba(127,127,127,0.10)" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x + node_w / 2}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="14" font-weight="600" fill="currentColor">{step}</text>'
        )
        if i < len(steps) - 1:
            y1, y2 = y + node_h, y + node_h + gap
            parts.append(
                f'<line x1="{x + node_w / 2}" y1="{y1}" x2="{x + node_w / 2}" y2="{y2}" '
                f'stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5" marker-end="url(#pipeArrow)"/>'
            )
        y += node_h + gap
    parts.append("</svg>")

    st.markdown("".join(parts), unsafe_allow_html=True)