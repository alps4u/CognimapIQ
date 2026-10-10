"""Streamlit rendering components: sidebar controls, KPI cards, map and charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.config import (
    ALL_MEASURES,
    COMPOSITE,
    COMPOSITE_LABEL,
    DIVERGING_RAMP,
    DOMAIN_COLORS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    MEASURE_COLORS,
    MEASURE_DESCRIPTIONS,
    MEASURE_LABELS,
    NO_DATA_COLOR,
    OUTCOME_MEASURE,
    POLICY_EVENTS,
    SENSORY_MEASURES,
    SEQUENTIAL_RAMP,
    VALUE_TYPES,
)
from src.model import ModelFit, ScenarioResult

RESIDUAL = "residual"
MAP_METRICS = {
    COMPOSITE: COMPOSITE_LABEL,
    OUTCOME_MEASURE: MEASURE_LABELS[OUTCOME_MEASURE],
    **{m: MEASURE_LABELS[m] for m in SENSORY_MEASURES},
    RESIDUAL: "Cognitive decline vs. expected (model residual)",
}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
@dataclass
class Controls:
    value_type: str
    states: list[str]
    min_adults: int
    weight_mode: str
    custom_weights: dict[str, float]
    map_metric: str


def render_sidebar(states: pd.DataFrame) -> Controls:
    sb = st.sidebar
    sb.header("Filters & controls")

    sb.subheader("Region")
    labels = dict(zip(states["state"], states["state_name"]))
    selected = sb.multiselect(
        "Health region (state)",
        options=list(labels),
        format_func=lambda s: labels[s],
        placeholder="All states",
    )
    min_adults = sb.select_slider(
        "Minimum adult population",
        options=[0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
        value=0,
        format_func=lambda v: f"{v:,}",
        help="Small counties carry wider model-based uncertainty in PLACES estimates.",
    )

    sb.subheader("Map layer")
    map_metric = sb.radio(
        "Indicator",
        options=list(MAP_METRICS),
        format_func=MAP_METRICS.get,
        label_visibility="collapsed",
    )

    sb.subheader("Composite Risk Score")
    weight_mode = sb.radio(
        "Weighting",
        ["Equal", "Data-driven", "Custom"],
        horizontal=True,
        help=(
            "Equal: each sensory domain counts the same. Data-driven: weights follow "
            "each domain's standardised association with cognitive decline in the "
            "selected year. Custom: set policy weights yourself."
        ),
    )
    custom = {m: 1.0 for m in SENSORY_MEASURES}
    if weight_mode == "Custom":
        for m in SENSORY_MEASURES:
            custom[m] = sb.slider(MEASURE_LABELS[m], 0.0, 1.0, 1.0, 0.05, key=f"w_{m}")

    sb.subheader("Data")
    value_type = sb.radio(
        "Prevalence type",
        list(VALUE_TYPES),
        format_func=VALUE_TYPES.get,
        help="Age-adjusted rates compare regions fairly despite different age structures.",
    )

    sb.divider()
    sb.caption(
        "Source: CDC PLACES county estimates (BRFSS small-area models), releases "
        "2023–2025. TEETHLOST is biennial and carried forward to the following year."
    )
    return Controls(value_type, selected, min_adults, weight_mode, custom, map_metric)


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def _fmt_pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}%"


def _delta(cur: float, prev: float | None, suffix: str = " pp") -> str | None:
    if prev is None or pd.isna(cur) or pd.isna(prev):
        return None
    d = round(cur - prev, 1) + 0.0  # avoid "-0.0"
    if d == 0:
        return None  # e.g. biennial TEETHLOST carried forward
    return f"{d:+.1f}{suffix}"


def render_kpis(stats: dict, prev: dict | None, region_label: str) -> None:
    st.markdown(f"**{region_label}**")
    c = st.columns(6)
    c[0].metric(
        "Counties", f"{stats['counties']:,}",
        help=f"{stats['adults'] / 1e6:,.1f} M adults in selection",
    )
    c[1].metric(
        "Avg. Composite Risk Score", "–" if pd.isna(stats["score_mean"]) else f"{stats['score_mean']:.0f}",
        _delta(stats["score_mean"], prev and prev["score_mean"], ""),
        delta_color="inverse",
        help="0–100 percentile of combined sensory burden, pooled across all years.",
    )
    for col, m in zip(c[2:], [OUTCOME_MEASURE, *SENSORY_MEASURES]):
        col.metric(
            MEASURE_LABELS[m], _fmt_pct(stats[m]),
            _delta(stats[m], prev and prev[m]),
            delta_color="inverse",
            help=MEASURE_DESCRIPTIONS[m] + " Adult-population-weighted.",
        )


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str, alpha: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


@st.cache_data(show_spinner=False)
def color_breaks(values: tuple[float, ...], diverging: bool) -> list[float]:
    """Class breaks computed on the pooled (all-year) distribution so colours
    mean the same thing at every step of the time slider."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []
    if diverging:
        half = (len(DIVERGING_RAMP) - 1) // 2
        q = np.quantile(np.abs(arr), np.linspace(0, 1, half + 2)[1:-1])
        q = np.maximum.accumulate(np.maximum(q, 1e-6))
        return [*(-q[::-1]), *q]
    k = len(SEQUENTIAL_RAMP)
    return list(np.quantile(arr, np.linspace(0, 1, k + 1)[1:-1]))


def _classify(v: float, breaks: list[float], ramp: list[str]) -> str:
    if pd.isna(v) or not breaks:
        return NO_DATA_COLOR
    return ramp[int(np.searchsorted(breaks, v, side="right"))]


def _fmt_metric(metric: str, v: float) -> str:
    if pd.isna(v):
        return "no data"
    if metric == COMPOSITE:
        return f"{v:.0f}"
    if metric == RESIDUAL:
        return f"{v:+.1f} pp"
    return f"{v:.1f}%"


def build_map_features(
    geojson: dict,
    year_df: pd.DataFrame,
    metric: str,
    breaks: list[float],
    selected_fips: set[str],
) -> list[dict]:
    ramp = DIVERGING_RAMP if metric == RESIDUAL else SEQUENTIAL_RAMP
    rows = year_df.set_index("fips")
    features = []
    for feat in geojson["features"]:
        fips = feat["properties"]["fips"]
        props = {"fips": fips, "name": feat["properties"]["name"]}
        if fips in rows.index:
            r = rows.loc[fips]
            in_sel = fips in selected_fips
            color = _classify(r.get(metric), breaks, ramp)
            props.update(
                name=f"{r['county']}, {r['state']}",
                fill=_hex_to_rgb(color, 235 if in_sel else 55),
                metric_txt=_fmt_metric(metric, r.get(metric)),
                score_txt=_fmt_metric(COMPOSITE, r.get(COMPOSITE)),
                tier=r.get("risk_tier") or "–",
                **{f"{m}_txt": _fmt_metric(m, r.get(m)) for m in ALL_MEASURES},
                resid_txt=_fmt_metric(RESIDUAL, r.get(RESIDUAL)),
                adults=f"{int(r['adult_population']):,}" if pd.notna(r["adult_population"]) else "–",
            )
        else:
            props.update(
                fill=_hex_to_rgb(NO_DATA_COLOR, 120),
                metric_txt="no data", score_txt="–", tier="–", resid_txt="–", adults="–",
                **{f"{m}_txt": "–" for m in ALL_MEASURES},
            )
        features.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})
    return features


def _view_state(year_df: pd.DataFrame, selected_fips: set[str], filtered: bool) -> pdk.ViewState:
    if filtered and selected_fips:
        sel = year_df[year_df["fips"].isin(selected_fips)]
        lat_min, lat_max = sel["lat"].min(), sel["lat"].max()
        lon_min, lon_max = sel["lon"].min(), sel["lon"].max()
        span = max(lat_max - lat_min, (lon_max - lon_min) * 0.6, 1.0)
        zoom = float(np.clip(np.log2(170 / span) - 0.6, 3, 9))
        return pdk.ViewState(
            latitude=(lat_min + lat_max) / 2, longitude=(lon_min + lon_max) / 2, zoom=zoom
        )
    return pdk.ViewState(latitude=38.5, longitude=-96.5, zoom=3.3)


TOOLTIP_HTML = (
    "<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.45'>"
    "<b style='font-size:13px'>{name}</b><br/>"
    "<b>{metric_label}:</b> {metric_txt}<br/>"
    "<hr style='margin:4px 0;border:0;border-top:1px solid #ddd'/>"
    "Composite Risk Score: <b>{score_txt}</b> ({tier})<br/>"
    "Cognitive decline: {COGNITION_txt} · vs expected {resid_txt}<br/>"
    "Vision: {VISION_txt} · Hearing: {HEARING_txt}<br/>"
    "Complete tooth loss (65+): {TEETHLOST_txt}<br/>"
    "Adults: {adults}</div>"
)


def render_map(
    geojson: dict,
    year_df: pd.DataFrame,
    metric: str,
    breaks: list[float],
    selected_fips: set[str],
    filtered: bool,
    key: str = "risk_map",
) -> None:
    features = build_map_features(geojson, year_df, metric, breaks, selected_fips)
    layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": features},
        id="counties",
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 140],
        line_width_min_pixels=0.4,
        auto_highlight=True,
        highlight_color=[11, 11, 11, 70],
        transitions={"getFillColor": 700},
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=_view_state(year_df, selected_fips, filtered),
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": TOOLTIP_HTML.replace("{metric_label}", MAP_METRICS[metric]),
            "style": {"backgroundColor": "white", "color": INK_PRIMARY, "border": "1px solid #ccc"},
        },
    )
    st.pydeck_chart(deck, height=560, key=key)


def render_legend(metric: str, breaks: list[float]) -> None:
    if not breaks:
        return
    diverging = metric == RESIDUAL
    ramp = DIVERGING_RAMP if diverging else SEQUENTIAL_RAMP
    fmt = (lambda v: f"{v:+.1f}") if diverging else (
        (lambda v: f"{v:.0f}") if metric == COMPOSITE else (lambda v: f"{v:.1f}%")
    )
    labels = [f"< {fmt(breaks[0])}"]
    labels += [f"{fmt(a)} – {fmt(b)}" for a, b in zip(breaks[:-1], breaks[1:])]
    labels += [f"≥ {fmt(breaks[-1])}"]
    swatches = "".join(
        f"<div style='flex:1;min-width:0'><div style='height:12px;background:{c};"
        f"border-radius:2px'></div><div style='font-size:11px;color:{INK_SECONDARY};"
        f"margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{lab}</div></div>"
        for c, lab in zip(ramp, labels)
    )
    note = (
        "Blue: lower cognitive decline than sensory burden predicts · Red: higher than predicted (pp)"
        if diverging
        else "Classes are pooled quantiles across all years, so colour shifts over time reflect real change."
    )
    st.markdown(
        f"<div style='display:flex;gap:4px;margin:4px 0 2px'>{swatches}</div>"
        f"<div style='font-size:11px;color:{INK_MUTED}'>{note} Gray: no estimate.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _base_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_SECONDARY, size=12),
        hoverlabel=dict(bgcolor="white", font_color=INK_PRIMARY),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont_color=INK_MUTED)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False, tickfont_color=INK_MUTED)
    return fig


def _year_to_date(y: int) -> str:
    return f"{int(y)}-07-01"


def trend_chart(trend: pd.DataFrame, current_year: int, title: str) -> go.Figure:
    """Prevalence by measure over time with policy / evidence milestones."""
    fig = go.Figure()
    for m in ALL_MEASURES:
        d = trend[trend["measure"] == m].sort_values("year")
        if d["value"].isna().all():
            continue
        fig.add_trace(
            go.Scatter(
                x=[_year_to_date(y) for y in d["year"]],
                y=d["value"],
                mode="lines+markers",
                name=MEASURE_LABELS[m],
                line=dict(color=MEASURE_COLORS[m], width=2),
                marker=dict(size=8, line=dict(color="white", width=2)),
                hovertemplate="%{y:.1f}%<extra>" + MEASURE_LABELS[m] + "</extra>",
            )
        )
        last = d.dropna(subset=["value"]).iloc[-1]
        fig.add_annotation(
            x=_year_to_date(last["year"]), y=last["value"], text=MEASURE_LABELS[m],
            showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=INK_SECONDARY),
        )
    years = trend["year"].unique()
    fig.add_vrect(
        x0=f"{current_year}-01-01", x1=f"{current_year}-12-31",
        fillcolor="#e1e0d9", opacity=0.35, line_width=0, layer="below",
    )
    for i, ev in enumerate(POLICY_EVENTS):
        fig.add_vline(x=ev.date, line=dict(color=DOMAIN_COLORS[ev.domain], width=1, dash="dot"))
        fig.add_annotation(
            x=ev.date, y=1.0, yref="paper", text=f"<b>{i + 1}</b>", showarrow=False,
            yanchor="bottom", font=dict(size=11, color=INK_PRIMARY),
            hovertext=f"{ev.date} · {ev.title}",
        )
    _base_layout(
        fig,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0),
        xaxis=dict(range=[f"{min(years) - 1}-06-01", "2025-01-31"], tickformat="%Y", dtick="M12"),
        yaxis=dict(title="Adult prevalence (%)", rangemode="tozero"),
    )
    return fig


def score_trend_chart(panel: pd.DataFrame, fips: set[str], title: str) -> go.Figure:
    """Distribution of the Composite Risk Score per year (median and IQR band)."""
    d = panel[panel["fips"].isin(fips)] if fips else panel
    g = d.groupby("year")[COMPOSITE].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    x = [_year_to_date(y) for y in g["year"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=g[0.25], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
            line=dict(width=0), name="Interquartile range",
            hovertemplate="25th pct: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=g[0.5], mode="lines+markers", name="Median",
            line=dict(color=MEASURE_COLORS["VISION"], width=2),
            marker=dict(size=8, line=dict(color="white", width=2)),
            hovertemplate="Median: %{y:.0f}<extra></extra>",
        )
    )
    _base_layout(
        fig, height=300,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        legend=dict(orientation="h", y=-0.18, x=0),
        xaxis=dict(tickvals=x, ticktext=[str(y) for y in g["year"]]),
        yaxis=dict(title="Score (0–100)", range=[0, 100]),
    )
    return fig


def scatter_chart(df: pd.DataFrame, fit: ModelFit | None) -> go.Figure:
    d = df.dropna(subset=[COMPOSITE, OUTCOME_MEASURE])
    fig = go.Figure(
        go.Scattergl(
            x=d[COMPOSITE], y=d[OUTCOME_MEASURE], mode="markers",
            marker=dict(size=8, color=MEASURE_COLORS["VISION"], opacity=0.55,
                        line=dict(color="white", width=1)),
            text=d["county"] + ", " + d["state"],
            hovertemplate="<b>%{text}</b><br>Composite Risk Score %{x:.0f}<br>"
                          "Cognitive decline %{y:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    if len(d) > 2:
        slope, intercept = np.polyfit(d[COMPOSITE], d[OUTCOME_MEASURE], 1)
        xs = np.array([d[COMPOSITE].min(), d[COMPOSITE].max()])
        fig.add_trace(
            go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", showlegend=False,
                       line=dict(color=INK_PRIMARY, width=2, dash="dash"), hoverinfo="skip")
        )
    r = d[[COMPOSITE, OUTCOME_MEASURE]].corr().iloc[0, 1] if len(d) > 2 else float("nan")
    _base_layout(
        fig, height=400,
        title=dict(text=f"Composite Risk Score vs. cognitive decline (r = {r:.2f}, n = {len(d):,})",
                   font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        xaxis=dict(title=COMPOSITE_LABEL, range=[0, 100]),
        yaxis=dict(title="Cognitive decline prevalence (%)"),
    )
    return fig


def coefficient_chart(fit: ModelFit) -> go.Figure:
    ms = SENSORY_MEASURES
    fig = go.Figure(
        go.Bar(
            y=[MEASURE_LABELS[m] for m in ms],
            x=[fit.std_beta[m] for m in ms],
            orientation="h",
            marker=dict(color=[MEASURE_COLORS[m] for m in ms], cornerradius=4),
            text=[f"{fit.std_beta[m]:+.2f}" for m in ms],
            textposition="outside",
            textfont=dict(color=INK_SECONDARY),
            hovertemplate="%{y}: β = %{x:.3f}<extra></extra>",
        )
    )
    _base_layout(
        fig, height=220,
        title=dict(text="Standardised association with cognitive decline (β)",
                   font=dict(size=14, color=INK_PRIMARY), x=0, y=0.97),
        xaxis=dict(zeroline=True, zerolinecolor="#c3c2b7", showgrid=True, gridcolor=GRIDLINE,
                   range=[min(0.0, *fit.std_beta.values()) * 1.3, max(0.0, *fit.std_beta.values()) * 1.3]),
        yaxis=dict(autorange="reversed", showgrid=False),
        bargap=0.45,
    )
    return fig


# --------------------------------------------------------------------------- #
# Tables & text blocks
# --------------------------------------------------------------------------- #
def render_policy_timeline() -> None:
    rows = [
        {"#": i + 1, "Date": ev.date, "Domain": ev.domain, "Milestone": ev.title, "Detail": ev.detail}
        for i, ev in enumerate(POLICY_EVENTS)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_model_table(fit: ModelFit) -> None:
    rows = []
    for m in SENSORY_MEASURES:
        rows.append({
            "Sensory domain": MEASURE_LABELS[m],
            "Coefficient (pp per 1 pp)": fit.coef[m],
            "Std. error": fit.std_err[m],
            "Standardised β": fit.std_beta[m],
            "Pearson r": fit.pearson_r[m],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["Coefficient (pp per 1 pp)", "Std. error", "Standardised β", "Pearson r"]
        },
    )


def render_scenario(result: ScenarioResult | None) -> None:
    if result is None:
        st.info("Not enough complete records in the selection to run the scenario.")
        return
    c = st.columns(3)
    c[0].metric("Expected cognitive decline (baseline)", f"{result.baseline_rate:.2f}%")
    c[1].metric(
        "Expected under scenario", f"{result.scenario_rate:.2f}%",
        f"{result.delta_pp:+.2f} pp", delta_color="inverse",
    )
    c[2].metric(
        "Adults with cognitive difficulty potentially averted",
        f"{result.adults_averted:,.0f}",
        help="Association-based estimate: change in model-expected prevalence × adult population. "
             "Indicative only — not a causal effect size.",
    )
    top = result.per_county.sort_values("adults_averted", ascending=False).head(15)
    st.markdown("**Where preventative investment reaches the most people (top 15 counties)**")
    st.dataframe(
        top.rename(columns={
            "county": "County", "state": "State", "baseline_pred": "Expected baseline (%)",
            "scenario_pred": "Expected scenario (%)", "adults_averted": "Adults averted (est.)",
        }).drop(columns="fips"),
        hide_index=True, width="stretch",
        column_config={
            "Expected baseline (%)": st.column_config.NumberColumn(format="%.2f"),
            "Expected scenario (%)": st.column_config.NumberColumn(format="%.2f"),
            "Adults averted (est.)": st.column_config.NumberColumn(format="%,.0f"),
        },
    )


def render_priority_table(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = ["county", "state", COMPOSITE, "risk_tier", "intervention_window",
            *SENSORY_MEASURES, OUTCOME_MEASURE, "predicted_cognition", RESIDUAL, "adult_population"]
    table = (
        df.dropna(subset=[COMPOSITE])
        .sort_values(COMPOSITE, ascending=False)
        .head(top_n)[cols]
        .rename(columns={
            "county": "County", "state": "State", COMPOSITE: COMPOSITE_LABEL,
            "risk_tier": "Tier", "intervention_window": "Intervention window",
            **{m: MEASURE_LABELS[m] for m in ALL_MEASURES},
            "predicted_cognition": "Expected cognitive decline",
            RESIDUAL: "Observed − expected (pp)", "adult_population": "Adults",
        })
    )
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        table, hide_index=True, width="stretch", height=520,
        column_config={
            COMPOSITE_LABEL: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Intervention window": st.column_config.CheckboxColumn(
                help="High sensory burden (score ≥ 75) while cognitive decline is still at or "
                     "below the level the model expects — a window of opportunity for "
                     "preventative care."
            ),
            **{MEASURE_LABELS[m]: pct for m in ALL_MEASURES},
            "Expected cognitive decline": pct,
            "Observed − expected (pp)": st.column_config.NumberColumn(format="%+.1f"),
            "Adults": st.column_config.NumberColumn(format="%,d"),
        },
    )
    return table
