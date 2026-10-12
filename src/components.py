"""Streamlit rendering components: sidebar controls, KPI cards, map and charts."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image

from src import data_loader as dl
from src import model
from src.config import (
    ACCESS_SOURCES,
    ALL_MEASURES,
    COGNITION_RAMP,
    COMPOSITE,
    COMPOSITE_LABEL,
    COMPOSITE_SHORT,
    COMPOSITE_COLOR,
    DOMAIN_COLORS,
    DOT_RADIUS_PX,
    DOT_SPACING_PX,
    DOT_SPACING_ZOOMS,
    FAVICON_PATH,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    MEASURE_COLORS,
    MEASURE_DESCRIPTIONS,
    MEASURE_LABELS,
    MEDICAID_BENEFITS,
    NH_CITATIONS,
    NH_MEASURES,
    NH_OTHER,
    NEUTRAL_FILL,
    NO_DATA_COLOR,
    OUTCOME_MEASURE,
    OVERLAY_STYLES,
    PLAN_BENEFITS,
    PATTERN_LINE_PX,
    PATTERN_OPACITY,
    PATTERN_SPACING_PX,
    PATTERN_TOP_DEFAULT,
    PATTERN_TOP_SHARES,
    POLICY_EVENTS,
    SENSORY_MEASURES,
    SEQUENTIAL_RAMP,
    UDS_SERVICES,
)
from src.model import ModelFit, ScenarioResult, weighted_mean

RESIDUAL = "residual"
MAP_MODES = {"sori": COMPOSITE_LABEL, "overlay": "Cognitive & sensory overlay"}
OVERLAY_HINTS = {
    "COGNITION": "blue fill",
    "VISION": "orange ╱ hatch",
    "HEARING": "teal ╲ hatch",
    "TEETHLOST": "black dots",
}


# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _icon_data_uri(size_px: int = 128) -> str:
    """App logo downsized and inlined, so the title needs no static file route."""
    buf = io.BytesIO()
    Image.open(FAVICON_PATH).resize((size_px, size_px), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_title(text: str) -> None:
    st.markdown(
        f"<h1 style='display:flex;align-items:center'>{text}"
        f"<img src='{_icon_data_uri()}' alt='' style='height:1.25em;width:1.25em;margin-left:0.3em'/></h1>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
@dataclass
class Controls:
    states: list[str]
    weight_mode: str
    custom_weights: dict[str, float]
    map_mode: str
    overlays: list[str]
    pattern_top_pct: int
    lookup_slot: st.delta_generator.DeltaGenerator


def _run_lookup() -> None:
    address = st.session_state.get("lookup_address", "").strip()
    if not address:
        st.session_state.pop("lookup", None)
        return
    try:
        match, error = dl.geocode_address(address), None
    except dl.GeocodeError as exc:
        match, error = None, str(exc)
    st.session_state["lookup"] = {"match": match, "error": error}


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

    sb.subheader("Address lookup")
    with sb.form("address_lookup", border=False):
        st.text_input(
            "US street address", key="lookup_address", placeholder="Street, City, State ZIP",
            help="Resolved to its county with the US Census Geocoder.",
        )
        st.form_submit_button("Find county", on_click=_run_lookup, width="stretch")
    lookup_slot = sb.container()

    sb.subheader("Map layer")
    map_mode = sb.radio(
        "Map view",
        options=list(MAP_MODES),
        format_func=MAP_MODES.get,
        label_visibility="collapsed",
    )
    overlays: list[str] = []
    if map_mode == "overlay":
        for m in [OUTCOME_MEASURE, *SENSORY_MEASURES]:
            if sb.checkbox(f"{MEASURE_LABELS[m]} ({OVERLAY_HINTS[m]})", value=m == OUTCOME_MEASURE, key=f"ov_{m}"):
                overlays.append(m)
    pattern_top_pct = sb.select_slider(
        "Pattern threshold (top %)",
        options=PATTERN_TOP_SHARES,
        value=PATTERN_TOP_DEFAULT,
        format_func=lambda v: f"Top {v}%",
        disabled=map_mode != "overlay",
        help="Hatching / dots mark counties in this top share of each sensory measure "
             "(pooled across years): cutoff = (100 − N)th percentile. Example: Top 25% marks counties "
             "at or above the 75th percentile, i.e. the highest quarter of counties. Applies to the overlay view.",
    )

    sb.subheader(COMPOSITE_LABEL)
    weight_mode = sb.radio(
        "Weighting",
        ["Equal", "Custom"],
        horizontal=True,
        help="SORI = weighted average of the vision, hearing and tooth-loss z-scores, then ranked 0–100. "
             "Equal: each domain counts the same. Custom: weights are rescaled to sum to 1. Example: "
             "sliders 1.0 / 0.5 / 0.5 become 50% / 25% / 25%, so vision counts twice as much as each other domain.",
    )
    custom = {m: 1.0 for m in SENSORY_MEASURES}
    if weight_mode == "Custom":
        for m in SENSORY_MEASURES:
            custom[m] = sb.slider(MEASURE_LABELS[m], 0.0, 1.0, 1.0, 0.05, key=f"w_{m}")

    sb.divider()
    sb.caption(
        "Source: CDC PLACES county estimates (BRFSS small-area models, age-adjusted), "
        "releases 2023–2025. TEETHLOST is biennial and carried forward to the following year."
    )
    return Controls(selected, weight_mode, custom, map_mode, overlays, pattern_top_pct, lookup_slot)


def lookup_pin(year_all: pd.DataFrame) -> dict | None:
    """Map pin for the looked-up county, or None when there is no successful lookup."""
    match = (st.session_state.get("lookup") or {}).get("match")
    if not match or match["lat"] is None:
        return None
    row = year_all[year_all["fips"] == match["fips"]]
    state = row["state"].iloc[0] if len(row) else None
    return {"fips": match["fips"], "state": state, "lat": match["lat"], "lon": match["lon"]}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render_lookup_result(
    slot: st.delta_generator.DeltaGenerator, year_all: pd.DataFrame, panel: pd.DataFrame,
    year: int, top_pct: int, selected_fips: set[str],
) -> None:
    """Sidebar card: where the looked-up county sits in the national distribution."""
    lookup = st.session_state.get("lookup")
    if not lookup:
        return
    with slot:
        match = lookup["match"]
        if lookup["error"]:
            st.error(f"Address lookup failed: {lookup['error']}")
        elif match is None:
            st.warning("No match found. Include street, city and state.")
        elif not (year_all["fips"] == match["fips"]).any():
            st.warning(f"{match['county_name']} (FIPS {match['fips']}) has no PLACES estimate for {year}.")
        else:
            r = year_all.loc[year_all["fips"] == match["fips"]].iloc[0]
            st.markdown(f"**{r['county']}, {r['state']}** (FIPS `{match['fips']}`)")
            st.caption(match["matched_address"])
            score = r.get(COMPOSITE)
            st.metric(
                f"{COMPOSITE_SHORT} ({year})", "–" if pd.isna(score) else f"{score:.0f}",
                r.get("risk_tier") or None, delta_color="off", delta_arrow="off",
                help="0–100 rank of combined sensory burden (weighted vision, hearing and tooth-loss "
                     "z-scores) against every county-year. Example: "
                     + ("–" if pd.isna(score) else f"{score:.0f} means a heavier burden than {score:.0f}% "
                        "of all county-years."),
            )
            for m in [OUTCOME_MEASURE, *SENSORY_MEASURES]:
                v = r.get(m)
                if pd.isna(v):
                    st.caption(f"{MEASURE_LABELS[m]}: no estimate")
                    continue
                pct = int(round((year_all[m].dropna() <= v).mean() * 100))
                flag = ""
                if m in SENSORY_MEASURES:
                    cutoff = percentile_cutoff(tuple(panel[m].astype(float)), 100 - top_pct)
                    flag = f", top {top_pct}%" if v >= cutoff else ""
                st.progress(pct / 100, text=f"{MEASURE_LABELS[m]} {v:.1f}% ({_ordinal(pct)} pct{flag})")
            st.caption(f"Percentile among US counties in {year}; higher = more prevalent.")
            if match["fips"] not in selected_fips:
                st.caption("Outside the selected region, so it is shown dimmed on the map.")
        if st.button("Clear lookup", key="clear_lookup", width="stretch"):
            st.session_state.pop("lookup", None)
            st.rerun()


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def _fmt_pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}%"


def _delta(cur: float, prev: float | None, vs: str, suffix: str = " pp") -> str | None:
    """Change label naming its reference, e.g. "+0.2 pp (vs 2022)"."""
    if prev is None or pd.isna(cur) or pd.isna(prev):
        return None
    d = round(cur - prev, 1) + 0.0  # avoid "-0.0"
    if d == 0:
        return None  # e.g. biennial TEETHLOST carried forward
    return f"{d:+.1f}{suffix} (vs {vs})"


def _vs(text: str, ref: str) -> str:
    return f"{text} (vs {ref})"


def _us_badge(container, value: float, us_value: float, higher_is_worse: bool = True) -> None:
    """Second comparison under a metric whose delta is vs the state, styled like a
    delta pill: "+0.5 pp (vs US)", red when worse, green when better."""
    if pd.isna(value) or pd.isna(us_value):
        return
    d = round(value - us_value, 1) + 0.0
    if d == 0:
        text, fg, bg = "0.0 pp (vs US)", INK_SECONDARY, "rgba(49,51,63,0.08)"
    else:
        worse = (d > 0) == higher_is_worse
        text = f"{'↑' if d > 0 else '↓'} {d:+.1f} pp (vs US)"
        fg, bg = ("#bd2d2d", "rgba(255,75,75,0.12)") if worse else ("#177a3a", "rgba(33,195,84,0.12)")
    # Pulled up under the metric's own delta pill (Streamlit adds a block gap between elements).
    container.markdown(
        f"<div style='margin-top:-0.85rem'><span style='display:inline-block;padding:0 0.45rem;"
        f"border-radius:999px;background:{bg};color:{fg};font-size:0.8rem;line-height:1.5rem'>"
        f"{text}</span></div>",
        unsafe_allow_html=True,
    )


WEIGHTED_RATE_HELP = (
    " Adult-population-weighted: Σ(county rate × adults) ÷ Σ adults. Example: counties of 100,000 adults "
    "at 10% and 300,000 adults at 6% average (10 × 100,000 + 6 × 300,000) ÷ 400,000 = 7.0%, not 8.0%."
)


def render_kpis(stats: dict, prev: dict | None, region_label: str, prev_year: int | None = None) -> None:
    st.markdown(f"**{region_label}**")
    c = st.columns(6)
    c[0].metric(
        "Counties", f"{stats['counties']:,}",
        help=f"{stats['adults'] / 1e6:,.1f} M adults in selection",
    )
    c[1].metric(
        f"Avg. {COMPOSITE_SHORT}", "–" if pd.isna(stats["score_mean"]) else f"{stats['score_mean']:.0f}",
        _delta(stats["score_mean"], prev and prev["score_mean"], str(prev_year), ""),
        delta_color="inverse",
        help=f"{COMPOSITE_LABEL}: each sensory rate (vision, hearing, tooth loss) becomes a z-score, the z-scores are "
            "averaged with the chosen weights, and the average is ranked 0–100 against every county-year. "
             "The card shows the simple average of county scores. Example: a county scoring 82 has a "
             "higher combined sensory burden than 82% of all county-years; an average of 50 is typical.",
    )
    for col, m in zip(c[2:], [OUTCOME_MEASURE, *SENSORY_MEASURES]):
        col.metric(
            MEASURE_LABELS[m], _fmt_pct(stats[m]),
            _delta(stats[m], prev and prev[m], str(prev_year)),
            delta_color="inverse",
            help=MEASURE_DESCRIPTIONS[m] + WEIGHTED_RATE_HELP,
        )


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str, alpha: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


@st.cache_data(show_spinner=False)
def color_breaks(values: tuple[float, ...], classes: int) -> list[float]:
    """Class breaks computed on the pooled (all-year) distribution so colours
    mean the same thing at every step of the time slider."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []
    return list(np.quantile(arr, np.linspace(0, 1, classes + 1)[1:-1]))


@st.cache_data(show_spinner=False)
def percentile_cutoff(values: tuple[float, ...], pct: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, pct)) if arr.size else float("nan")


def _classify(v: float, breaks: list[float], ramp: list[str]) -> str:
    if pd.isna(v) or not breaks:
        return NO_DATA_COLOR
    return ramp[int(np.searchsorted(breaks, v, side="right"))]


def _fmt_metric(metric: str, v: float) -> str:
    if pd.isna(v):
        return "no data"
    if metric == COMPOSITE:
        return f"{v:.0f}"
    return f"{v:.1f}%"


def build_map_features(
    geojson: dict,
    year_df: pd.DataFrame,
    fill_metric: str | None,
    breaks: list[float],
    ramp: list[str],
    selected_fips: set[str],
) -> list[dict]:
    """County polygons with fill colour and tooltip text. `fill_metric=None`
    draws a neutral base so pattern overlays read on their own."""
    rows = year_df.set_index("fips").to_dict("index")  # dict lookups; .loc per county is slow
    features = []
    for feat in geojson["features"]:
        fips = feat["properties"]["fips"]
        props = {"fips": fips, "name": feat["properties"]["name"]}
        r = rows.get(fips)
        if r is not None:
            in_sel = fips in selected_fips
            color = _classify(r.get(fill_metric), breaks, ramp) if fill_metric else NEUTRAL_FILL
            props.update(
                name=f"{r['county']}, {r['state']}",
                state=r["state"],
                fill=_hex_to_rgb(color, 235 if in_sel else 55),
                score_txt=_fmt_metric(COMPOSITE, r.get(COMPOSITE)),
                tier=r["risk_tier"] if isinstance(r.get("risk_tier"), str) else "–",  # NaN when no score
                **{f"{m}_txt": _fmt_metric(m, r.get(m)) for m in ALL_MEASURES},
                adults=f"{int(r['adult_population']):,}" if pd.notna(r["adult_population"]) else "–",
            )
        else:
            props.update(
                fill=_hex_to_rgb(NO_DATA_COLOR, 120),
                score_txt="–", tier="–", adults="–",
                **{f"{m}_txt": "–" for m in ALL_MEASURES},
            )
        features.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})
    return features


DECK_WORLD_PX = 512  # deck.gl world size at zoom 0
MAP_HEIGHT_PX = 560
MAP_WIDTH_PX = 1100  # approximate main-column width on a wide layout
US_VIEW = pdk.ViewState(latitude=38.5, longitude=-96.5, zoom=3.3)


def _mercator_y(lat: float) -> float:
    """Web Mercator y in deck.gl world pixels at zoom 0 (0 at the top, 512 at the bottom)."""
    phi = np.radians(np.clip(lat, -85.05, 85.05))
    return 256 * (1 - np.log(np.tan(np.pi / 4 + phi / 2)) / np.pi)


def view_state(selected_fips: set[str], filtered: bool, padding: float = 0.08) -> pdk.ViewState:
    """Fit the map to the selected states' boundaries; whole US otherwise."""
    if not (filtered and selected_fips):
        return US_VIEW
    bounds = dl.selection_bounds(tuple(sorted(selected_fips)))
    if bounds is None:
        return US_VIEW
    min_lon, min_lat, max_lon, max_lat = bounds
    dx = (max_lon - min_lon) / 360 * DECK_WORLD_PX
    y_top, y_bottom = _mercator_y(max_lat), _mercator_y(min_lat)
    dy = y_bottom - y_top
    usable_w = MAP_WIDTH_PX * (1 - 2 * padding)
    usable_h = MAP_HEIGHT_PX * (1 - 2 * padding)
    zoom = np.log2(min(usable_w / max(dx, 1e-6), usable_h / max(dy, 1e-6)))
    center_y = (y_top + y_bottom) / 2
    center_lat = float(np.degrees(2 * np.arctan(np.exp(np.pi * (1 - center_y / 256))) - np.pi / 2))
    center_lon = (min_lon + max_lon) / 2
    if center_lon > 180:
        center_lon -= 360
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=float(np.clip(zoom, 2.5, 11)))


TOOLTIP_HTML = (
    "<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.45'>"
    "<b style='font-size:13px'>{name}</b><br/>"
    + COMPOSITE_SHORT + ": <b>{score_txt}</b> ({tier})<br/>"
    "<hr style='margin:4px 0;border:0;border-top:1px solid #ddd'/>"
    "Cognitive decline: <b>{COGNITION_txt}</b><br/>"
    "Vision: {VISION_txt}<br/>Hearing: {HEARING_txt}<br/>"
    "Complete tooth loss (65+): {TEETHLOST_txt}<br/>"
    "Adults: {adults}</div>"
)


def _county_layer(features: list[dict]) -> pdk.Layer:
    return pdk.Layer(
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


def dot_spacing_px(zoom: float) -> float:
    """Screen spacing of the dot pattern: wider when zoomed out, quantised for cache reuse."""
    px = np.interp(zoom, DOT_SPACING_ZOOMS, DOT_SPACING_PX)
    return float(round(px * 2) / 2)


def _in_view(df: pd.DataFrame, view: pdk.ViewState, margin: float = 1.0) -> pd.Series:
    """Counties whose centroid lies within the initial viewport, padded by `margin`
    viewport widths on every side so a little panning still shows patterns."""
    scale = DECK_WORLD_PX * 2**view.zoom  # screen px per world unit
    half_w = MAP_WIDTH_PX / 2 * (1 + 2 * margin) / scale * 360
    cy = _mercator_y(view.latitude) / DECK_WORLD_PX
    half_h = MAP_HEIGHT_PX / 2 * (1 + 2 * margin) / scale
    lat_hi = float(np.degrees(2 * np.arctan(np.exp(np.pi * (1 - 2 * (cy - half_h)))) - np.pi / 2))
    lat_lo = float(np.degrees(2 * np.arctan(np.exp(np.pi * (1 - 2 * (cy + half_h)))) - np.pi / 2))
    dlon = (df["lon"] - view.longitude + 180) % 360 - 180
    return dlon.abs().le(half_w) & df["lat"].between(lat_lo, lat_hi)


def pattern_layers(
    year_df: pd.DataFrame,
    overlays: list[str],
    cutoffs: dict[str, float],
    selected_fips: set[str],
    view: pdk.ViewState,
) -> list[pdk.Layer]:
    """Hatch / dot layers for counties at or above each measure's cutoff.

    Pattern geometry is only built near the initial viewport: a zoomed-in view
    (e.g. after an address lookup) would otherwise generate fine-spaced patterns
    for the whole country.
    """
    spacing = dl.pattern_spacing_m(view.zoom, PATTERN_SPACING_PX)
    dot_spacing = dl.pattern_spacing_m(view.zoom, dot_spacing_px(view.zoom))
    in_sel = year_df[year_df["fips"].isin(selected_fips)]
    in_sel = in_sel[_in_view(in_sel, view)]
    layers = []
    for m in overlays:
        if m not in OVERLAY_STYLES:
            continue
        fips = tuple(sorted(in_sel.loc[in_sel[m] >= cutoffs[m], "fips"]))
        if not fips:
            continue
        style = OVERLAY_STYLES[m]
        if style["pattern"] == "dots":
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=dl.dot_points(fips, dot_spacing), id=f"pattern_{m}",
                get_position=["lon", "lat"], get_fill_color=_hex_to_rgb(style["color"]),
                get_radius=DOT_RADIUS_PX, radius_units='"pixels"', stroked=False, pickable=False,
            ))
        else:
            layers.append(pdk.Layer(
                "PathLayer", data=dl.hatch_paths(fips, spacing, style["pattern"]), id=f"pattern_{m}",
                get_path="path", get_color=_hex_to_rgb(style["color"], round(255 * PATTERN_OPACITY)),
                get_width=PATTERN_LINE_PX, width_units='"pixels"', cap_rounded=False, pickable=False,
            ))
    return layers


def _pin_layers(pin: dict | None) -> list[pdk.Layer]:
    if pin is None:
        return []
    return [pdk.Layer(
        "ScatterplotLayer", data=[pin], id="lookup_pin",
        get_position=["lon", "lat"], get_fill_color=_hex_to_rgb(INK_PRIMARY),
        get_line_color=[255, 255, 255, 255], get_radius=6, radius_units='"pixels"',
        stroked=True, get_line_width=2, line_width_units='"pixels"', pickable=False,
    )]


def _outline_layer(geojson: dict, layer_id: str, width_px: float) -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer", data=geojson, id=layer_id,
        filled=False, stroked=True, get_line_color=_hex_to_rgb(INK_PRIMARY),
        get_line_width=width_px, line_width_units='"pixels"', line_joint_rounded=True,
        pickable=False,
    )


def highlight_layers(
    year: int, selected_fips: set[str], filtered: bool, pin: dict | None,
) -> list[pdk.Layer]:
    """Black outlines marking the selected geography: chosen states and the looked-up county."""
    layers = []
    if filtered and selected_fips:
        layers.append(_outline_layer(
            dl.selection_outline(tuple(sorted(selected_fips)), year, by_state=True), "selected_states", 2,
        ))
    if pin is not None:
        layers.append(_outline_layer(
            dl.selection_outline((pin["fips"],), year, by_state=False), "selected_county", 2.5,
        ))
    return layers


# deck.gl places the tooltip at a fixed offset from the cursor, and Streamlit's
# overflow-hidden map container clips it near the edges. This re-places it
# centred above the cursor (below when there is no room) and keeps it inside the
# map, like Leaflet's auto-direction tooltips. It also keeps a double-click on a
# county from deselecting it (see the end of the script).
TOOLTIP_PLACEMENT_JS = r"""
<script>
(() => {
  if (window.__cognimapTooltip) return;
  window.__cognimapTooltip = true;
  const GAP = 14, PAD = 6;
  // deck.gl's own tooltip y can be offset inside Streamlit, so track the real cursor.
  let cx = 0, cy = 0;
  window.addEventListener("pointermove", (e) => { cx = e.clientX; cy = e.clientY; }, true);
  const place = (el) => {
    if (el.style.display === "none" || el.dataset.placed === el.style.transform) return;
    const map = el.closest('[data-testid="stDeckGlJsonChart"]');
    if (!map || !el.offsetParent) return;
    const box = map.getBoundingClientRect(), origin = el.offsetParent.getBoundingClientRect();
    const w = el.offsetWidth, h = el.offsetHeight;
    let left = cx - w / 2, top = cy - h - GAP;
    if (top < box.top + PAD) top = cy + GAP;
    left = Math.min(Math.max(left, box.left + PAD), box.right - w - PAD);
    top = Math.min(Math.max(top, box.top + PAD), box.bottom - h - PAD);
    const t = `translate(${left - origin.left}px, ${top - origin.top}px)`;
    el.dataset.placed = t;
    el.style.transform = t;
  };
  const watched = new WeakSet();
  const watch = () => document.querySelectorAll(".deck-tooltip").forEach((el) => {
    if (watched.has(el)) return;
    watched.add(el);
    new MutationObserver(() => place(el)).observe(el, { attributes: true, attributeFilter: ["style"] });
  });
  watch();
  new MutationObserver(watch).observe(document.body, { childList: true, subtree: true });

  // Map selection toggles: a second click on the same county deselects it, so a
  // double-click would cancel the click that opens the county profile before it
  // reaches Streamlit. Swallow the second press of a quick repeat at the same spot.
  let lastUp = { t: -1e9, x: 0, y: 0 };
  const onMap = (e) => e.target instanceof Element && e.target.closest('[data-testid="stDeckGlJsonChart"]');
  for (const type of ["pointerdown", "pointerup"]) {
    window.addEventListener(type, (e) => {
      if (!onMap(e)) return;
      const repeat = performance.now() - lastUp.t < 500 && Math.hypot(e.clientX - lastUp.x, e.clientY - lastUp.y) < 12;
      if (repeat) {
        e.stopImmediatePropagation();
        e.preventDefault();
      } else if (type === "pointerup") {
        lastUp = { t: performance.now(), x: e.clientX, y: e.clientY };
      }
    }, true);
  }
})();
</script>
"""


def install_tooltip_placement() -> None:
    st.html(TOOLTIP_PLACEMENT_JS, unsafe_allow_javascript=True)


def open_profile_on_click(map_key: str, tab_key: str, tab_label: str) -> None:
    """`on_select` callback for the risk map: open the clicked county in the profile tab.

    The map key carries a nonce that is bumped here, so the next render starts
    without a selection (Streamlit would otherwise keep the clicked county
    selected and dim the rest of the map)."""
    state = st.session_state.get(map_key) or {}
    objects = (state.get("selection") or {}).get("objects") or {}
    picked = (objects.get("counties") or [None])[0]
    props = (picked or {}).get("properties") or {}
    if not props.get("state"):  # gray counties have no estimate to profile
        return
    st.session_state.update(
        profile_fips=props["fips"], profile_state=props["state"], **{tab_key: tab_label},
        _map_nonce=st.session_state.get("_map_nonce", 0) + 1,
    )


class CompactDeck(pdk.Deck):
    """pydeck pretty-prints its JSON (indent=2), which is ~70% whitespace for the county
    map. Every year-slider step resends the whole spec, so send it compact."""

    def to_json(self) -> str:
        return json.dumps(json.loads(super().to_json()), separators=(",", ":"))


def render_map(
    layers: list[pdk.Layer], view: pdk.ViewState, pin: dict | None = None,
    highlight: list[pdk.Layer] = (), key: str = "risk_map", on_click=None,
) -> None:
    deck = CompactDeck(
        layers=[*layers, *highlight, *_pin_layers(pin)],
        initial_view_state=view,
        # Clicks open the county profile, so double-click zoom is off (wheel and +/- still zoom).
        views=[pdk.View(type="MapView", controller={"doubleClickZoom": False})],
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": TOOLTIP_HTML,
            "style": {"backgroundColor": "white", "color": INK_PRIMARY, "border": "1px solid #ccc",
                      "borderRadius": "4px", "boxShadow": "0 1px 4px rgba(0,0,0,.2)",
                      "maxWidth": "280px"},
        },
    )
    if on_click is None:
        st.pydeck_chart(deck, height=MAP_HEIGHT_PX, key=key)
    else:
        st.pydeck_chart(deck, height=MAP_HEIGHT_PX, key=key, on_select=on_click,
                        selection_mode="single-object")


def render_sori_map(
    geojson: dict, year_df: pd.DataFrame, panel: pd.DataFrame,
    selected_fips: set[str], view: pdk.ViewState, pin: dict | None = None,
    highlight: list[pdk.Layer] = (), **map_kw,
) -> None:
    breaks = color_breaks(tuple(panel[COMPOSITE].astype(float)), len(SEQUENTIAL_RAMP))
    features = build_map_features(geojson, year_df, COMPOSITE, breaks, SEQUENTIAL_RAMP, selected_fips)
    render_map([_county_layer(features)], view, pin, highlight, **map_kw)
    _ramp_legend(SEQUENTIAL_RAMP, breaks, lambda v: f"{v:.0f}", COMPOSITE_LABEL)
    st.markdown(
        f"<div style='font-size:11px;color:{INK_MUTED}'>Classes are pooled quantiles across all "
        "years, so colour shifts over time reflect real change. Gray: no estimate.</div>",
        unsafe_allow_html=True,
    )


def render_overlay_map(
    geojson: dict, year_df: pd.DataFrame, panel: pd.DataFrame,
    selected_fips: set[str], view: pdk.ViewState, overlays: list[str],
    top_pct: int = PATTERN_TOP_DEFAULT, pin: dict | None = None,
    highlight: list[pdk.Layer] = (), **map_kw,
) -> None:
    show_cog = OUTCOME_MEASURE in overlays
    breaks = (
        color_breaks(tuple(panel[OUTCOME_MEASURE].astype(float)), len(COGNITION_RAMP))
        if show_cog else []
    )
    features = build_map_features(
        geojson, year_df, OUTCOME_MEASURE if show_cog else None, breaks, COGNITION_RAMP, selected_fips
    )
    cutoffs = {
        m: percentile_cutoff(tuple(panel[m].astype(float)), 100 - top_pct) for m in SENSORY_MEASURES
    }
    layers = [_county_layer(features), *pattern_layers(year_df, overlays, cutoffs, selected_fips, view)]
    render_map(layers, view, pin, highlight, **map_kw)

    if show_cog:
        _ramp_legend(COGNITION_RAMP, breaks, lambda v: f"{v:.1f}%",
                     f"{MEASURE_LABELS[OUTCOME_MEASURE]} (blue fill)")
    items = [
        "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:18px'>"
        f"{_pattern_swatch(m)}<span>{MEASURE_LABELS[m]} ≥ {cutoffs[m]:.1f}%</span></span>"
        for m in SENSORY_MEASURES if m in overlays
    ]
    if items:
        st.markdown(
            f"<div style='font-size:12px;color:{INK_SECONDARY};margin:8px 0 2px'>{''.join(items)}</div>"
            f"<div style='font-size:11px;color:{INK_MUTED}'>Patterns mark counties in the top "
            f"{top_pct}% of each measure (pooled across years). Gray: no estimate.</div>",
            unsafe_allow_html=True,
        )


def _pattern_swatch(measure: str, width: int = 26, height: int = 14) -> str:
    style = OVERLAY_STYLES[measure]
    c = style["color"]
    if style["pattern"] == "dots":
        body = "".join(
            f"<circle cx='{x + (2 if row % 2 else 0)}' cy='{y}' r='1.1' fill='{c}'/>"
            for row, y in enumerate(range(2, height, 4)) for x in range(2, width, 4)
        )
    else:
        rising = style["pattern"] == "ne"
        step = round(PATTERN_SPACING_PX * 2 ** 0.5)  # horizontal step for 45° lines
        body = "".join(
            f"<line x1='{x}' y1='{height if rising else 0}' x2='{x + height}' "
            f"y2='{0 if rising else height}' stroke='{c}' stroke-width='{PATTERN_LINE_PX}' "
            f"stroke-opacity='{PATTERN_OPACITY}'/>"
            for x in range(-height, width + 1, step)
        )
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "style='border:1px solid #999;background:#fff;overflow:hidden;vertical-align:middle'>"
        f"{body}</svg>"
    )


def _ramp_legend(ramp: list[str], breaks: list[float], fmt, title: str) -> None:
    if not breaks:
        return
    labels = [f"< {fmt(breaks[0])}"]
    labels += [f"{fmt(a)} – {fmt(b)}" for a, b in zip(breaks[:-1], breaks[1:])]
    labels += [f"≥ {fmt(breaks[-1])}"]
    swatches = "".join(
        f"<div style='flex:1;min-width:0'><div style='height:12px;background:{c};"
        f"border-radius:2px'></div><div style='font-size:11px;color:{INK_SECONDARY};"
        f"margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{lab}</div></div>"
        for c, lab in zip(ramp, labels)
    )
    st.markdown(
        f"<div style='font-size:12px;color:{INK_SECONDARY};margin-top:4px'>{title}</div>"
        f"<div style='display:flex;gap:4px;margin:2px 0 2px'>{swatches}</div>",
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


def _spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Positions for labels at heights `ys` (sorted ascending) at least `gap` apart,
    each run of colliding labels centred on its members' mean and kept in [lo, hi]."""
    clusters = [[y] for y in ys]  # each cluster: original heights of its labels
    def place(c):
        start = sum(c) / len(c) - (len(c) - 1) * gap / 2
        start = min(max(start, lo), hi - (len(c) - 1) * gap)
        return [start + i * gap for i in range(len(c))]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters) - 1):
            if place(clusters[i])[-1] + gap > place(clusters[i + 1])[0]:
                clusters[i:i + 2] = [clusters[i] + clusters[i + 1]]
                merged = True
                break
    return [y for c in clusters for y in place(c)]


LABEL_MARGIN_PX = 120  # right margin that holds line-end labels in narrow charts


def _add_end_labels(
    fig: go.Figure, x: str, labels: list[tuple[float, str, str]], y_top: float, plot_px: float,
    gap_px: float = 15,
) -> None:
    """Line-end labels (y, text, colour) in one column at `x` (the last year), spread
    vertically so they never overlap. Lines ending earlier are labelled in the same
    column, so labels never collide sideways either. The y axis must run from 0 to
    `y_top`; `plot_px` is the approximate plot-area height used to turn the pixel gap
    into data units. Labels take their line's colour so a nudged label still reads
    as its line's."""
    gap = y_top * gap_px / plot_px
    labels = sorted(labels, key=lambda lab: lab[0])
    positions = _spread([lab[0] for lab in labels], gap, gap / 2, y_top - gap / 2)
    for (_, text, color), y in zip(labels, positions):
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False, xanchor="left", yanchor="middle", xshift=8,
            font=dict(size=11, color=color),
        )


def trend_chart(trend: pd.DataFrame, current_year: int, title: str) -> go.Figure:
    """Prevalence by measure over time with policy / evidence milestones."""
    fig = go.Figure()
    labels = []
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
        labels.append((float(last["value"]), MEASURE_LABELS[m], MEASURE_COLORS[m]))
    y_top = max(trend["value"].max() * 1.12, 1)
    _add_end_labels(fig, _year_to_date(trend["year"].max()), labels, y_top, plot_px=250)
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
            hovertext=f"{ev.date}: {ev.title}",
        )
    _base_layout(
        fig,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0),
        xaxis=dict(range=[f"{min(years) - 1}-06-01", "2025-01-31"], tickformat="%Y", dtick="M12"),
        yaxis=dict(title="Adult prevalence (%)", range=[0, y_top]),
    )
    fig.update_layout(margin=dict(r=LABEL_MARGIN_PX))  # line-end labels may run past the plot area
    return fig


# UDS service lines: dental in the tooth-loss hue, vision in the vision hue; line
# style tells services of the same domain apart.
UDS_SERVICE_STYLES = {
    "dental_patients": (MEASURE_COLORS["TEETHLOST"], "solid"),
    "dental_visits": (MEASURE_COLORS["TEETHLOST"], "dash"),
    "oral_exam_patients": (MEASURE_COLORS["TEETHLOST"], "dot"),
    "sealant_patients": (MEASURE_COLORS["TEETHLOST"], "dashdot"),
    "fluoride_patients": (MEASURE_COLORS["TEETHLOST"], "longdash"),
    "vision_patients": (MEASURE_COLORS["VISION"], "solid"),
    "vision_visits": (MEASURE_COLORS["VISION"], "dash"),
    "eye_exam_patients": (MEASURE_COLORS["VISION"], "dot"),
}


def uds_county_chart(totals: pd.DataFrame, reporting: pd.DataFrame, n_orgs: int, title: str) -> go.Figure:
    """Estimated county totals of UDS services by year (rows: services, columns:
    years), drawn like the prevalence trend chart. Lines join only years where
    every organisation in the county published the count; years with gaps are
    open markers, since their totals are understated and not comparable."""
    fig = go.Figure()
    years = [int(y) for y in totals.columns]
    x = [_year_to_date(y) for y in years]
    labels = []
    for service, row in totals.iterrows():
        if row.isna().all():
            continue
        color, dash = UDS_SERVICE_STYLES[service]
        label = UDS_SERVICES[service]
        values = [None if pd.isna(v) else float(v) for v in row]
        counts = [int(c) for c in reporting.loc[service]]
        complete = [v if c == n_orgs else None for v, c in zip(values, counts)]
        partial = [v if c < n_orgs else None for v, c in zip(values, counts)]
        fig.add_trace(go.Scatter(
            x=x, y=complete, mode="lines+markers", name=label, legendgroup=label,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8, line=dict(color="white", width=2)),
            hovertemplate="%{y:,.0f}<extra>" + label + "</extra>",
        ))
        if any(v is not None for v in partial):
            fig.add_trace(go.Scatter(
                x=x, y=partial, mode="markers", name=label, legendgroup=label, showlegend=False,
                marker=dict(size=8, color="white", line=dict(color=color, width=2)),
                customdata=[f"{c} of {n_orgs}" for c in counts],
                hovertemplate="%{y:,.0f} (only %{customdata} orgs published)<extra>" + label + "</extra>",
            ))
        last = max(i for i, v in enumerate(values) if v is not None)
        labels.append((values[last], label, color))
    y_top = max(float(totals.max().max()) * 1.12, 1) if totals.notna().any().any() else 1
    # Sits in a narrow column: the coloured line-end labels name the services, so no
    # legend (eight entries would crowd out the plot); hover still shows each name.
    _add_end_labels(fig, x[-1], labels, y_top, plot_px=300)
    _base_layout(
        fig, height=380, showlegend=False,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        hovermode="x unified",
        # ticks only on data years, short form so they fit; labels sit in the right margin
        xaxis=dict(range=[f"{min(years) - 1}-06-01", f"{max(years)}-12-31"],
                   tickvals=x, ticktext=[f"'{y % 100:02d}" for y in years], tickangle=0),
        yaxis=dict(title="Patients / visits (est.)", range=[0, y_top]),
    )
    fig.update_layout(margin=dict(r=100))
    return fig


def score_trend_chart(panel: pd.DataFrame, fips: set[str], title: str) -> go.Figure:
    """Distribution of the SORI per year (median and IQR band)."""
    d = panel[panel["fips"].isin(fips)] if fips else panel
    g = d.groupby("year")[COMPOSITE].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    x = [_year_to_date(y) for y in g["year"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=g[0.25], fill="tonexty", fillcolor=f"rgba({','.join(map(str, _hex_to_rgb(COMPOSITE_COLOR)[:3]))},0.15)",
            line=dict(width=0), name="Interquartile range",
            hovertemplate="25th pct: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=g[0.5], mode="lines+markers", name="Median",
            line=dict(color=COMPOSITE_COLOR, width=2),
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
            marker=dict(size=8, color=COMPOSITE_COLOR, opacity=0.55,
                        line=dict(color="white", width=1)),
            text=d["county"] + ", " + d["state"],
            hovertemplate="<b>%{text}</b><br>" + COMPOSITE_SHORT + " %{x:.0f}<br>"
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
        title=dict(text=f"{COMPOSITE_LABEL} vs. cognitive decline (r = {r:.2f}, n = {len(d):,})",
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
        _vs(f"{result.delta_pp:+.2f} pp", "baseline"), delta_color="inverse",
    )
    c[2].metric(
        "Adults with cognitive difficulty potentially averted",
        f"{result.adults_averted:,.0f}",
        help="Association-based estimate: Σ (drop in model-expected prevalence × adult population) over "
             "counties. Example: a 0.40 pp drop in a county of 500,000 adults ≈ 0.004 × 500,000 = 2,000 adults. "
             "Indicative only, not a causal effect size.",
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
    cols = ["county", "state", COMPOSITE, "risk_tier",
            *SENSORY_MEASURES, OUTCOME_MEASURE, "predicted_cognition", RESIDUAL, "adult_population"]
    table = (
        model.priority_ranking(df)
        .head(top_n)[cols]
        .rename(columns={
            "county": "County", "state": "State", COMPOSITE: COMPOSITE_LABEL,
            "risk_tier": "Tier",
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
            **{MEASURE_LABELS[m]: pct for m in ALL_MEASURES},
            "Expected cognitive decline": pct,
            "Observed − expected (pp)": st.column_config.NumberColumn(format="%+.1f"),
            "Adults": st.column_config.NumberColumn(format="%,d"),
        },
    )
    return table


# --------------------------------------------------------------------------- #
# County profile
# --------------------------------------------------------------------------- #
@dataclass
class NursingHomes:
    facilities: pd.DataFrame
    by_fips: pd.DataFrame   # model.nh_summaries
    by_state: pd.DataFrame
    us: dict


@dataclass
class CountyAccess:
    medicaid: pd.DataFrame
    plans: pd.DataFrame
    plan_counties: pd.DataFrame
    plan_shares: pd.DataFrame
    sites: pd.DataFrame
    uds: pd.DataFrame
    nursing_homes: NursingHomes | None = None


def _show_county(fips: str, state: str) -> None:
    st.session_state.update(profile_fips=fips, profile_state=state)


def _county_name(year_all: pd.DataFrame, fips: str) -> str:
    r = year_all.loc[year_all["fips"] == fips].iloc[0]
    return f"{r['county']}, {r['state']}"


def render_county_picker(year_all: pd.DataFrame, pin: dict | None) -> str:
    """State + county selectors for the profile tab (session keys `profile_state` /
    `profile_fips`, seeded by the caller with the alphabetically first county),
    plus a shortcut to the address-lookup county."""
    states = year_all[["state", "state_name"]].drop_duplicates().sort_values("state_name")
    names = dict(zip(states["state"], states["state_name"]))
    c1, c2 = st.columns([1, 2])
    state = c1.selectbox("State", list(names), format_func=names.get, key="profile_state")
    counties = year_all[year_all["state"] == state].sort_values("county")
    if st.session_state.get("profile_fips") not in set(counties["fips"]):
        st.session_state["profile_fips"] = counties["fips"].iloc[0]
    # PLACES names independent cities like their namesake county (Fairfax 51059 / 51600).
    dup = counties["county"].duplicated(keep=False)
    is_city = counties["fips"].str[2:].astype(int) >= 500
    names_shown = counties["county"].where(~(dup & is_city), counties["county"] + " city")
    labels = dict(zip(counties["fips"], names_shown))
    fips = c2.selectbox(
        "County", list(labels), format_func=lambda f: f"{labels[f]} ({f})", key="profile_fips",
        help="Defaults to the alphabetically first county (Autauga, Alabama).",
    )

    if pin and pin["state"] and pin["fips"] != fips:
        col, _ = st.columns([1, 2])
        col.button(f"📍 Looked-up: {_county_name(year_all, pin['fips'])}", key="profile_pin",
                   on_click=_show_county, args=(pin["fips"], pin["state"]), width="stretch")
    return fips


def _section(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def _profile_metrics(r: pd.Series, state_rows: pd.DataFrame, year_all: pd.DataFrame) -> None:
    # Same 6-column grid as the KPI row, first slot (Counties) left empty, so each card
    # sits under its national counterpart.
    c = st.columns(6)[1:]
    score = r.get(COMPOSITE)
    c[0].metric(
        COMPOSITE_SHORT, "–" if pd.isna(score) else f"{score:.0f}", r.get("risk_tier") or None,
        delta_color="off", delta_arrow="off",
        help=f"{COMPOSITE_LABEL}: each sensory rate (vision, hearing, tooth loss) becomes a z-score, the z-scores are "
            "averaged with the chosen weights, and the average is ranked 0–100 against every county-year. "
             + ("" if pd.isna(score) else f"Example: {score:.0f} means this county's combined sensory burden "
                f"is heavier than {score:.0f}% of all county-years."),
    )
    for col, m in zip(c[1:], [OUTCOME_MEASURE, *SENSORY_MEASURES]):
        v = r.get(m)
        state_avg = weighted_mean(state_rows[m], state_rows["adult_population"])
        us_avg = weighted_mean(year_all[m], year_all["adult_population"])
        pct = (year_all[m].dropna() <= v).mean() * 100 if pd.notna(v) else float("nan")
        col.metric(
            MEASURE_LABELS[m], _fmt_pct(v),
            _delta(v, state_avg, r["state"]),
            delta_color="inverse",
            help=(f"{MEASURE_DESCRIPTIONS[m]} Change = county rate − adult-weighted average: "
                  f"vs {r['state']} {v:.1f}% − {state_avg:.1f}% = {v - state_avg:+.1f} pp; "
                  f"vs US {v:.1f}% − {us_avg:.1f}% = {v - us_avg:+.1f} pp. National percentile {pct:.0f}: "
                  f"the rate is at or above that of {pct:.0f}% of US counties.") if pd.notna(pct)
            else MEASURE_DESCRIPTIONS[m],
        )
        _us_badge(col, v, us_avg)


_MEDICAID_STATUS = {"Yes": "✓ Covered", "No": "✕ Not covered", "NR": "– Not reported"}


def _render_medicaid(medicaid: pd.DataFrame, state: str, state_name: str) -> None:
    rows = medicaid[medicaid["state"] == state].set_index("benefit")
    year = int(medicaid["survey_year"].max())
    _section(f"{state_name} Medicaid adult benefits",
             f"KFF Medicaid Benefits Database, {year} survey (state-reported fee-for-service coverage "
             "for categorically needy adults). Coverage may have changed since.")
    items = []
    for b, label in MEDICAID_BENEFITS.items():
        status = rows.loc[b, "covered"] if b in rows.index else "NR"
        weight = 600 if status == "Yes" else 400
        items.append(
            f"<div style='display:flex;justify-content:space-between;padding:7px 0;"
            f"border-bottom:1px solid {GRIDLINE}'><span>{label}</span>"
            f"<span style='font-weight:{weight};color:{INK_PRIMARY if status == 'Yes' else INK_SECONDARY}'>"
            f"{_MEDICAID_STATUS.get(status, status)}</span></div>"
        )
    st.markdown("".join(items), unsafe_allow_html=True)
    details = rows.reindex(list(MEDICAID_BENEFITS))[["covered", "copay", "limits", "prior_approval"]]
    if details[["copay", "limits", "prior_approval"]].notna().any().any():
        with st.expander("Copays, limits and prior approval"):
            st.dataframe(
                details.rename(index=MEDICAID_BENEFITS, columns={
                    "covered": "Covered", "copay": "Copay", "limits": "Limits",
                    "prior_approval": "Prior approval"}),
                width="stretch",
            )


def _render_ma_plans(access: CountyAccess, fips: str, state: str, year_all: pd.DataFrame) -> None:
    plan_year = int(access.plans["year"].max())
    _section("Medicare Advantage plans in this county",
             f"CMS Plan Benefit Package {plan_year}: individual (non-employer) Medicare Advantage plans "
             "offered in the county, including special needs plans. Benefits are supplemental to "
             "Original Medicare.")
    state_fips = set(year_all.loc[year_all["state"] == state, "fips"])
    summary = model.plan_access_summary(access.plan_shares, fips, state_fips)
    if summary is None:
        note = (" CMS lists Connecticut by its legacy counties, which do not map to planning regions."
                if state == "CT" else "")
        st.info("No individual Medicare Advantage plans found for this county." + note)
        return
    local = access.plans[access.plans["bid_id"].isin(
        access.plan_counties.loc[access.plan_counties["fips"] == fips, "bid_id"])]
    c1, c2 = st.columns(2)
    c1.metric("Plans offered", f"{len(local):,}")
    c2.metric("Special needs plans", f"{int(local['snp'].sum()):,}",
              help="SNPs enrol only eligible people (dual-eligible, chronic condition or institutional).")
    st.dataframe(
        summary.assign(benefit=summary["benefit"].map(PLAN_BENEFITS)).rename(columns={
            "benefit": "Benefit", "plans": "Plans", "share": "Share of plans",
            "state_median": f"{state} county median", "us_median": "US county median"}),
        hide_index=True, width="stretch",
        column_config={
            "Share of plans": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="percent",
                help="Plans offering the benefit ÷ plans offered in the county. Example: 40 of 52 plans = 77%."),
            f"{state} county median": st.column_config.NumberColumn(
                format="percent", help=f"Middle value of the same share across {state} counties: half of "
                                       "the counties are above it, half below."),
            "US county median": st.column_config.NumberColumn(
                format="percent", help="Middle value of the same share across all US counties."),
        },
    )
    with st.expander(f"All {len(local)} plans"):
        flags = {b: label for b, label in PLAN_BENEFITS.items() if b != "dental_any"}
        st.dataframe(
            local.sort_values(["organization", "plan_name"])[
                ["plan_name", "organization", "plan_type", "snp", *flags]
            ].rename(columns={"plan_name": "Plan", "organization": "Organization",
                              "plan_type": "Type", "snp": "SNP", **flags}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.CheckboxColumn() for c in ["SNP", *flags.values()]},
        )


def _sites_map(local: pd.DataFrame, fips: str, year: int) -> None:
    points = local.dropna(subset=["lat", "lon"]).assign(
        label=lambda d: d["site_name"], org=lambda d: d["health_center"],
        where=lambda d: d["address"].fillna("") + ", " + d["city"].fillna(""),
    )[["lon", "lat", "label", "org", "where", "setting"]]
    layers = [
        _outline_layer(dl.selection_outline((fips,), year, by_state=False), "profile_county", 2),
        pdk.Layer(
            "ScatterplotLayer", data=points, id="hc_sites", get_position=["lon", "lat"],
            get_fill_color=_hex_to_rgb(COMPOSITE_COLOR), get_line_color=[255, 255, 255, 255],
            get_radius=6, radius_units='"pixels"', stroked=True, get_line_width=1.5,
            line_width_units='"pixels"', pickable=True,
        ),
    ]
    deck = CompactDeck(
        layers=layers, initial_view_state=view_state({fips}, True, padding=0.12),
        map_provider="carto", map_style="light",
        tooltip={
            "html": "<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.45'>"
                    "<b>{label}</b><br/>{org}<br/>{where}<br/><span style='color:#52514e'>{setting}</span></div>",
            "style": {"backgroundColor": "white", "color": INK_PRIMARY, "border": "1px solid #ccc",
                      "borderRadius": "4px", "boxShadow": "0 1px 4px rgba(0,0,0,.2)", "maxWidth": "280px"},
        },
    )
    st.pydeck_chart(deck, height=380, key="profile_sites_map")


def _render_county_uds_summary(access: CountyAccess, orgs: pd.DataFrame, place: str) -> None:
    totals, reporting = model.uds_county_summary(orgs, access.uds, list(UDS_SERVICES))
    st.plotly_chart(uds_county_chart(totals, reporting, len(orgs),
                                     f"All services, all organizations in {place} (est.)"),
                    width="stretch")
    latest = access.uds[access.uds["year"] == access.uds["year"].max()]
    ex = (orgs.merge(latest[["bhcmis_id", "oral_exam_patients"]], on="bhcmis_id")
          .dropna(subset=["oral_exam_patients"]).head(1))
    example = ""
    if not ex.empty:
        e = ex.iloc[0]
        example = (f" Example: {e['health_center'].title()} reported {int(e['oral_exam_patients']):,} oral "
                   f"exams with {int(e['sites_in_county'])} of its {int(e['sites_total'])} sites here → "
                   f"{int(e['oral_exam_patients']):,} × {int(e['sites_in_county'])}/{int(e['sites_total'])} ≈ "
                   f"{e['oral_exam_patients'] * e['sites_in_county'] / e['sites_total']:,.0f} counted for the county.")
    st.caption(
        (f"Estimated from the {len(orgs)} organizations with sites in the county: each one's "
         if len(orgs) != 1 else "Estimated from the one organization with sites in the county: its ")
        + "UDS count × its share of sites located here. Lines join years where all of them "
        "published; open markers are years with unpublished counts, so those totals are "
        "understated and not comparable." + example
    )


def _render_health_centers(
    access: CountyAccess, fips: str, year: int, local: pd.DataFrame, orgs: pd.DataFrame, place: str,
) -> None:
    _section("Health centers (HRSA)",
             "HRSA-funded health center and look-alike service delivery sites located in the county.")
    if local.empty:
        st.info("No HRSA health center service delivery sites are located in this county.")
        return
    c = st.columns(4)
    c[0].metric("Sites", f"{len(local):,}")
    c[1].metric("Organizations", f"{local['bhcmis_id'].nunique():,}")
    c[2].metric("School-based", f"{int((local['setting'] == 'School').sum()):,}")
    c[3].metric("Mobile vans", f"{int((local['location_type'] == 'Mobile Van').sum()):,}")

    _render_uds_history(access.uds, orgs)

    st.markdown(f"**Service delivery sites in this county ({len(local):,})**")
    map_col, list_col = st.columns([2, 3])
    with map_col:
        _sites_map(local, fips, year)
    with list_col:
        st.dataframe(
            local.sort_values(["health_center", "site_name"]).assign(
                address=lambda d: d["address"].fillna("") + ", " + d["city"].fillna("") + " " + d["zip"].fillna("")
            )[["site_name", "health_center", "address", "setting", "location_type", "hours_per_week"]]
            .rename(columns={"site_name": "Site", "health_center": "Organization", "address": "Address",
                             "setting": "Setting", "location_type": "Type", "hours_per_week": "Hours/week"}),
            hide_index=True, width="stretch", height=380,
            column_config={"Hours/week": st.column_config.NumberColumn(format="%.0f")},
        )


def _fmt_count(v) -> str:
    return "–" if pd.isna(v) else f"{int(v):,}"


def _published(values: list) -> list[float]:
    known = [float(v) for _, v in values if pd.notna(v)]
    return known if len(known) >= 2 else []


def _change_since_first(values: list) -> str:
    """Change from the first to the last published year, e.g. "+12% (2020→2025)"."""
    known = [(y, v) for y, v in values if pd.notna(v)]
    if len(known) < 2 or known[0][1] == 0:
        return "–"
    (y0, v0), (y1, v1) = known[0], known[-1]
    return f"{(v1 - v0) / v0:+.0%} ({y1} vs {y0})"


def _render_uds_history(uds: pd.DataFrame, orgs: pd.DataFrame) -> None:
    years = sorted(int(y) for y in uds["year"].unique())
    st.markdown(f"**Dental and vision services by organization (UDS {years[0]}–{years[-1]})**")
    services = list(UDS_SERVICES)
    service = st.selectbox(
        "Service", services, index=services.index("oral_exam_patients"),
        format_func=UDS_SERVICES.get, key="uds_service",
        help="Exams, sealants and fluoride count patients. Table 5 dental and vision totals are "
             "withheld for most organizations before 2022.",
    )
    hist = model.uds_service_history(orgs, uds, service)
    series = [list(zip(years, row)) for row in hist[years].itertuples(index=False)]
    table = pd.DataFrame({
        "Organization": hist["health_center"],
        "Sites here": hist["sites_in_county"],
        "All sites": hist["sites_total"],
        **{str(y): hist[y].map(_fmt_count) for y in years},
        # Sparklines can't show gaps, so they join the published years only.
        "Trend": [_published(pts) for pts in series],
        "Change": [_change_since_first(pts) for pts in series],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            **{str(y): st.column_config.TextColumn(alignment="right") for y in years},
            "Trend": st.column_config.LineChartColumn(
                width="small", help=f"Published years between {years[0]} and {years[-1]}"),
            "Change": st.column_config.TextColumn(
                alignment="right",
                help="(last published value − first published value) ÷ first. Example: 758 in 2020 and "
                     "1,292 in 2025 → (1,292 − 758) ÷ 758 = +70% (2025 vs 2020)."),
        },
    )

    with st.expander("All services for one organization, year by year"):
        names = dict(zip(orgs["bhcmis_id"], orgs["health_center"]))
        if st.session_state.get("uds_org") not in names:
            st.session_state["uds_org"] = orgs["bhcmis_id"].iloc[0]
        org_id = st.selectbox("Organization", list(names), format_func=names.get, key="uds_org")
        detail = model.uds_org_history(uds, org_id, services)
        detail = detail.map(_fmt_count).rename(index=UDS_SERVICES)
        detail.columns = [str(y) for y in detail.columns]
        st.dataframe(
            detail, width="stretch",
            column_config={c: st.column_config.TextColumn(alignment="right") for c in detail.columns},
        )

    st.caption(
        "UDS reports services per organization across all of its sites, including sites outside this "
        "county; compare *Sites here* with *All sites* before reading the totals as county activity. "
        "A dash (–) means not published: the public UDS file withholds some counts (most Table 5 "
        "dental and vision totals before 2022), an organization may not have reported that year, and look-alikes "
        "report separately. UDS has no hearing or audiology service line."
    )


# --------------------------------------------------------------------------- #
# Nursing homes and dementia care
# --------------------------------------------------------------------------- #
# Cell colours shared with the delta badges under the metrics.
WORSE_STYLE = "color:#bd2d2d;background-color:rgba(255,75,75,0.12)"
BETTER_STYLE = "color:#177a3a;background-color:rgba(33,195,84,0.12)"


def _diff_cell(value: float, ref: float, higher_better: bool, unit: str) -> tuple[str, str]:
    """Difference from a reference with an arrow, e.g. "↑ +1.8 pp", and its status:
    "worse" / "better", or "" when missing or within ±10% of the reference."""
    if pd.isna(value) or pd.isna(ref):
        return "–", ""
    d = value - ref
    text = f"{d:+.1f} pp" if unit == "%" else f"{d:+.2f}"
    if text.lstrip("+-").strip("0. p") == "":  # rounds to zero
        return text.lstrip("+-"), ""
    arrow = "↑" if d > 0 else "↓"
    if abs(d) <= max(abs(ref) * 0.10, 0.05):
        return f"{arrow} {text}", ""
    better = (d > 0) == higher_better
    return f"{arrow} {text}", "better" if better else "worse"


def _nh_comparison(county: pd.Series, state: pd.Series | None, us: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indicator table (county, then each reference preceded by the county's difference
    from it) plus a same-shaped frame of "worse" / "better" flags for colouring the
    difference and reference cells."""
    rows, flags = [], []
    items = [(k, label, hb, "%") for k, (label, hb) in NH_MEASURES.items()]
    items += [(k, label, False, "per home") for k, label in NH_CITATIONS.items()]
    items += [(k, label, hb, "") for k, (label, hb) in NH_OTHER.items()]
    for key, label, higher_better, unit in items:
        def fmt(v):
            if v is None or pd.isna(v):
                return "–"
            return f"{v:.1f}%" if unit == "%" else f"{v:.2f}"
        c, st_v, us_v = county.get(key), (state.get(key) if state is not None else None), us.get(key)
        vs_state, state_flag = _diff_cell(c, st_v if st_v is not None else float("nan"), higher_better, unit)
        vs_us, us_flag = _diff_cell(c, us_v, higher_better, unit)
        rows.append({"Indicator": label, "County": fmt(c), "vs State": vs_state, "State": fmt(st_v),
                     "vs US": vs_us, "US": fmt(us_v)})
        # each reference value is coloured like its comparison cell
        flags.append({"vs State": state_flag, "State": state_flag, "vs US": us_flag, "US": us_flag})
    table = pd.DataFrame(rows)
    status = pd.DataFrame(flags).reindex(columns=table.columns, fill_value="")
    return table, status


def _style_flags(table: pd.DataFrame, status: pd.DataFrame):
    styles = status.replace({"worse": WORSE_STYLE, "better": BETTER_STYLE})
    return table.style.apply(lambda _: styles, axis=None)


def nh_burden_chart(by_fips: pd.DataFrame, names: dict[str, str], state_fips: set[str], fips: str,
                    us: dict, title: str) -> go.Figure:
    """Counties of one state: cognitive difficulty among adults 65+ against long-stay
    antipsychotic use in their nursing homes; dashed lines are US references."""
    d = by_fips.loc[by_fips.index.isin(state_fips)].dropna(subset=["cog65_rate", "ls_antipsychotic"])
    others, me = d[d.index != fips], d[d.index == fips]
    size = lambda b: (6 + 16 * np.sqrt(b / max(d["beds"].max(), 1))).tolist()  # noqa: E731
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=others["cog65_rate"], y=others["ls_antipsychotic"], mode="markers", name="Other counties",
        marker=dict(size=size(others["beds"]), color=GRIDLINE, line=dict(color=INK_MUTED, width=1)),
        text=[names.get(f, f) for f in others.index], customdata=others["beds"],
        hovertemplate="%{text}<br>65+ cognitive difficulty %{x:.1f}%<br>"
                      "Antipsychotics %{y:.1f}%<br>%{customdata:,.0f} beds<extra></extra>",
    ))
    if len(me):
        fig.add_trace(go.Scatter(
            x=me["cog65_rate"], y=me["ls_antipsychotic"], mode="markers+text", name=names.get(fips, fips),
            marker=dict(size=size(me["beds"]), color=COMPOSITE_COLOR, line=dict(color="white", width=2)),
            text=[names.get(fips, fips)], textposition="top center",
            textfont=dict(color=COMPOSITE_COLOR, size=12),
            hovertemplate="65+ cognitive difficulty %{x:.1f}%<br>Antipsychotics %{y:.1f}%<extra></extra>",
        ))
    ref = dict(color=INK_MUTED, width=1, dash="dash")
    fig.add_vline(x=us["county_median_cog65_rate"], line=ref)  # US county median
    fig.add_hline(y=us["ls_antipsychotic"], line=ref)           # US resident-weighted average
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="top", showarrow=False,
                       text="Higher burden, more antipsychotic use", font=dict(size=11, color=INK_SECONDARY))
    _base_layout(
        fig, height=380, showlegend=False,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY), x=0, y=0.98),
        xaxis=dict(title="Adults 65+ with a cognitive difficulty (%)", showgrid=True, gridcolor=GRIDLINE),
        yaxis=dict(title="Long-stay residents on antipsychotics (%)", rangemode="tozero"),
    )
    return fig


def _nh_findings(name: str, c: pd.Series, us: dict, cog_pct: float) -> list[str]:
    """Short policy reading of the county against national references."""
    out = [f"**{name}**: {c['cog65_rate']:.1f}% of adults 65+ report a cognitive difficulty "
           f"(US {us['cog65_rate']:.1f}%; {_ordinal(round(cog_pct))} percentile of counties)."]
    if c.get("facilities", 0) == 0:
        out.append("No certified nursing home is located in the county, so residents needing "
                   "dementia care must be placed elsewhere.")
        return out
    n = int(c["facilities"])
    homes = "Its nursing home has" if n == 1 else f"Its {n} nursing homes have"
    out.append(f"{homes} {c['beds_per_100_cog65']:.0f} certified beds per "
               f"100 older adults with a cognitive difficulty (US {us['beds_per_100_cog65']:.0f}) and give "
               f"antipsychotics to {c['ls_antipsychotic']:.1f}% of long-stay residents (US {us['ls_antipsychotic']:.1f}%).")
    high_burden = c["cog65_rate"] > us["county_median_cog65_rate"]
    if high_burden and c["ls_antipsychotic"] > us["ls_antipsychotic"] * 1.1:
        out.append("⚠️ **Dementia-care priority:** above-median cognitive burden with above-average "
                   "antipsychotic use, a target for dementia-care training and non-drug behavioural care.")
    if high_burden and c["beds_per_100_cog65"] < us["beds_per_100_cog65"] * 0.9:
        out.append("⚠️ **Capacity gap:** fewer nursing home beds per older adult with a cognitive "
                   "difficulty than the US; prioritise home- and community-based dementia services.")
    if c.get("cit_dementia_care", 0) > us["cit_dementia_care"] * 1.5 and c["cit_dementia_care"] > 0:
        out.append("⚠️ Dementia-care (F744) citations per home exceed 1.5× the US rate.")
    return out


def _render_nursing_homes(nh: NursingHomes | None, fips: str, r: pd.Series, year_all: pd.DataFrame) -> None:
    _section("Nursing homes & dementia care (CMS)",
             "Medicare- and Medicaid-certified nursing homes in the county set against older adults with "
             "a cognitive difficulty. Quality measures are resident-weighted averages.")
    if nh is None:
        st.info("Nursing home data is not built yet. Run `python scripts/preprocess_access.py`.")
        return
    if fips not in nh.by_fips.index or pd.isna(nh.by_fips.loc[fips, "cog65_rate"]):
        st.info("No ACS estimate of older adults with a cognitive difficulty for this county.")
        return
    c = nh.by_fips.loc[fips]
    state = nh.by_state.loc[fips[:2]] if fips[:2] in nh.by_state.index else None
    us = nh.us
    cog_pct = (nh.by_fips["cog65_rate"].dropna() <= c["cog65_rate"]).mean() * 100

    m = st.columns(5)
    m[0].metric(
        "65+ cognitive difficulty", f"{c['cog65_rate']:.1f}%",
        None if state is None else _delta(c["cog65_rate"], state["cog65_rate"], r["state"]),
        delta_color="inverse",
        help=(f"Adults 65+ with a cognitive difficulty ÷ all adults 65+ × 100 (ACS 5-year, table B18104). "
              f"Here: {int(c['cog65']):,} ÷ {int(c['pop65']):,} × 100 = {c['cog65_rate']:.1f}%, about 1 in "
              f"{max(round(100 / c['cog65_rate']), 1) if c['cog65_rate'] else '–'} older adults. "
              f"vs US: {c['cog65_rate']:.1f}% − {us['cog65_rate']:.1f}% = "
              f"{c['cog65_rate'] - us['cog65_rate']:+.1f} pp."),
    )
    _us_badge(m[0], c["cog65_rate"], us["cog65_rate"])
    m[1].metric("Nursing homes", f"{int(c['facilities']):,}")
    m[2].metric("Certified beds", f"{int(c['beds']):,}",
                help=None if c["facilities"] == 0 else
                f"Occupancy = average residents per day ÷ certified beds. Here: {c['residents']:,.0f} ÷ "
                f"{int(c['beds']):,} = {c['residents'] / c['beds']:.0%} of beds in use on a typical day.")
    m[3].metric("Beds per 100 at-risk seniors",
                "–" if pd.isna(c["beds_per_100_cog65"]) else f"{c['beds_per_100_cog65']:.0f}",
                None if pd.isna(c["beds_per_100_cog65"]) else
                _vs(f"{c['beds_per_100_cog65'] - us['beds_per_100_cog65']:+.0f}", "US"),
                delta_color="normal",  # fewer beds than the US = capacity gap (red), as in the findings
                help=("Certified nursing home beds ÷ adults 65+ with a cognitive difficulty × 100. "
                      + ("" if pd.isna(c["beds_per_100_cog65"]) else
                         f"Here: {int(c['beds']):,} ÷ {int(c['cog65']):,} × 100 = {c['beds_per_100_cog65']:.0f}, "
                         f"i.e. about {c['beds_per_100_cog65']:.0f} beds for every 100 such seniors "
                         f"(US {us['beds_per_100_cog65']:.0f}). ")
                      + "Green: more beds than the US, red: fewer (a capacity gap). Not all residents have "
                        "dementia and not all people with dementia need a nursing home; read as capacity context."))
    m[4].metric("Avg. overall rating", "–" if pd.isna(c.get("overall_rating")) else f"{c['overall_rating']:.1f} ★",
                None if pd.isna(c.get("overall_rating")) else
                _vs(f"{c['overall_rating'] - us['overall_rating']:+.1f}", "US"),
                delta_color="normal",
                help="Resident-weighted CMS overall star rating: Σ(stars × residents) ÷ Σ residents. Example: "
                     "a 5★ home with 100 residents and a 2★ home with 50 average (5 × 100 + 2 × 50) ÷ 150 = 4.0★.")

    for line in _nh_findings(f"{r['county']}, {r['state']}", c, us, cog_pct):
        st.markdown(line)

    with st.container():
        st.markdown("**Dementia-care indicators**")
        if c["facilities"] == 0:
            st.info("No nursing homes in the county to report on.")
        else:
            table, status = _nh_comparison(c, state, us)
            weighted = ("Resident-weighted: Σ(home's score × its residents) ÷ Σ residents. Example: 10% in a "
                        "home of 100 residents and 20% in one of 50 → (10 × 100 + 20 × 50) ÷ 150 = 13.3%. "
                        "Citation rows: citations ÷ homes.")
            diff_help = ("County − {ref}. Red: the county is worse, green: better (direction depends on the "
                         "indicator; lower is better except star rating and RN hours). The {ref} value cell takes "
                         "the same colour; within ±10% of it both stay uncoloured. Example: 17.0% vs {ref} 15.2% "
                         "→ ↑ +1.8 pp, 12% higher, red for antipsychotic use.")
            st.dataframe(_style_flags(table, status), hide_index=True, width="stretch",
                         height=35 * (len(table) + 1) + 3,
                         column_config={
                             "County": st.column_config.TextColumn(alignment="right", help=weighted),
                             "State": st.column_config.TextColumn(alignment="right", help=weighted),
                             "US": st.column_config.TextColumn(alignment="right", help=weighted),
                             "vs State": st.column_config.TextColumn(
                                 alignment="right", help=diff_help.format(ref="state")),
                             "vs US": st.column_config.TextColumn(
                                 alignment="right", help=diff_help.format(ref="US")),
                         })

    homes = nh.facilities[nh.facilities["fips"] == fips]
    if len(homes):
        with st.expander(f"All {len(homes)} nursing homes in the county"):
            st.dataframe(
                homes.sort_values("name").assign(
                    psychotropic=lambda d: d["cit_psychotropic_gdr"] + d["cit_chemical_restraint"])[
                    ["name", "city", "beds", "residents", "overall_rating", "ls_antipsychotic",
                     "cit_dementia_care", "psychotropic", "rn_hprd"]
                ].rename(columns={
                    "name": "Nursing home", "city": "City", "beds": "Beds", "residents": "Residents/day",
                    "overall_rating": "Rating", "ls_antipsychotic": "Antipsychotics (long-stay)",
                    "cit_dementia_care": "F744", "psychotropic": "F758 + F605", "rn_hprd": "RN h/resident-day"}),
                hide_index=True, width="stretch",
                column_config={
                    "Beds": st.column_config.NumberColumn(format="%d"),
                    "Residents/day": st.column_config.NumberColumn(format="%.0f"),
                    "Rating": st.column_config.NumberColumn(format="%d ★"),
                    "Antipsychotics (long-stay)": st.column_config.NumberColumn(format="%.1f%%"),
                    "RN h/resident-day": st.column_config.NumberColumn(format="%.2f"),
                },
            )
    period = nh.facilities["mds_period"].iloc[0] if len(nh.facilities) else ""
    st.caption(
        f"CMS Care Compare: MDS quality measures {period} (four-quarter averages); citations from the last "
        "three standard and complaint inspection cycles. Older adults with a cognitive difficulty: ACS "
        "2019–2023 5-year estimates (self- or proxy-reported serious difficulty concentrating, remembering "
        "or making decisions; household and group-quarters population), with wide margins of error in "
        "small counties. Ecological comparison: it flags where to look, not why."
    )


def render_county_profile(
    fips: str, panel: pd.DataFrame, year: int, access: CountyAccess | None,
) -> None:
    county_rows = panel[panel["fips"] == fips]
    year_all = panel[panel["year"] == year]
    current = county_rows[county_rows["year"] == year]
    if current.empty:
        st.warning(f"No PLACES estimate for this county in {year}.")
        return
    r = current.iloc[0]
    adults = r.get("adult_population")
    st.markdown(f"### {r['county']}, {r['state_name']}")
    st.caption(f"FIPS {fips}, {int(adults):,} adults, {year} BRFSS estimates (age-adjusted)"
               if pd.notna(adults) else f"FIPS {fips}, {year} BRFSS estimates (age-adjusted)")
    _profile_metrics(r, year_all[year_all["state"] == r["state"]], year_all)

    local, orgs = (model.county_health_centers(access.sites, fips) if access is not None
                   else (pd.DataFrame(), pd.DataFrame()))
    trend_col, org_col, burden_col = st.columns(3, gap="medium")
    with trend_col:
        st.plotly_chart(
            trend_chart(model.national_trend(county_rows), year,
                        f"Prevalence trend in {r['county']}, {r['state']}"),
            width="stretch",
        )
    with burden_col:
        nh = access.nursing_homes if access is not None else None
        if nh is None:
            st.info("Nursing home data is not built yet.")
        else:
            names = dict(zip(year_all["fips"], year_all["county"]))
            state_fips = set(year_all.loc[year_all["state"] == r["state"], "fips"])
            st.plotly_chart(nh_burden_chart(nh.by_fips, names, state_fips, fips, nh.us,
                                            f"Cognitive burden vs. antipsychotic use in {r['state']}"),
                            width="stretch")
    with org_col:
        if orgs.empty:
            st.info("No HRSA health center organizations operate sites in this county."
                    if access is not None else "Health center data is not built yet.")
        else:
            _render_county_uds_summary(access, orgs, f"{r['county']}, {r['state']}")

    st.divider()
    if access is None:
        st.info("Care-access data is not built yet. Run `python scripts/preprocess_access.py` to add "
                "Medicaid benefits, Medicare Advantage plans and health center data.")
        return
    _render_health_centers(access, fips, year, local, orgs, f"{r['county']}, {r['state']}")
    st.divider()
    left, right = st.columns([2, 3], gap="large")
    with left:
        _render_medicaid(access.medicaid, r["state"], r["state_name"])
    with right:
        _render_ma_plans(access, fips, r["state"], year_all)
    st.divider()
    _render_nursing_homes(access.nursing_homes, fips, r, year_all)

    st.caption("Sources: " + ", ".join(f"[{name}]({url})" for name, url in ACCESS_SOURCES.values()))
