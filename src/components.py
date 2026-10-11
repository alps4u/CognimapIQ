"""Streamlit rendering components: sidebar controls, KPI cards, map and charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src import data_loader as dl
from src.config import (
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
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    MEASURE_COLORS,
    MEASURE_DESCRIPTIONS,
    MEASURE_LABELS,
    NEUTRAL_FILL,
    NO_DATA_COLOR,
    OUTCOME_MEASURE,
    OVERLAY_STYLES,
    PATTERN_LINE_PX,
    PATTERN_OPACITY,
    PATTERN_SPACING_PX,
    PATTERN_TOP_DEFAULT,
    PATTERN_TOP_SHARES,
    POLICY_EVENTS,
    SAMPLE_ADDRESS,
    SENSORY_MEASURES,
    SEQUENTIAL_RAMP,
)
from src.model import ModelFit, ScenarioResult

RESIDUAL = "residual"
MAP_MODES = {"sori": COMPOSITE_LABEL, "overlay": "Cognitive & sensory overlay"}
OVERLAY_HINTS = {
    "COGNITION": "blue fill",
    "VISION": "orange ╱ hatch",
    "HEARING": "teal ╲ hatch",
    "TEETHLOST": "brown dots",
}


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
            "US street address", value=SAMPLE_ADDRESS, key="lookup_address",
            placeholder=f"e.g. {SAMPLE_ADDRESS}",
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
            if sb.checkbox(f"{MEASURE_LABELS[m]} — {OVERLAY_HINTS[m]}", value=True, key=f"ov_{m}"):
                overlays.append(m)
    pattern_top_pct = sb.select_slider(
        "Pattern threshold (top %)",
        options=PATTERN_TOP_SHARES,
        value=PATTERN_TOP_DEFAULT,
        format_func=lambda v: f"Top {v}%",
        disabled=map_mode != "overlay",
        help="Hatching / dots mark counties in this top share of each sensory measure "
             "(pooled across years). Applies to the overlay view.",
    )

    sb.subheader(COMPOSITE_LABEL)
    weight_mode = sb.radio(
        "Weighting",
        ["Equal", "Custom"],
        horizontal=True,
        help="Equal: each sensory domain counts the same. Custom: set policy weights yourself.",
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
            st.markdown(f"**{r['county']}, {r['state']}** · FIPS `{match['fips']}`")
            st.caption(match["matched_address"])
            score = r.get(COMPOSITE)
            st.metric(
                f"{COMPOSITE_SHORT} ({year})", "–" if pd.isna(score) else f"{score:.0f}",
                r.get("risk_tier") or None, delta_color="off", delta_arrow="off",
                help="0–100 percentile of combined sensory burden, pooled across all years.",
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
                    flag = f" · top {top_pct}%" if v >= cutoff else ""
                st.progress(pct / 100, text=f"{MEASURE_LABELS[m]} {v:.1f}% · {_ordinal(pct)} pct{flag}")
            st.caption(f"Percentile among US counties in {year}; higher = more prevalent.")
            if match["fips"] not in selected_fips:
                st.caption("Outside the selected region — shown dimmed on the map.")
        if st.button("Clear lookup", key="clear_lookup", width="stretch"):
            st.session_state.pop("lookup", None)
            st.rerun()


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
        f"Avg. {COMPOSITE_SHORT}", "–" if pd.isna(stats["score_mean"]) else f"{stats['score_mean']:.0f}",
        _delta(stats["score_mean"], prev and prev["score_mean"], ""),
        delta_color="inverse",
        help=f"{COMPOSITE_LABEL}: 0–100 percentile of combined sensory burden, pooled across all years.",
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
    rows = year_df.set_index("fips")
    features = []
    for feat in geojson["features"]:
        fips = feat["properties"]["fips"]
        props = {"fips": fips, "name": feat["properties"]["name"]}
        if fips in rows.index:
            r = rows.loc[fips]
            in_sel = fips in selected_fips
            color = _classify(r.get(fill_metric), breaks, ramp) if fill_metric else NEUTRAL_FILL
            props.update(
                name=f"{r['county']}, {r['state']}",
                fill=_hex_to_rgb(color, 235 if in_sel else 55),
                score_txt=_fmt_metric(COMPOSITE, r.get(COMPOSITE)),
                tier=r.get("risk_tier") or "–",
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
    "Vision: {VISION_txt} · Hearing: {HEARING_txt}<br/>"
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
# map, like Leaflet's auto-direction tooltips.
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
})();
</script>
"""


def install_tooltip_placement() -> None:
    st.html(TOOLTIP_PLACEMENT_JS, unsafe_allow_javascript=True)


def render_map(
    layers: list[pdk.Layer], view: pdk.ViewState, pin: dict | None = None,
    highlight: list[pdk.Layer] = (), key: str = "risk_map",
) -> None:
    deck = pdk.Deck(
        layers=[*layers, *highlight, *_pin_layers(pin)],
        initial_view_state=view,
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": TOOLTIP_HTML,
            "style": {"backgroundColor": "white", "color": INK_PRIMARY, "border": "1px solid #ccc",
                      "borderRadius": "4px", "boxShadow": "0 1px 4px rgba(0,0,0,.2)",
                      "maxWidth": "280px"},
        },
    )
    st.pydeck_chart(deck, height=MAP_HEIGHT_PX, key=key)


def render_sori_map(
    geojson: dict, year_df: pd.DataFrame, panel: pd.DataFrame,
    selected_fips: set[str], view: pdk.ViewState, pin: dict | None = None,
    highlight: list[pdk.Layer] = (),
) -> None:
    breaks = color_breaks(tuple(panel[COMPOSITE].astype(float)), len(SEQUENTIAL_RAMP))
    features = build_map_features(geojson, year_df, COMPOSITE, breaks, SEQUENTIAL_RAMP, selected_fips)
    render_map([_county_layer(features)], view, pin, highlight)
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
    highlight: list[pdk.Layer] = (),
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
    render_map(layers, view, pin, highlight)

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
    cols = ["county", "state", COMPOSITE, "risk_tier",
            *SENSORY_MEASURES, OUTCOME_MEASURE, "predicted_cognition", RESIDUAL, "adult_population"]
    table = (
        df.dropna(subset=[COMPOSITE])
        .sort_values(COMPOSITE, ascending=False)
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
