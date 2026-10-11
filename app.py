"""Cognimap IQ — Senior Sensory Health & Cognitive Risk GIS Dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src import components as ui
from src import data_loader as dl
from src import model
from src.config import COMPOSITE, COMPOSITE_LABEL, COMPOSITE_SHORT, FAVICON_PATH, MEASURE_LABELS, SENSORY_MEASURES, VALUE_TYPE, VALUE_TYPE_LABEL

st.set_page_config(
    page_title="Cognimap IQ",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

PLAY_INTERVAL_S = 1.6


# --------------------------------------------------------------------------- #
# Cached analytics
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def model_fits(value_type: str) -> dict[int, model.ModelFit | None]:
    """National association model per year (independent of region filters)."""
    panel = dl.get_panel(value_type)
    return {y: model.fit_model(g, y) for y, g in panel.groupby("year")}


@st.cache_data(show_spinner="Scoring counties…")
def scored_panel(value_type: str, weights: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    panel = model.add_composite_score(dl.get_panel(value_type), dict(weights))
    fits = model_fits(value_type)
    parts = [model.add_model_outputs(g, fits.get(y)) for y, g in panel.groupby("year")]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
try:
    dl.load_places()
    dl.load_counties()
except dl.SnapshotMissingError as exc:
    ui.render_title("Cognimap IQ")
    st.error(str(exc))
    st.code("python scripts/preprocess.py", language="bash")
    st.stop()

years = dl.available_years()
controls = ui.render_sidebar(dl.state_options())

# Time-slider playback: advance the slider value before the widget is created.
if st.session_state.get("year") not in years:
    st.session_state["year"] = years[-1]
if st.session_state.get("playing") and not st.session_state.pop("_play_started", False):
    idx = years.index(st.session_state["year"])
    if idx + 1 < len(years):
        st.session_state["year"] = years[idx + 1]
    else:
        st.session_state["playing"] = False


def _toggle_play() -> None:
    if st.session_state.get("playing"):
        st.session_state["playing"] = False
    else:
        st.session_state.update(playing=True, _play_started=True, year=years[0])


# --------------------------------------------------------------------------- #
# Header & temporal slider
# --------------------------------------------------------------------------- #
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
with head_l:
    ui.render_title("Cognimap IQ")
    st.caption(
        "Spatial evidence linking sensory impairment — vision, hearing and mastication — "
        "to cognitive decline, to guide budgets toward low-cost preventative care during "
        "the intervention window of opportunity."
    )
with head_r:
    slider_col, play_col = st.columns([4, 1], vertical_alignment="bottom")
    year = slider_col.select_slider("Analysis year (BRFSS survey year)", options=years, key="year")
    play_col.button(
        "⏸" if st.session_state.get("playing") else "▶",
        help="Play the timeline year by year",
        on_click=_toggle_play,
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
fits = model_fits(VALUE_TYPE)
fit = fits.get(year)

if controls.weight_mode == "Custom":
    weights = model.normalize_weights(controls.custom_weights)
else:
    weights = model.normalize_weights({})

panel = scored_panel(VALUE_TYPE, tuple(sorted(weights.items())))

region = panel[panel["state"].isin(controls.states)] if controls.states else panel
selected_fips = set(region["fips"])
filtered = bool(controls.states)

year_all = panel[panel["year"] == year]
year_df = region[region["year"] == year]
prev_years = [y for y in years if y < year]
prev_df = region[region["year"] == prev_years[-1]] if prev_years else None

if controls.states:
    names = dict(zip(dl.state_options()["state"], dl.state_options()["state_name"]))
    region_label = ", ".join(names[s] for s in controls.states)
else:
    region_label = "United States (all counties)"

ui.render_lookup_result(controls.lookup_slot, year_all, panel, year, controls.pattern_top_pct,
                        selected_fips)
pin = ui.lookup_pin(year_all)

stats = model.summarize(year_df)
prev_stats = model.summarize(prev_df) if prev_df is not None and len(prev_df) else None

ui.render_kpis(stats, prev_stats, f"{region_label} — {year}")
if year_df.empty:
    st.warning("No counties match the current filters.")
    st.stop()

tab_map, tab_time, tab_model, tab_table = st.tabs(
    ["🗺️ Spatial risk map", "📈 Longitudinal & policy view", "🧮 Predictive analytics", "📋 Priority regions"]
)

# --------------------------------------------------------------------------- #
# Primary view — spatial map
# --------------------------------------------------------------------------- #
with tab_map:
    geojson = dl.year_geojson(year)
    # A looked-up address zooms to its state unless the user has chosen a region.
    if pin and pin["state"] and not filtered:
        view = ui.view_state(set(year_all.loc[year_all["state"] == pin["state"], "fips"]), True)
    else:
        view = ui.view_state(selected_fips, filtered)
    highlight = ui.highlight_layers(year, selected_fips, filtered, pin)
    if controls.map_mode == "overlay":
        st.markdown(f"#### Cognitive & sensory overlay · {year}")
        if not controls.overlays:
            st.info("Select at least one layer in the sidebar.")
        ui.render_overlay_map(geojson, year_all, panel, selected_fips, view,
                              controls.overlays, controls.pattern_top_pct, pin, highlight)
    else:
        st.markdown(f"#### {COMPOSITE_LABEL} · {year}")
        ui.render_sori_map(geojson, year_all, panel, selected_fips, view, pin, highlight)

    teeth_year = year_all["teethlost_source_year"].dropna()
    if len(teeth_year) and int(teeth_year.iloc[0]) != year:
        st.caption(
            f"Complete tooth loss is surveyed every two years; {year} uses "
            f"{int(teeth_year.iloc[0])} estimates."
        )

    w_txt = " · ".join(f"{MEASURE_LABELS[m]} {weights[m]:.0%}" for m in SENSORY_MEASURES)
    st.caption(f"{COMPOSITE_SHORT} weights ({controls.weight_mode.lower()}): {w_txt}")

# --------------------------------------------------------------------------- #
# Secondary view — longitudinal & policy
# --------------------------------------------------------------------------- #
with tab_time:
    left, right = st.columns([3, 2])
    with left:
        trend = model.national_trend(region)
        st.plotly_chart(
            ui.trend_chart(trend, year, f"Prevalence trend — {region_label}"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            ui.score_trend_chart(panel, selected_fips if filtered else set(),
                                 f"{COMPOSITE_SHORT} distribution over time"),
            width="stretch",
        )
    st.markdown("##### Policy & evidence milestones")
    st.caption("Numbered markers on the trend chart. Dotted lines are coloured by sensory domain.")
    ui.render_policy_timeline()

    if len(years) > 1:
        st.markdown(f"##### Largest increases in {COMPOSITE_LABEL}, {years[0]} → {years[-1]}")
        change = model.county_change(region, COMPOSITE, years[0], years[-1])
        st.dataframe(
            change.sort_values("change", ascending=False).head(15)
            .rename(columns={"county": "County", "state": "State", "start": str(years[0]),
                             "end": str(years[-1]), "change": "Change"})
            .drop(columns="fips"),
            hide_index=True, width="stretch",
            column_config={
                str(years[0]): st.column_config.NumberColumn(format="%.0f"),
                str(years[-1]): st.column_config.NumberColumn(format="%.0f"),
                "Change": st.column_config.NumberColumn(format="%+.0f"),
            },
        )

# --------------------------------------------------------------------------- #
# Predictive analytics engine
# --------------------------------------------------------------------------- #
with tab_model:
    if fit is None:
        st.info("Not enough data to fit the association model for this year.")
    else:
        st.markdown(
            f"**Association model ({year}, national, {VALUE_TYPE_LABEL.lower()})** — "
            f"Cognitive decline ~ Vision + Hearing + Complete tooth loss · "
            f"n = {fit.n:,} counties · R² = {fit.r2:.2f}"
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ui.scatter_chart(year_df, fit), width="stretch")
        with c2:
            st.plotly_chart(ui.coefficient_chart(fit), width="stretch")
            ui.render_model_table(fit)
            st.caption(
                "Ecological, cross-sectional association across counties. It identifies where "
                "sensory burden and cognitive decline co-occur; it does not establish causation."
            )

        st.divider()
        st.markdown("#### Preventative-care scenario")
        st.caption(
            "Simulate relative reductions in sensory impairment prevalence for the selected "
            "region (e.g. subsidised hearing aids, vision screening, dental/denture coverage) "
            "and see the model-expected change in cognitive decline."
        )
        sc = st.columns(3)
        reductions = {
            m: sc[i].slider(f"{MEASURE_LABELS[m]} reduction", 0, 50, 20, 5, format="%d%%",
                            key=f"red_{m}") / 100
            for i, m in enumerate(SENSORY_MEASURES)
        }
        ui.render_scenario(model.run_scenario(year_df, fit, reductions))

# --------------------------------------------------------------------------- #
# Priority regions
# --------------------------------------------------------------------------- #
with tab_table:
    t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
    t1.markdown(f"#### Intervention priority — {region_label}, {year}")
    top_n = t2.selectbox("Show", [25, 50, 100, 250, 500], index=1)
    ui.render_priority_table(year_df, top_n)
    export = year_df.drop(columns=[c for c in year_df.columns if c.startswith("z_")])
    st.download_button(
        "Download selection (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"cognimap_{year}.csv",
        mime="text/csv",
    )

st.caption(
    "Prototype (PoC) built on CDC PLACES as a stand-in for Canadian longitudinal data "
    "(CCHS, CLSA). Health-region boundaries will replace US counties after migration."
)

ui.install_tooltip_placement()

# Continue timeline playback after the page has rendered.
if st.session_state.get("playing"):
    time.sleep(PLAY_INTERVAL_S)
    st.rerun()
