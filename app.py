"""Cognimap IQ — Senior Sensory Health & Cognitive Risk GIS Dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from functools import partial

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
ui.install_map_styles()

PLAY_INTERVAL_S = 1.6

# Only the open tab is built on each run (st.tabs with on_change="rerun" exposes
# `.open`), which saves ~1 s per rerun. Streamlit drops the state of widgets that
# weren't drawn, so the hidden tabs' picks are re-stored here to survive a tab switch.
KEEP_WIDGET_STATE = ["map_geo", "profile_state", "profile_fips", "uds_service", "uds_org",
                     "top_n", *(f"red_{m}" for m in SENSORY_MEASURES)]
for _k in KEEP_WIDGET_STATE:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


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


@st.cache_data(show_spinner=False)
def state_panel(value_type: str, weights: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    return model.state_summary(scored_panel(value_type, weights))


@st.cache_data(show_spinner=False)
def plan_shares() -> pd.DataFrame:
    return model.plan_benefit_shares(dl.load_ma_plans(), dl.load_ma_plan_counties())


@st.cache_data(show_spinner=False)
def nursing_home_summaries() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return model.nh_summaries(dl.load_nh_facilities(), dl.load_senior_cognition())


@st.cache_resource(show_spinner=False)
def warm_profile(_panel: pd.DataFrame) -> None:
    """Pay the County profile's one-off costs (care-access summaries, county shapes,
    Plotly's lazily loaded validators, pandas Styler) once per server process."""
    import pandas.io.formats.style  # noqa: F401  (loads jinja2 for the flagged tables)

    county_access()
    dl.year_geometry(dl.available_years()[-1])
    one_county = _panel[_panel["fips"] == _panel["fips"].iloc[0]]
    ui.trend_chart(model.national_trend(one_county), int(one_county["year"].max()), "").to_json()


def county_access() -> ui.CountyAccess | None:
    """Care-access tables for the county profile, or None before preprocess_access.py has run."""
    if not dl.access_snapshot_ready():
        return None
    nursing_homes = None
    if dl.nh_snapshot_ready():
        by_fips, by_state, us = nursing_home_summaries()
        nursing_homes = ui.NursingHomes(dl.load_nh_facilities(), by_fips, by_state, us)
    return ui.CountyAccess(
        medicaid=dl.load_medicaid_benefits(), plans=dl.load_ma_plans(),
        plan_counties=dl.load_ma_plan_counties(), plan_shares=plan_shares(),
        sites=dl.load_hc_sites(), uds=dl.load_hc_uds(), nursing_homes=nursing_homes,
    )


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
with head_r:
    slider_col, play_col = st.columns([4, 1], vertical_alignment="bottom")
    # Moving the slider by hand stops playback, so the next tick doesn't jump past the pick.
    year = slider_col.select_slider("Analysis year (BRFSS survey year)", options=years, key="year",
                                    on_change=lambda: st.session_state.update(playing=False))
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

PROFILE_TAB = "🏥 County profile"

# County profile defaults to the alphabetically first state and county (Alabama,
# Autauga); a manual pick or map click is kept while that county has data this year.
if st.session_state.get("profile_fips") not in set(year_all["fips"]):
    first = year_all.sort_values(["state_name", "county"]).iloc[0]
    st.session_state.update(profile_fips=first["fips"], profile_state=first["state"])

stats = model.summarize(year_df)
prev_stats = model.summarize(prev_df) if prev_df is not None and len(prev_df) else None

ui.render_kpis(stats, prev_stats, f"{region_label}, {year}", prev_years[-1] if prev_years else None)
if year_df.empty:
    st.warning("No counties match the current filters.")
    st.stop()

tab_map, tab_profile, tab_time, tab_model, tab_table = st.tabs(
    ["🗺️ Spatial risk map", PROFILE_TAB, "📈 Longitudinal & policy view",
     "🧮 Predictive analytics", "📋 Priority regions"],
    key="main_tab", on_change="rerun",  # keyed so a map click can switch to the profile
)

# --------------------------------------------------------------------------- #
# Primary view — spatial map
# --------------------------------------------------------------------------- #
if tab_map.open:  # hidden tabs are skipped; see KEEP_WIDGET_STATE
    with tab_map:
        overlay = controls.map_mode == "overlay"
        title_col, geo_col = st.columns([3, 1], vertical_alignment="bottom")
        title_col.markdown(
            f"#### {'Cognitive & sensory overlay' if overlay else COMPOSITE_LABEL} ({year})"
        )
        by_state = ui.render_geography_toggle(geo_col) == "state"
        # A looked-up address zooms to its state unless the user has chosen a region.
        if pin and pin["state"] and not filtered:
            view = ui.view_state(set(year_all.loc[year_all["state"] == pin["state"], "fips"]), True)
        else:
            view = ui.view_state(selected_fips, filtered)
        highlight = ui.highlight_layers(year, selected_fips, filtered, pin)
        if by_state:
            states = state_panel(VALUE_TYPE, tuple(sorted(weights.items())))
            geojson, map_panel = dl.state_geojson(year), states
            map_year = states[states["year"] == year]
            map_selected = {f[:2] for f in selected_fips}
            # States aren't profiled, so the state map has no click-through.
            map_kw = {"key": "risk_map_state", "unit": "states", "markers": dl.small_state_points(year)}
        else:
            geojson, map_panel, map_year, map_selected = dl.year_geojson(year), panel, year_all, selected_fips
            # Clicking a county opens it in the County profile tab.
            map_key = f"risk_map_{st.session_state.get('_map_nonce', 0)}"
            map_kw = {"key": map_key,
                      "on_click": partial(ui.open_profile_on_click, map_key, "main_tab", PROFILE_TAB)}
        if overlay:
            if not controls.overlays:
                st.info("Select at least one layer in the sidebar.")
            if by_state:
                # Patterns are drawn over each selected county, carrying its state's values.
                values = map_year.set_index("fips")[[*SENSORY_MEASURES]]
                counties = year_all.loc[year_all["fips"].isin(selected_fips), ["fips", "lat", "lon"]]
                map_kw["pattern_df"] = counties.join(values, on=counties["fips"].str[:2])
            ui.render_overlay_map(geojson, map_year, map_panel, map_selected, view,
                                  controls.overlays, controls.pattern_top_pct, pin, highlight, **map_kw)
        else:
            ui.render_sori_map(geojson, map_year, map_panel, map_selected, view, pin, highlight, **map_kw)

        teeth_year = year_all["teethlost_source_year"].dropna()
        if len(teeth_year) and int(teeth_year.iloc[0]) != year:
            st.caption(
                f"Complete tooth loss is surveyed every two years; {year} uses "
                f"{int(teeth_year.iloc[0])} estimates."
            )

        w_txt = ", ".join(f"{MEASURE_LABELS[m]} {weights[m]:.0%}" for m in SENSORY_MEASURES)
        st.caption(f"{COMPOSITE_SHORT} weights ({controls.weight_mode.lower()}): {w_txt}")
        if by_state:
            st.caption("Switch to County to click a county and open its profile.")
        else:
            st.caption("Click a county to open it in the County profile tab.")
        # Right after the map, so the zoom box doesn't wait for the other tabs on a cold start.
        ui.install_map_scripts()

# --------------------------------------------------------------------------- #
# County profile — sensory burden and access to preventative care
# --------------------------------------------------------------------------- #
if tab_profile.open:  # hidden tabs are skipped; see KEEP_WIDGET_STATE
    with tab_profile:
        profile_fips = ui.render_county_picker(year_all, pin)
        ui.render_county_profile(profile_fips, panel, year, county_access())

# --------------------------------------------------------------------------- #
# Secondary view — longitudinal & policy
# --------------------------------------------------------------------------- #
if tab_time.open:  # hidden tabs are skipped; see KEEP_WIDGET_STATE
    with tab_time:
        left, right = st.columns([3, 2])
        with left:
            trend = model.national_trend(region)
            st.plotly_chart(
                ui.trend_chart(trend, year, f"Prevalence trend in {region_label}"),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                ui.score_trend_chart(panel, selected_fips if filtered else set(),
                                     f"{COMPOSITE_SHORT} distribution over time"),
                width="stretch",
            )
        st.markdown("##### Policy & evidence milestones")
        state_events = dl.load_policy_events()
        if controls.states:
            state_events = state_events[state_events["state"].isin(controls.states)]
        st.caption(f"US milestones are numbered markers on the trend chart above, with dotted lines coloured by "
                   f"sensory domain. State Medicaid milestones for {region_label} are listed in the table only.")
        ui.render_policy_timeline(state_events,
                                  dict(zip(dl.state_options()["state"], dl.state_options()["state_name"])))

        if len(years) > 1:
            st.markdown(f"##### Largest increases in {COMPOSITE_LABEL}, {years[0]} → {years[-1]}")
            change = model.county_change(region, COMPOSITE, years[0], years[-1])
            change_col = f"Change ({years[-1]} vs {years[0]})"
            st.dataframe(
                change.sort_values("change", ascending=False).head(15)
                .rename(columns={"county": "County", "state": "State", "start": str(years[0]),
                                 "end": str(years[-1]), "change": change_col})
                .drop(columns="fips"),
                hide_index=True, width="stretch",
                column_config={
                    str(years[0]): st.column_config.NumberColumn(format="%.0f"),
                    str(years[-1]): st.column_config.NumberColumn(format="%.0f"),
                    change_col: st.column_config.NumberColumn(
                        format="%+.0f",
                        help=f"{COMPOSITE_SHORT} {years[-1]} − {COMPOSITE_SHORT} {years[0]}. Example: 40 → 55 = +15, "
                             "i.e. the county moved 15 places up the 0–100 burden ranking."),
                },
            )

# --------------------------------------------------------------------------- #
# Predictive analytics engine
# --------------------------------------------------------------------------- #
if tab_model.open:  # hidden tabs are skipped; see KEEP_WIDGET_STATE
    with tab_model:
        if fit is None:
            st.info("Not enough data to fit the association model for this year.")
        else:
            st.markdown(
                f"**Association model ({year}, national, {VALUE_TYPE_LABEL.lower()}):** "
                f"Cognitive decline ~ Vision + Hearing + Complete tooth loss; "
                f"n = {fit.n:,} counties, R² = {fit.r2:.2f}"
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
if tab_table.open:  # hidden tabs are skipped; see KEEP_WIDGET_STATE
    with tab_table:
        t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
        t1.markdown(f"#### Intervention priority in {region_label} ({year})")
        top_n = t2.selectbox("Show Top", [25, 50, 100, 250, 500], index=1, key="top_n")
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

# After the map has been sent, so it doesn't delay the first screen (~0.8 s on a cold
# start); the first County profile open then takes ~0.6 s instead of ~1.3 s.
if tab_map.open:
    warm_profile(panel)

# Continue timeline playback after the page has rendered.
if st.session_state.get("playing"):
    time.sleep(PLAY_INTERVAL_S)
    st.rerun()
