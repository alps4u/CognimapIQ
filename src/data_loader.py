"""Cached loading of the local Parquet snapshot and per-year geometry assembly."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import streamlit as st
from pyproj import Transformer

from src.config import (
    COUNTIES_PATH,
    CRS,
    HC_SITES_PATH,
    HC_UDS_PATH,
    MA_PLAN_COUNTIES_PATH,
    MA_PLANS_PATH,
    MEDICAID_PATH,
    NH_FACILITIES_PATH,
    PLACES_PATH,
    POLICY_EVENTS_PATH,
    SENIOR_COGNITION_PATH,
    STATE_SHAPES_PATH,
)


class SnapshotMissingError(FileNotFoundError):
    """Raised when the pre-processed snapshot has not been built yet."""


class GeocodeError(RuntimeError):
    """Raised when the address geocoder cannot be reached or answers unexpectedly."""


def _require(path) -> None:
    if not path.exists():
        raise SnapshotMissingError(
            f"{path.name} not found. Build the snapshot first: python scripts/"
            + ("preprocess_access.py" if path in ACCESS_PATHS else "preprocess.py")
        )


@st.cache_data(show_spinner="Loading CDC PLACES snapshot…")
def load_places() -> pd.DataFrame:
    _require(PLACES_PATH)
    df = pd.read_parquet(PLACES_PATH)
    df["fips"] = df["fips"].astype(str)
    return df


@st.cache_data(show_spinner="Loading county boundaries…")
def load_counties() -> gpd.GeoDataFrame:
    _require(COUNTIES_PATH)
    gdf = gpd.read_parquet(COUNTIES_PATH)
    if gdf.crs is None or gdf.crs.to_string() != CRS:
        gdf = gdf.to_crs(CRS)
    return gdf


@st.cache_data(show_spinner=False)
def get_panel(value_type: str) -> pd.DataFrame:
    """All county-years for one prevalence type (age-adjusted or crude)."""
    df = load_places()
    return df[df["value_type"] == value_type].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def available_years() -> list[int]:
    return sorted(load_places()["year"].unique().tolist())


@st.cache_data(show_spinner=False)
def state_options() -> pd.DataFrame:
    df = load_places()[["state", "state_name"]].drop_duplicates()
    return df.sort_values("state_name").reset_index(drop=True)


def _stable_order(by_year: dict[int, gpd.GeoDataFrame], year: int, key: str) -> gpd.GeoDataFrame:
    """One year's shapes in an order that is the same in every year.

    deck.gl's fill-colour transition interpolates vertex by vertex, so a shape has to
    sit at the same place in every year's layer; otherwise the old year's colours land
    on the wrong polygons mid-transition and the map breaks into shards. Shapes shared
    by all years come first in `key` order; shapes that exist or differ only in some
    years (Connecticut's counties vs planning regions) go last, where they can't shift
    the rest.
    """
    wkb = {y: dict(zip(g[key], g.geometry.to_wkb())) for y, g in by_year.items()}
    gdf = by_year[year]
    shared = gdf[key].map(lambda k: all(w.get(k) == wkb[year][k] for w in wkb.values()))
    out = gdf.assign(_late=~shared).sort_values(["_late", key]).drop(columns="_late")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner="Assembling county geometry…")
def year_geometry(year: int) -> gpd.GeoDataFrame:
    """Boundary set for one analysis year, in the same feature order every year."""
    return _stable_order({y: _year_features(y) for y in available_years()}, year, "fips")


@st.cache_data(show_spinner=False)
def _year_features(year: int) -> gpd.GeoDataFrame:
    return county_features(load_counties(), load_places(), year)


def county_features(counties: gpd.GeoDataFrame, places: pd.DataFrame, year: int) -> gpd.GeoDataFrame:
    """Boundary set matching the FIPS scheme used by one analysis year.

    Boundaries changed during the study period (e.g. Connecticut moved from
    counties to planning regions). Features with data that year are always kept;
    features without data are kept only if they are current boundaries that do
    not overlap a feature with data, so retired and replacement units never
    stack on the map.
    """
    gdf = counties
    fips_with_data = set(places.loc[lambda d: d["year"] == year, "fips"])
    has_data = gdf["fips"].isin(fips_with_data)
    with_data = gdf[has_data]
    candidates = gdf[~has_data & gdf["current"]]

    points = gpd.GeoDataFrame(
        candidates[["fips"]], geometry=candidates.geometry.representative_point(), crs=gdf.crs
    )
    overlapping = gpd.sjoin(points, with_data[["geometry"]], predicate="within")["fips"]
    no_data = candidates[~candidates["fips"].isin(overlapping)]
    out = pd.concat([with_data, no_data], ignore_index=True)
    return gpd.GeoDataFrame(out[["fips", "name", "statefp", "geometry"]], crs=gdf.crs)


@st.cache_data(show_spinner=False)
def selection_bounds(fips: tuple[str, ...]) -> tuple[float, float, float, float] | None:
    """(min_lon, min_lat, max_lon, max_lat) of the selected counties' boundaries.

    Longitudes are unwrapped across the antimeridian (Alaska's Aleutians) so the
    extent stays compact; min_lon may then exceed 180.
    """
    gdf = load_counties()
    sel = gdf[gdf["fips"].isin(fips)]
    if sel.empty:
        return None
    coords = sel.geometry.get_coordinates()
    lon, lat = coords["x"].to_numpy(), coords["y"].to_numpy()
    if lon.max() - lon.min() > 180:
        lon = np.where(lon < 0, lon + 360, lon)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


@st.cache_data(show_spinner=False)
def selection_outline(fips: tuple[str, ...], year: int, by_state: bool) -> dict:
    """Outline GeoJSON for highlighting a selection: the selected counties dissolved
    into their states (`by_state`), or the county polygons themselves."""
    gdf = year_geometry(year)
    sel = gdf[gdf["fips"].isin(fips)]
    if by_state:
        states = state_geometry(year)
        sel = states[states["statefp"].isin(sel["statefp"].unique())]
    return json.loads(sel[["geometry"]].to_json(drop_id=True))


@st.cache_data(show_spinner="Assembling state geometry…")
def state_geometry(year: int) -> gpd.GeoDataFrame:
    """The year's state shapes, in the same order every year."""
    return _stable_order({y: _state_shapes(y) for y in available_years()}, year, "statefp")


@st.cache_data(show_spinner=False)
def _load_state_shapes() -> gpd.GeoDataFrame | None:
    return gpd.read_parquet(STATE_SHAPES_PATH) if STATE_SHAPES_PATH.exists() else None


@st.cache_data(show_spinner=False)
def _state_shapes(year: int) -> gpd.GeoDataFrame:
    """Precomputed by scripts/preprocess.py; dissolving here takes ~1 s per year."""
    shapes = _load_state_shapes()
    if shapes is not None and year in set(shapes["year"]):
        return shapes.loc[shapes["year"] == year, ["statefp", "geometry"]].reset_index(drop=True)
    return dissolve_states(year_geometry(year))


def dissolve_states(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Counties dissolved into one shape per state (`statefp`)."""
    # County edges don't meet exactly, so a plain dissolve leaves seam lines inside
    # each state. Grow-union-shrink closes the gaps; dropping holes removes slivers.
    eps = 0.005  # degrees (~500 m)
    statefps, shapes = [], []
    for statefp, g in gdf.groupby("statefp"):
        merged = shapely.union_all(shapely.buffer(g.geometry.values, eps)).buffer(-eps)
        statefps.append(statefp)
        shapes.append(shapely.MultiPolygon(
            [shapely.Polygon(p.exterior) for p in _polygon_parts(merged) if not p.is_empty]
        ).simplify(0.002))
    return gpd.GeoDataFrame({"statefp": statefps}, geometry=shapes, crs=gdf.crs)


@st.cache_data(show_spinner=False)
def small_state_points(year: int, max_km2: float = 1_000) -> dict[str, tuple[float, float]]:
    """(lon, lat) inside states too small to see on a national map (DC), keyed by state FIPS."""
    states = state_geometry(year)
    small = states[states.to_crs("EPSG:5070").area < max_km2 * 1e6]
    return {f: (pt.x, pt.y) for f, pt in zip(small["statefp"], small.geometry.representative_point())}


@st.cache_data(show_spinner=False)
def state_geojson(year: int) -> dict:
    """State shapes as GeoJSON keyed like the county map (`fips` = state FIPS)."""
    states = state_geometry(year).rename(columns={"statefp": "fips"})
    states["name"] = states["fips"]
    return _geojson(states[["fips", "name", "geometry"]])


@st.cache_data(show_spinner=False)
def year_geojson(year: int) -> dict:
    """Year geometry as a plain GeoJSON dict (computed once per year)."""
    return _geojson(year_geometry(year))


# Map coordinates are sent on a 0.001° grid (~110 m). Boundaries are already
# simplified at 0.005°, so this is invisible, and it trims the county map payload.
COORD_GRID = 0.001


def _geojson(gdf: gpd.GeoDataFrame) -> dict:
    snapped = gdf.assign(geometry=shapely.set_precision(gdf.geometry.values, COORD_GRID))
    return json.loads(snapped.to_json(drop_id=True))


# --------------------------------------------------------------------------- #
# Care-access snapshot (Medicaid benefits, Medicare Advantage plans, health centers)
# --------------------------------------------------------------------------- #
NH_PATHS = (NH_FACILITIES_PATH, SENIOR_COGNITION_PATH)
ACCESS_PATHS = (MEDICAID_PATH, MA_PLANS_PATH, MA_PLAN_COUNTIES_PATH, HC_SITES_PATH, HC_UDS_PATH, *NH_PATHS)


def access_snapshot_ready() -> bool:
    return all(p.exists() for p in ACCESS_PATHS if p not in NH_PATHS)


def nh_snapshot_ready() -> bool:
    """Nursing home data arrived later; older snapshots may not have it."""
    return all(p.exists() for p in NH_PATHS)


@st.cache_data(show_spinner=False)
def load_nh_facilities() -> pd.DataFrame:
    _require(NH_FACILITIES_PATH)
    return pd.read_parquet(NH_FACILITIES_PATH)


@st.cache_data(show_spinner=False)
def load_senior_cognition() -> pd.DataFrame:
    _require(SENIOR_COGNITION_PATH)
    return pd.read_parquet(SENIOR_COGNITION_PATH)


@st.cache_data(show_spinner=False)
def load_policy_events() -> pd.DataFrame:
    """State policy milestones; empty when scripts/preprocess_policy.py has not run."""
    if not POLICY_EVENTS_PATH.exists():
        return pd.DataFrame(columns=["date", "scope", "state", "domain", "title", "detail", "source"])
    return pd.read_parquet(POLICY_EVENTS_PATH)


@st.cache_data(show_spinner=False)
def load_medicaid_benefits() -> pd.DataFrame:
    _require(MEDICAID_PATH)
    return pd.read_parquet(MEDICAID_PATH)


@st.cache_data(show_spinner=False)
def load_ma_plans() -> pd.DataFrame:
    _require(MA_PLANS_PATH)
    return pd.read_parquet(MA_PLANS_PATH)


@st.cache_data(show_spinner=False)
def load_ma_plan_counties() -> pd.DataFrame:
    _require(MA_PLAN_COUNTIES_PATH)
    return pd.read_parquet(MA_PLAN_COUNTIES_PATH)


@st.cache_data(show_spinner=False)
def load_hc_sites() -> pd.DataFrame:
    _require(HC_SITES_PATH)
    return pd.read_parquet(HC_SITES_PATH)


@st.cache_data(show_spinner=False)
def load_hc_uds() -> pd.DataFrame:
    _require(HC_UDS_PATH)
    return pd.read_parquet(HC_UDS_PATH)


# --------------------------------------------------------------------------- #
# Address lookup
# --------------------------------------------------------------------------- #
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"


@st.cache_data(show_spinner="Looking up address…", ttl=86_400, max_entries=256)
def geocode_address(address: str, timeout: int = 20) -> dict | None:
    """Resolve a US street address to its county via the US Census Geocoder.

    Returns {matched_address, fips, county_name, lat, lon}, or None when no address matches.
    """
    params = {"address": address, "benchmark": "Public_AR_Current", "vintage": "Current_Current",
              "layers": "Counties", "format": "json"}
    try:
        resp = requests.get(GEOCODER_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        matches = resp.json()["result"]["addressMatches"]
    except requests.RequestException as exc:
        raise GeocodeError(f"Census geocoder unavailable ({exc.__class__.__name__}).") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodeError("Census geocoder returned an unexpected response.") from exc

    for match in matches:
        counties = (match.get("geographies") or {}).get("Counties") or []
        if counties and counties[0].get("GEOID"):
            coords = match.get("coordinates") or {}
            return {
                "matched_address": match.get("matchedAddress", address),
                "fips": str(counties[0]["GEOID"]).zfill(5),
                "county_name": counties[0].get("NAME", ""),
                "lat": coords.get("y"),
                "lon": coords.get("x"),
            }
    return None


# --------------------------------------------------------------------------- #
# Overlay pattern geometry (hatching / dot fills drawn as real geometry)
# --------------------------------------------------------------------------- #
WEB_MERCATOR = "EPSG:3857"
_MERCATOR_WORLD_M = 40_075_016.686


def pattern_spacing_m(zoom: float, px: float) -> float:
    """Projected-metre spacing that renders as roughly `px` pixels at `zoom`.

    Quantised to half zoom levels so cached geometry is reused.
    """
    z = round(zoom * 2) / 2
    return px * _MERCATOR_WORLD_M / (512 * 2**z)  # deck.gl world is 512 px at zoom 0


@st.cache_data(show_spinner=False)
def _counties_mercator() -> gpd.GeoDataFrame:
    return load_counties()[["fips", "geometry"]].to_crs(WEB_MERCATOR)


def _polygon_parts(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def _hatch_part(part, spacing: float, direction: str):
    """Parallel 45° lines clipped to one polygon, aligned to a global grid so
    the pattern runs continuously across neighbouring counties."""
    minx, miny, maxx, maxy = part.bounds
    step = spacing * np.sqrt(2)
    if direction == "ne":  # rising to the upper right: y = x + c
        lo, hi = miny - maxx, maxy - minx
        cs = np.arange(np.floor(lo / step) * step, hi + step, step)
        lines = [((minx, minx + c), (maxx, maxx + c)) for c in cs]
    else:  # rising to the upper left: y = -x + c
        lo, hi = minx + miny, maxx + maxy
        cs = np.arange(np.floor(lo / step) * step, hi + step, step)
        lines = [((minx, c - minx), (maxx, c - maxx)) for c in cs]
    if not lines:
        return []
    clipped = shapely.intersection(shapely.linestrings(lines), part)
    out = []
    for g in clipped:
        if g.is_empty:
            continue
        out.extend(g.geoms if hasattr(g, "geoms") else [g])
    return [g for g in out if g.geom_type == "LineString"]


@st.cache_data(show_spinner=False, max_entries=64)
def hatch_paths(fips: tuple[str, ...], spacing: float, direction: str) -> list[dict]:
    """Hatch lines for the given counties as PathLayer records (EPSG:4326)."""
    gdf = _counties_mercator()
    to_wgs84 = Transformer.from_crs(WEB_MERCATOR, CRS, always_xy=True)
    records = []
    for geom in gdf.loc[gdf["fips"].isin(fips), "geometry"]:
        for part in _polygon_parts(geom):
            for line in _hatch_part(part, spacing, direction):
                xs, ys = to_wgs84.transform(*np.asarray(line.coords).T)
                records.append({"path": np.column_stack([xs, ys]).round(5).tolist()})
    return records


@st.cache_data(show_spinner=False, max_entries=64)
def dot_points(fips: tuple[str, ...], spacing: float) -> pd.DataFrame:
    """Staggered dot grid inside the given counties (EPSG:4326 lon/lat).

    A county smaller than the grid spacing gets one dot at its representative
    point, so every highlighted county stays marked on wide views.
    """
    gdf = _counties_mercator()
    xs_all, ys_all = [], []
    for geom in gdf.loc[gdf["fips"].isin(fips), "geometry"]:
        n_dots = 0
        for part in _polygon_parts(geom):
            minx, miny, maxx, maxy = part.bounds
            ys = np.arange(np.floor(miny / spacing) * spacing, maxy + spacing, spacing)
            for y in ys:
                shift = spacing / 2 if round(y / spacing) % 2 else 0.0
                xs = np.arange(np.floor(minx / spacing) * spacing + shift, maxx + spacing, spacing)
                inside = shapely.contains_xy(part, xs, np.full_like(xs, y))
                xs_all.append(xs[inside])
                ys_all.append(np.full(inside.sum(), y))
                n_dots += int(inside.sum())
        if n_dots == 0:
            pt = geom.representative_point()
            xs_all.append(np.array([pt.x]))
            ys_all.append(np.array([pt.y]))
    if not xs_all:
        return pd.DataFrame(columns=["lon", "lat"])
    to_wgs84 = Transformer.from_crs(WEB_MERCATOR, CRS, always_xy=True)
    lon, lat = to_wgs84.transform(np.concatenate(xs_all), np.concatenate(ys_all))
    return pd.DataFrame({"lon": lon, "lat": lat})
