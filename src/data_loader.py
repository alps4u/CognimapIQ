"""Cached loading of the local Parquet snapshot and per-year geometry assembly."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import streamlit as st
from pyproj import Transformer

from src.config import COUNTIES_PATH, CRS, PLACES_PATH


class SnapshotMissingError(FileNotFoundError):
    """Raised when the pre-processed snapshot has not been built yet."""


def _require(path) -> None:
    if not path.exists():
        raise SnapshotMissingError(
            f"{path.name} not found. Build the snapshot first: python scripts/preprocess.py"
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


@st.cache_data(show_spinner="Assembling county geometry…")
def year_geometry(year: int) -> gpd.GeoDataFrame:
    """Boundary set matching the FIPS scheme used by one analysis year.

    Boundaries changed during the study period (e.g. Connecticut moved from
    counties to planning regions). Features with data that year are always kept;
    features without data are kept only if they are current boundaries that do
    not overlap a feature with data, so retired and replacement units never
    stack on the map.
    """
    gdf = load_counties()
    fips_with_data = set(load_places().loc[lambda d: d["year"] == year, "fips"])
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
def year_geojson(year: int) -> dict:
    """Year geometry as a plain GeoJSON dict (computed once per year)."""
    return json.loads(year_geometry(year).to_json(drop_id=True))


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
    return px * _MERCATOR_WORLD_M / (256 * 2**z)


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
    """Staggered dot grid inside the given counties (EPSG:4326 lon/lat)."""
    gdf = _counties_mercator()
    xs_all, ys_all = [], []
    for geom in gdf.loc[gdf["fips"].isin(fips), "geometry"]:
        for part in _polygon_parts(geom):
            minx, miny, maxx, maxy = part.bounds
            ys = np.arange(np.floor(miny / spacing) * spacing, maxy + spacing, spacing)
            for i, y in enumerate(ys):
                shift = spacing / 2 if round(y / spacing) % 2 else 0.0
                xs = np.arange(np.floor(minx / spacing) * spacing + shift, maxx + spacing, spacing)
                inside = shapely.contains_xy(part, xs, np.full_like(xs, y))
                xs_all.append(xs[inside])
                ys_all.append(np.full(inside.sum(), y))
    if not xs_all:
        return pd.DataFrame(columns=["lon", "lat"])
    to_wgs84 = Transformer.from_crs(WEB_MERCATOR, CRS, always_xy=True)
    lon, lat = to_wgs84.transform(np.concatenate(xs_all), np.concatenate(ys_all))
    return pd.DataFrame({"lon": lon, "lat": lat})
