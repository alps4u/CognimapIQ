"""Cached loading of the local Parquet snapshot and per-year geometry assembly."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import streamlit as st

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
