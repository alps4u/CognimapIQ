"""Build the local Parquet / GeoParquet snapshot used by the dashboard.

Downloads CDC PLACES county releases (Socrata API) and US Census cartographic
county boundaries, then writes:

    data/processed/places_county.parquet   tidy wide table, one row per county-year-value type
    data/processed/counties.parquet        county polygons (GeoParquet, EPSG:4326)
    data/processed/county_shapes.parquet   map-ready county set per analysis year
    data/processed/state_shapes.parquet    counties dissolved per state and analysis year

Run once (or whenever you want to refresh the snapshot):

    python scripts/preprocess.py
    python scripts/preprocess.py --shapes-only   # rebuild the two shape files from the files above
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import county_features, dissolve_states, snap, stable_order  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

MEASURES = ["VISION", "HEARING", "TEETHLOST", "COGNITION"]
VALUE_TYPES = ["AgeAdjPrv", "CrdPrv"]

# PLACES county releases that carry all four measures. The disability measures
# (VISION, HEARING, COGNITION) are annual; TEETHLOST is biennial, so each release
# reports the most recent available TEETHLOST year (carried forward).
RELEASES = {
    # analysis year: (Socrata dataset id, release label)
    2021: ("h3ej-a9ec", "PLACES 2023 release"),
    2022: ("fu4u-a9bh", "PLACES 2024 release"),
    2023: ("swc5-untb", "PLACES 2025 release"),
}

# Census cartographic boundary files (1:20M). 2020 vintage carries the legacy
# Connecticut counties used by older PLACES releases; 2023 vintage carries the
# Connecticut planning regions used by newer releases.
BOUNDARIES = {
    2020: "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip",
    2023: "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip",
}

SOCRATA = "https://data.cdc.gov/resource/{id}.json"


def fetch_release(dataset_id: str) -> pd.DataFrame:
    measures = ",".join(f"'{m}'" for m in MEASURES)
    params = {
        "$where": f"measureid in({measures})",
        "$limit": 100000,
    }
    resp = requests.get(SOCRATA.format(id=dataset_id), params=params, timeout=180)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        raise RuntimeError(f"No rows returned for dataset {dataset_id}")
    return df


def tidy_release(raw: pd.DataFrame, year: int, label: str) -> pd.DataFrame:
    df = raw[raw["datavaluetypeid"].isin(VALUE_TYPES)].copy()
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df["totalpopulation"] = pd.to_numeric(df["totalpopulation"], errors="coerce")
    # Adult (18+) population is only published from the 2024 release onward.
    if "totalpop18plus" not in df.columns:
        df["totalpop18plus"] = pd.NA
    df["totalpop18plus"] = pd.to_numeric(df["totalpop18plus"], errors="coerce")
    coords = df["geolocation"].apply(
        lambda g: g.get("coordinates") if isinstance(g, dict) else [None, None]
    )
    df["lon"] = coords.str[0].astype(float)
    df["lat"] = coords.str[1].astype(float)

    teeth_year = (
        df.loc[df["measureid"] == "TEETHLOST", "year"].astype(int).max()
        if (df["measureid"] == "TEETHLOST").any()
        else pd.NA
    )

    wide = df.pivot_table(
        index=["locationid", "datavaluetypeid"],
        columns="measureid",
        values="data_value",
        aggfunc="first",
    ).reset_index()

    attrs = (
        df.sort_values("measureid")
        .groupby("locationid")
        .agg(
            county=("locationname", "first"),
            state=("stateabbr", "first"),
            state_name=("statedesc", "first"),
            population=("totalpopulation", "max"),
            adult_population=("totalpop18plus", "max"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
    )

    out = wide.merge(attrs, on="locationid", how="left").rename(
        columns={"locationid": "fips", "datavaluetypeid": "value_type"}
    )
    out["year"] = year
    out["teethlost_source_year"] = teeth_year
    out["source"] = label
    for m in MEASURES:
        if m not in out.columns:
            out[m] = pd.NA
    cols = [
        "fips", "county", "state", "state_name", "year", "value_type",
        *MEASURES, "population", "adult_population", "lat", "lon", "teethlost_source_year", "source",
    ]
    return out[cols]


def build_places() -> pd.DataFrame:
    frames = []
    for year, (dataset_id, label) in RELEASES.items():
        print(f"  downloading {label} ({dataset_id}) -> analysis year {year}")
        raw = fetch_release(dataset_id)
        frames.append(tidy_release(raw, year, label))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["state"] != "US"]  # drop national aggregate rows if present
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    # Back-fill adult population for releases without it, using each county's
    # adult share from the nearest release that publishes it (national share as a
    # last resort).
    share = (df["adult_population"] / df["population"]).groupby(df["fips"]).transform("mean")
    share = share.fillna((df["adult_population"].sum() / df.loc[df["adult_population"].notna(), "population"].sum()))
    df["adult_population_estimated"] = df["adult_population"].isna()
    df["adult_population"] = df["adult_population"].fillna((df["population"] * share).round())
    df["population"] = df["population"].astype("Int64")
    df["adult_population"] = df["adult_population"].astype("Int64")
    df["teethlost_source_year"] = df["teethlost_source_year"].astype("Int64")
    for m in MEASURES:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def download_boundary(vintage: int, url: str) -> gpd.GeoDataFrame:
    target = RAW_DIR / Path(url).name
    if not target.exists():
        print(f"  downloading {url}")
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    with zipfile.ZipFile(target) as zf:
        zf.testzip()
    gdf = gpd.read_file(f"zip://{target}")
    gdf = gdf[["GEOID", "NAME", "STATEFP", "geometry"]].rename(
        columns={"GEOID": "fips", "NAME": "name", "STATEFP": "statefp"}
    )
    gdf["vintage"] = vintage
    return gdf


def build_counties() -> gpd.GeoDataFrame:
    current = download_boundary(2023, BOUNDARIES[2023])
    legacy = download_boundary(2020, BOUNDARIES[2020])
    # Keep every current feature, plus legacy features whose FIPS no longer exists
    # (legacy Connecticut counties, retired Alaska census areas, ...).
    legacy_only = legacy[~legacy["fips"].isin(current["fips"])]
    gdf = pd.concat([current, legacy_only], ignore_index=True)
    gdf["current"] = gdf["vintage"] == 2023
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=current.crs).to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(0.005, preserve_topology=True)
    return gdf


def write_shapes(counties: gpd.GeoDataFrame, places: pd.DataFrame) -> None:
    """Map shapes per analysis year (each year's county set differs, e.g. Connecticut),
    which the app would otherwise assemble at start-up (~4 s states, ~1 s counties):

    county_shapes: the year's counties in a feature order shared by all years, snapped
                   to the map's coordinate grid, as the county map draws them.
    state_shapes:  the year's counties dissolved into one shape per state.
    """
    places = places.assign(fips=places["fips"].astype(str))
    by_year = {y: county_features(counties, places, y) for y in sorted(places["year"].unique())}
    shapes = {
        "county_shapes": [snap(stable_order(by_year, y, "fips")).assign(year=y) for y in by_year],
        "state_shapes": [dissolve_states(g).assign(year=y) for y, g in by_year.items()],
    }
    for name, frames in shapes.items():
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=counties.crs)
        path = OUT_DIR / f"{name}.parquet"
        gdf.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(ROOT)} ({len(gdf):,} features)")


def main() -> int:
    if "--shapes-only" in sys.argv:
        print("County and state shapes")
        write_shapes(gpd.read_parquet(OUT_DIR / "counties.parquet"),
                           pd.read_parquet(OUT_DIR / "places_county.parquet"))
        return 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] CDC PLACES county measures")
    places = build_places()
    places_path = OUT_DIR / "places_county.parquet"
    places.to_parquet(places_path, index=False)
    print(f"  wrote {places_path.relative_to(ROOT)} ({len(places):,} rows)")

    print("[2/3] County boundaries")
    counties = build_counties()
    counties_path = OUT_DIR / "counties.parquet"
    counties.to_parquet(counties_path, index=False)
    print(f"  wrote {counties_path.relative_to(ROOT)} ({len(counties):,} features)")

    missing = set(places["fips"]) - set(counties["fips"])
    if missing:
        print(f"  note: {len(missing)} PLACES FIPS without boundary: {sorted(missing)[:10]}")

    print("[3/3] County and state shapes")
    write_shapes(counties, places)
    return 0


if __name__ == "__main__":
    sys.exit(main())
