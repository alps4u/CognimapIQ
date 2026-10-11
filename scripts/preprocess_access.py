"""Build the care-access snapshot used by the county profile view.

Downloads three public sources and writes Parquet files next to the PLACES
snapshot (run scripts/preprocess.py first; county names and FIPS come from it):

    data/processed/medicaid_benefits.parquet  KFF Medicaid Benefits Database: adult dental,
                                              eyeglasses and hearing-aid coverage by state
    data/processed/ma_plans.parquet           CMS PBP: one row per individual Medicare
                                              Advantage plan (segment) with benefit flags
    data/processed/ma_plan_counties.parquet   CMS PBP service areas: (bid_id, fips) pairs
    data/processed/hc_sites.parquet           HRSA health center service delivery sites
    data/processed/hc_uds.parquet             HRSA UDS dental / vision services by awardee and year

    python scripts/preprocess_access.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
PLACES_PATH = OUT_DIR / "places_county.parquet"

HEADERS = {"User-Agent": "Mozilla/5.0 (cognimapIQ data preprocessing)"}


# --------------------------------------------------------------------------- #
# KFF Medicaid Benefits Database (state indicator pages embed their table JSON)
# --------------------------------------------------------------------------- #
KFF_PAGES = {
    "dental": "https://www.kff.org/medicaid/state-indicator/dental-services/",
    "vision": "https://www.kff.org/medicaid/state-indicator/eyeglasses/",
    "hearing": "https://www.kff.org/medicaid/state-indicator/hearing-aids/",
}
KFF_COLUMNS = {
    "Benefit Covered": "covered",
    "Coverage Code": "coverage_code",
    "Copayment Required?": "copay",
    "Limits on Services": "limits",
    "Prior Approval Required": "prior_approval",
}


# --------------------------------------------------------------------------- #
# CMS Medicare Advantage Plan Benefit Package (PBP)
# --------------------------------------------------------------------------- #
PBP_YEAR = 2026
PBP_URL = f"https://www.cms.gov/files/zip/pbp-benefits-{PBP_YEAR}.zip"
PBP_KEY = ["pbp_a_hnumber", "pbp_a_plan_identifier", "segment_id"]
# Medicare Advantage plan types sold to individuals (excludes PDPs, cost plans,
# PACE, fallback and employer-only direct contracts).
MA_PLAN_TYPES = {
    "01": "HMO", "02": "HMO-POS", "04": "Local PPO", "05": "PSO", "07": "MSA",
    "08": "PFFS", "09": "PFFS", "31": "Regional PPO",
    "42": "HMO", "43": "HMO-POS", "44": "Local PPO", "45": "PSO",
}


# --------------------------------------------------------------------------- #
# HRSA health centers
# --------------------------------------------------------------------------- #
HRSA_SITES_URL = "https://data.hrsa.gov/DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.csv"
# Awardee files share these field codes from 2020 on (2019 and earlier use another layout).
UDS_YEARS = range(2020, 2026)
UDS_URL = "https://data.hrsa.gov/DataDownload/StaticDocuments/H80-{year}.xlsx"
UDS_FIELDS = {
    # Table 5: staffing and utilization
    "T5_L19_Cb": "dental_visits",
    "T5_L19_Cb2": "dental_virtual_visits",
    "T5_L19_Cc": "dental_patients",
    "T5_L22d_Cb": "vision_visits",
    "T5_L22d_Cb2": "vision_virtual_visits",
    "T5_L22d_Cc": "vision_patients",
    # Table 6A: selected services (patients)
    "T6a_L26d_Cb": "eye_exam_patients",
    "T6a_L28_Cb": "oral_exam_patients",
    "T6a_L30_Cb": "sealant_patients",
    "T6a_L31_Cb": "fluoride_patients",
    "T6a_L33_Cb": "oral_surgery_patients",
}


def download(url: str, name: str) -> Path:
    target = RAW_DIR / name
    if not target.exists():
        print(f"  downloading {url}")
        resp = requests.get(url, headers=HEADERS, timeout=600)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    return target


# --------------------------------------------------------------------------- #
def _kff_tables(html: str) -> list:
    """The `gdocsObject` array embedded in a KFF state-indicator page:
    [[year, [header, types, *rows]], ...]."""
    marker = '"gdocsObject":'
    start = html.index(marker) + len(marker)
    obj, _ = json.JSONDecoder().raw_decode(html[start:])
    return obj


def build_medicaid(states: pd.DataFrame) -> pd.DataFrame:
    name_to_abbr = dict(zip(states["state_name"], states["state"]))
    frames = []
    for benefit, url in KFF_PAGES.items():
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        tables = _kff_tables(resp.text)
        surveys = [t for t in tables if str(t[0]).isdigit()]  # skip "Notes" / "Sources" tabs
        year, table = max(surveys, key=lambda t: int(t[0]))  # latest survey
        header, rows = table[0], table[2:]
        df = pd.DataFrame([r + [None] * (len(header) - len(r)) for r in rows], columns=header)
        df = df.rename(columns={header[0]: "state_name", **KFF_COLUMNS})
        df = df[df["state_name"].isin(name_to_abbr)]
        df["state"] = df["state_name"].map(name_to_abbr)
        df["benefit"] = benefit
        df["survey_year"] = int(year)
        frames.append(df[["state", "benefit", "survey_year", *KFF_COLUMNS.values()]])
        print(f"  KFF {benefit}: {year} survey, {len(df)} states")
    out = pd.concat(frames, ignore_index=True)
    for c in KFF_COLUMNS.values():
        out[c] = out[c].replace({"-": None, "": None})
    out["covered"] = out["covered"].fillna("NR")
    return out


# --------------------------------------------------------------------------- #
def _norm_county(name: str) -> str:
    s = str(name)
    try:  # repair UTF-8 read as Latin-1 ("DoÃ±a Ana" -> "Doña Ana")
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("saint ", "st ").replace("sainte ", "ste ")
    s = re.sub(r"\b(county|parish|borough|census area|city and borough|municipality|municipio)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def _county_lookup(places: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    latest = places.sort_values("year").drop_duplicates("fips", keep="last")
    lookup: dict[tuple[str, str], list[str]] = {}
    for r in latest.itertuples():
        lookup.setdefault((r.state, _norm_county(r.county)), []).append(r.fips)
    return lookup


def _match_county(lookup: dict[tuple[str, str], list[str]], state: str, name: str) -> str | None:
    """FIPS for a PBP county name. PLACES names independent cities like their
    namesake county ("Fairfax" is both 51059 and 51600), so "X City" picks the
    independent city (county code >= 500) and plain "X" picks the county."""
    exact = lookup.get((state, _norm_county(name)), [])
    if len(exact) == 1:  # includes counties named "... City" (Charles City, Carson City)
        return exact[0]
    is_city = name.strip().lower().endswith(" city")
    base = _norm_county(re.sub(r"(?i)\s+city$", "", name.strip())) if is_city else _norm_county(name)
    candidates = [f for f in lookup.get((state, base), []) if (int(f[2:]) >= 500) == is_city]
    return candidates[0] if len(candidates) == 1 else None


def _read_pbp(zf: zipfile.ZipFile, member: str, **kw) -> pd.DataFrame:
    with zf.open(member) as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1", **kw)
    return df.loc[:, ~df.columns.duplicated()]


def build_ma_plans(places: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zpath = download(PBP_URL, f"pbp-benefits-{PBP_YEAR}.zip")
    with zipfile.ZipFile(zpath) as zf:
        a = _read_pbp(zf, "pbp_Section_A.txt", usecols=[
            *PBP_KEY, "bid_id", "pbp_a_plan_type", "pbp_a_eghp_yn", "pbp_a_special_need_flag",
            "pbp_a_special_need_plan_type", "pbp_a_plan_name", "pbp_a_org_marketing_name",
        ])
        dental = _read_pbp(zf, "pbp_b16_dental.txt")
        vision = _read_pbp(zf, "pbp_b17_eye_exams_wear.txt",
                           usecols=[*PBP_KEY, "pbp_b17a_bendesc_yn", "pbp_b17b_bendesc_yn"])
        hearing = _read_pbp(zf, "pbp_b18_hearing_exams_aids.txt",
                            usecols=[*PBP_KEY, "pbp_b18a_bendesc_yn", "pbp_b18b_bendesc_yn"])
        area = _read_pbp(zf, "PlanArea.txt", usecols=["bid_id", "county", "stcd"])
        region = _read_pbp(zf, "PlanRegionArea.txt", usecols=["bid_id", "region_type", "region"])

    # Supplemental dental: any preventive (16b) or comprehensive (16c) service
    # with a benefit type filled in counts as offered.
    prev = [c for c in dental.columns if c.startswith("pbp_b16b_bendesc_") and c.endswith("_amo")]
    comp = [c for c in dental.columns if c.startswith("pbp_b16c_bendesc_") and c.endswith("_amo")]
    dental = dental[PBP_KEY].assign(
        dental_preventive=dental[prev].notna().any(axis=1),
        dental_comprehensive=dental[comp].notna().any(axis=1),
    )

    plans = (
        a[a["pbp_a_plan_type"].isin(MA_PLAN_TYPES) & a["pbp_a_eghp_yn"].eq("2")]
        .merge(dental, on=PBP_KEY, how="left")
        .merge(vision, on=PBP_KEY, how="left")
        .merge(hearing, on=PBP_KEY, how="left")
    )
    plans = pd.DataFrame({
        "bid_id": plans["bid_id"],
        "plan_name": plans["pbp_a_plan_name"],
        "organization": plans["pbp_a_org_marketing_name"],
        "plan_type": plans["pbp_a_plan_type"].map(MA_PLAN_TYPES),
        "snp": plans["pbp_a_special_need_flag"].eq("1"),
        "dental_preventive": plans["dental_preventive"].fillna(False).astype(bool),
        "dental_comprehensive": plans["dental_comprehensive"].fillna(False).astype(bool),
        "eye_exam": plans["pbp_b17a_bendesc_yn"].eq("1"),
        "eyewear": plans["pbp_b17b_bendesc_yn"].eq("1"),
        "hearing_exam": plans["pbp_b18a_bendesc_yn"].eq("1"),
        "hearing_aids": plans["pbp_b18b_bendesc_yn"].eq("1"),
    })
    plans["dental_any"] = plans["dental_preventive"] | plans["dental_comprehensive"]
    plans["year"] = PBP_YEAR

    # Service areas. PlanArea uses SSA county codes, so match on state + county name.
    lookup = _county_lookup(places)
    local = area[area["bid_id"].isin(plans["bid_id"])].copy()
    local["fips"] = [_match_county(lookup, s, c) for s, c in zip(local["stcd"], local["county"])]
    unmatched = local.loc[local["fips"].isna(), ["stcd", "county"]].drop_duplicates()
    if len(unmatched):
        print(f"  note: {len(unmatched)} PBP counties without a PLACES match "
              f"(e.g. {', '.join((unmatched['county'] + ' ' + unmatched['stcd']).head(8))})")
    pairs = [local.dropna(subset=["fips"])[["bid_id", "fips"]]]

    # Regional PPOs list MA regions ("Louisiana and Mississippi"): expand to every
    # county of each named state.
    regional = region[region["bid_id"].isin(plans["bid_id"]) & region["region_type"].eq("MA")]
    states = places[["state", "state_name", "fips"]].drop_duplicates("fips")
    state_names = sorted(states["state_name"].unique(), key=len, reverse=True)
    for bid, reg in zip(regional["bid_id"], regional["region"]):
        names = []
        for n in state_names:  # longest first, so "West Virginia" is consumed before "Virginia"
            pattern = rf"\b{re.escape(n)}\b"
            if re.search(pattern, reg):
                names.append(n)
                reg = re.sub(pattern, "", reg)
        fips = states.loc[states["state_name"].isin(names), "fips"]
        pairs.append(pd.DataFrame({"bid_id": bid, "fips": fips.to_numpy()}))
    plan_counties = pd.concat(pairs, ignore_index=True).drop_duplicates()
    return plans, plan_counties


# --------------------------------------------------------------------------- #
def build_hc_sites() -> pd.DataFrame:
    path = download(HRSA_SITES_URL, "hrsa_health_center_sites.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df["Health Center Type Description"] != "Administrative"]  # service delivery only
    out = pd.DataFrame({
        "site_name": df["Site Name"],
        "health_center": df["Health Center Name"],
        "bhcmis_id": df["BHCMIS Organization Identification Number"],
        "address": df["Site Address"],
        "city": df["Site City"],
        "state": df["Site State Abbreviation"],
        "zip": df["Site Postal Code"].str[:5],
        "phone": df["Site Telephone Number"],
        "website": df["Site Web Address"],
        "fips": df["State and County Federal Information Processing Standard Code"].str.zfill(5),
        "setting": df["Health Center Service Delivery Site Location Setting Description"],
        "location_type": df["Health Center Location Type Description"],
        "lookalike": df["Health Center Type"].str.contains("Look-Alike", na=False),
        "hours_per_week": pd.to_numeric(df["Operating Hours per Week"], errors="coerce"),
        "lon": pd.to_numeric(df["Geocoding Artifact Address Primary X Coordinate"], errors="coerce"),
        "lat": pd.to_numeric(df["Geocoding Artifact Address Primary Y Coordinate"], errors="coerce"),
    })
    return out.reset_index(drop=True)


def _uds_year(year: int) -> pd.DataFrame:
    path = download(UDS_URL.format(year=year), f"uds_awardees_{year}.xlsx")
    # Table sheets carry field descriptions in row 2; the info sheet does not.
    tables = pd.read_excel(path, sheet_name=["Table5", "Table6A"], dtype=str, engine="openpyxl", skiprows=[1])
    info = pd.read_excel(path, sheet_name="HealthCenterInfo", dtype=str, engine="openpyxl")
    info.columns = info.columns.str.replace(" ", "")  # 2020 uses "BHCMIS ID", "Health Center Name", ...
    info = info[["BHCMISID", "HealthCenterName", "HealthCenterState"]]
    t5 = tables["Table5"][["BHCMISID", *[c for c in UDS_FIELDS if c.startswith("T5_")]]]
    t6 = tables["Table6A"][["BHCMISID", *[c for c in UDS_FIELDS if c.startswith("T6a_")]]]
    df = info.merge(t5, on="BHCMISID", how="left").merge(t6, on="BHCMISID", how="left")
    df = df.rename(columns={"BHCMISID": "bhcmis_id", "HealthCenterName": "health_center",
                            "HealthCenterState": "state", **UDS_FIELDS})
    for c in UDS_FIELDS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")  # "---" = withheld
    df["year"] = year
    print(f"  UDS {year}: {len(df):,} awardees")
    return df


def build_uds() -> pd.DataFrame:
    """One row per awardee and reporting year."""
    return pd.concat([_uds_year(y) for y in UDS_YEARS], ignore_index=True)


def main() -> int:
    if not PLACES_PATH.exists():
        print("Run scripts/preprocess.py first (county names and FIPS come from the PLACES snapshot).")
        return 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    places = pd.read_parquet(PLACES_PATH, columns=["fips", "county", "state", "state_name", "year"])
    states = places[["state", "state_name"]].drop_duplicates()

    def write(df: pd.DataFrame, name: str) -> None:
        path = OUT_DIR / name
        df.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(ROOT)} ({len(df):,} rows)")

    print("[1/4] KFF Medicaid Benefits Database")
    write(build_medicaid(states), "medicaid_benefits.parquet")

    print(f"[2/4] CMS Medicare Advantage PBP {PBP_YEAR}")
    plans, plan_counties = build_ma_plans(places)
    write(plans, "ma_plans.parquet")
    write(plan_counties, "ma_plan_counties.parquet")

    print("[3/4] HRSA health center sites")
    sites = build_hc_sites()
    write(sites, "hc_sites.parquet")

    print(f"[4/4] HRSA UDS {UDS_YEARS[0]}–{UDS_YEARS[-1]} awardee services")
    uds = build_uds()
    write(uds, "hc_uds.parquet")
    linked = sites["bhcmis_id"].isin(uds["bhcmis_id"]).mean()
    print(f"  {linked:.0%} of sites link to a UDS awardee record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
