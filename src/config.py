"""Static configuration: paths, measure metadata, palettes and policy timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
PLACES_PATH = DATA_DIR / "places_county.parquet"
COUNTIES_PATH = DATA_DIR / "counties.parquet"
FAVICON_PATH = ROOT / "assets" / "favicon.png"
# Care-access snapshot (scripts/preprocess_access.py)
MEDICAID_PATH = DATA_DIR / "medicaid_benefits.parquet"
MA_PLANS_PATH = DATA_DIR / "ma_plans.parquet"
MA_PLAN_COUNTIES_PATH = DATA_DIR / "ma_plan_counties.parquet"
HC_SITES_PATH = DATA_DIR / "hc_sites.parquet"
HC_UDS_PATH = DATA_DIR / "hc_uds.parquet"
NH_FACILITIES_PATH = DATA_DIR / "nh_facilities.parquet"
SENIOR_COGNITION_PATH = DATA_DIR / "senior_cognition.parquet"

CRS = "EPSG:4326"

SENSORY_MEASURES = ["VISION", "HEARING", "TEETHLOST"]
OUTCOME_MEASURE = "COGNITION"
ALL_MEASURES = [*SENSORY_MEASURES, OUTCOME_MEASURE]

MEASURE_LABELS = {
    "VISION": "Vision impairment",
    "HEARING": "Hearing impairment",
    "TEETHLOST": "Complete tooth loss (65+)",
    "COGNITION": "Cognitive decline",
}
MEASURE_DESCRIPTIONS = {
    "VISION": "Blindness or serious difficulty seeing, even with glasses (adults 18+).",
    "HEARING": "Deafness or serious difficulty hearing (adults 18+).",
    "TEETHLOST": "All natural teeth lost, used as a mastication proxy (adults 65+).",
    "COGNITION": "Serious difficulty concentrating, remembering or making decisions (adults 18+).",
}

# Age-adjusted rates compare regions fairly despite different age structures.
VALUE_TYPE = "AgeAdjPrv"
VALUE_TYPE_LABEL = "Age-adjusted prevalence"

COMPOSITE = "COMPOSITE_RISK"
COMPOSITE_LABEL = "Sensory Organ Risk Index (SORI)"
COMPOSITE_SHORT = "SORI"

RISK_TIERS = [  # (lower bound on 0–100 score, label)
    (90, "Critical"),
    (75, "High"),
    (50, "Elevated"),
    (0, "Baseline"),
]

# Sequential blue ramp (light -> dark) for magnitude.
SEQUENTIAL_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
# Overlay view: cognitive decline as a lighter blue fill (so patterns stay
# legible on top), sensory measures as hatching / dots.
COGNITION_RAMP = ["#e3eefc", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6"]
NEUTRAL_FILL = "#f4f3ef"
# Stripe hues (ColorBrewer Dark2) stay distinct from each other and from the blue fill
# under protan/deutan simulation.
OVERLAY_STYLES = {
    "VISION": {"pattern": "ne", "color": "#d95f02"},       # orange, rising to upper right
    "HEARING": {"pattern": "nw", "color": "#1b9e77"},      # teal, rising to upper left
    "TEETHLOST": {"pattern": "dots", "color": "#0b0b0b"},  # black dots
}
# Screen-space pattern geometry: 2 px stripes every 6 px; small dots on a staggered grid.
PATTERN_SPACING_PX = 6
PATTERN_LINE_PX = 2
PATTERN_OPACITY = 0.95
DOT_RADIUS_PX = 0.9
# Dot spacing widens on wide views so dots stay legible and the point count stays bounded:
# DOT_SPACING_PX[0] at DOT_SPACING_ZOOMS[0] (whole US) easing to [1] at [1] (state level).
DOT_SPACING_PX = (7, 4)
DOT_SPACING_ZOOMS = (3.5, 7.0)
# Patterns mark counties in the top N% of each sensory measure (sidebar choice).
PATTERN_TOP_SHARES = [10, 20, 25, 30, 50]
PATTERN_TOP_DEFAULT = 25
NO_DATA_COLOR = "#d9d8d2"

# Fixed categorical order (identity follows the measure, never its rank). Charts reuse the
# overlay map's hues so a measure looks the same everywhere.
MEASURE_COLORS = {
    "VISION": OVERLAY_STYLES["VISION"]["color"],
    "HEARING": OVERLAY_STYLES["HEARING"]["color"],
    "TEETHLOST": OVERLAY_STYLES["TEETHLOST"]["color"],
    "COGNITION": COGNITION_RAMP[-1],
}
# The composite index keeps the SORI map's navy.
COMPOSITE_COLOR = SEQUENTIAL_RAMP[5]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


# County profile: care-access measures and their sources.
MEDICAID_BENEFITS = {  # KFF Medicaid Benefits Database topic -> label
    "dental": "Adult dental",
    "vision": "Eyeglasses",
    "hearing": "Hearing aids",
}
PLAN_BENEFITS = {  # CMS PBP supplemental benefit flag -> label
    "dental_any": "Dental (preventive or comprehensive)",
    "dental_comprehensive": "Comprehensive dental",
    "hearing_aids": "Hearing aids",
    "eyewear": "Eyewear",
}
UDS_SERVICES = {  # HRSA UDS column -> label
    "dental_patients": "Dental patients",
    "dental_visits": "Dental visits",
    "oral_exam_patients": "Oral exams",
    "sealant_patients": "Sealants",
    "fluoride_patients": "Fluoride",
    "vision_patients": "Vision patients",
    "vision_visits": "Vision visits",
    "eye_exam_patients": "Eye exams",
}
# Nursing home dementia-care indicators (CMS Care Compare). `True` = higher is better.
NH_MEASURES = {
    "ls_antipsychotic": ("Long-stay residents on antipsychotics", False),
    "ss_new_antipsychotic": ("Short-stay residents newly on antipsychotics", False),
    "ls_antianxiety_hypnotic": ("Long-stay residents on antianxiety / hypnotic drugs", False),
    "ls_restraint": ("Long-stay residents physically restrained", False),
    "ls_depressive": ("Long-stay residents with depressive symptoms", False),
    "ls_adl_decline": ("Long-stay residents needing more help with daily activities", False),
    "ls_falls_major_injury": ("Long-stay residents with a fall causing major injury", False),
}
NH_CITATIONS = {  # survey citations per nursing home, last three inspection cycles
    "cit_dementia_care": "Dementia care citations (F744)",
    "cit_psychotropic_gdr": "Psychotropic dose-reduction citations (F758)",
    "cit_chemical_restraint": "Unnecessary psychotropic citations (F605)",
}
NH_OTHER = {
    "overall_rating": ("Overall star rating (1–5)", True),
    "rn_hprd": ("RN hours per resident per day", True),
}

ACCESS_SOURCES = {
    "medicaid": ("KFF Medicaid Benefits Database", "https://www.kff.org/medicaid/state-indicator/dental-services/"),
    "pbp": ("CMS Medicare Advantage Plan Benefit Package (PBP)",
            "https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/benefits-data"),
    "sites": ("HRSA Health Center Service Delivery Sites", "https://data.hrsa.gov/data/download"),
    "uds": ("HRSA Uniform Data System (UDS)", "https://data.hrsa.gov/data/download"),
    "nh": ("CMS Care Compare nursing homes (Provider Data Catalog)",
           "https://data.cms.gov/provider-data/topics/nursing-homes"),
    "acs": ("US Census ACS 5-year, table B18104", "https://www.census.gov/programs-surveys/acs"),
}


@dataclass(frozen=True)
class PolicyEvent:
    date: str          # ISO date
    domain: str        # Vision / Hearing / Mastication / Cognition / Evidence
    title: str
    detail: str


# Policy and evidence milestones shown on the longitudinal timeline. These are
# national (US) reference events for the PoC; swap for provincial / health-region
# interventions when migrating to Canadian data.
POLICY_EVENTS = [
    PolicyEvent(
        "2020-07-30", "Evidence",
        "Lancet Commission: hearing loss as a modifiable dementia risk",
        "2020 Commission update lists hearing loss as the largest potentially "
        "modifiable mid-life risk factor for dementia.",
    ),
    PolicyEvent(
        "2022-08-16", "Hearing",
        "FDA final rule: over-the-counter hearing aids",
        "Creates an OTC hearing-aid category for adults with perceived mild-to-"
        "moderate hearing loss (effective Oct 17, 2022), lowering access cost.",
    ),
    PolicyEvent(
        "2022-11-01", "Mastication",
        "CMS CY2023 fee schedule: medically necessary dental services",
        "Clarifies Medicare Part A/B payment for dental services inextricably "
        "linked to covered medical treatment (from Jan 1, 2023).",
    ),
    PolicyEvent(
        "2023-07-18", "Evidence",
        "ACHIEVE trial results published",
        "Hearing intervention slowed 3-year cognitive decline in older adults at "
        "higher risk of dementia (secondary-population finding).",
    ),
    PolicyEvent(
        "2024-07-31", "Evidence",
        "Lancet Commission adds untreated vision loss",
        "2024 update adds untreated vision loss and high LDL cholesterol, "
        "bringing modifiable risk factors to 14.",
    ),
]

DOMAIN_COLORS = {
    "Vision": MEASURE_COLORS["VISION"],
    "Hearing": MEASURE_COLORS["HEARING"],
    "Mastication": MEASURE_COLORS["TEETHLOST"],
    "Cognition": MEASURE_COLORS["COGNITION"],
    "Evidence": INK_MUTED,
}
