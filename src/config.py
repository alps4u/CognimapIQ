"""Static configuration: paths, measure metadata, palettes and policy timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
PLACES_PATH = DATA_DIR / "places_county.parquet"
COUNTIES_PATH = DATA_DIR / "counties.parquet"

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
    "TEETHLOST": "All natural teeth lost — mastication proxy (adults 65+).",
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
OVERLAY_STYLES = {
    "VISION": {"pattern": "ne", "color": "#d62828"},       # red, rising to upper right
    "HEARING": {"pattern": "nw", "color": "#7b2cbf"},      # purple, rising to upper left
    "TEETHLOST": {"pattern": "dots", "color": "#0b0b0b"},  # black dots
}
PATTERN_SPACING_PX = 4
# Patterns mark counties at or above this percentile of each sensory measure.
PATTERN_PERCENTILE = 75
NO_DATA_COLOR = "#d9d8d2"

# Fixed categorical order (identity follows the measure, never its rank).
MEASURE_COLORS = {
    "VISION": "#2a78d6",
    "HEARING": "#eb6834",
    "TEETHLOST": "#1baf7a",
    "COGNITION": "#eda100",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


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
