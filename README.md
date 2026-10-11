# Cognimap IQ

An interactive web GIS dashboard showing how cognitive decline relates to three sensory impairments (vision, hearing and mastication) across US counties, and what preventative and dementia care is available locally. It is built for public health policy makers, health authority analysts and budget planners who want evidence for moving budgets from reactive care (long-term care, emergency admissions) toward low-cost preventative sensory care during the intervention window of opportunity.

This prototype (PoC) uses **CDC PLACES** county estimates as a stand-in for Canadian longitudinal data (CCHS, CLSA). Background and goals are in [`project_architecture.md`](project_architecture.md), the technical design is in [`architecture.md`](architecture.md), and coding guidelines are in [`claude.md`](claude.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate               # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/preprocess.py         # CDC PLACES + county boundaries (about 1 minute)
python scripts/preprocess_access.py  # care-access and nursing home data (a few minutes, ~470 MB download)
streamlit run app.py                 # http://localhost:8501
```

The processed snapshot in `data/processed/` is committed, so the app runs without the preprocessing steps. Raw downloads go to `data/raw/` (git-ignored). If the care-access files are missing, the County profile shows how to build them and the rest of the app still works.

## What the dashboard shows

| Tab | What it shows |
|---|---|
| **Spatial risk map** (primary view) | Pydeck/WebGL county map in two modes. **Sensory Organ Risk Index (SORI)** is a choropleth. **Cognitive & sensory overlay** shows cognitive decline as a blue fill with sensory layers on top: vision as orange hatching (╱), hearing as teal hatching (╲) and complete tooth loss as black dots, marking counties in the top N% chosen in the sidebar. Colour classes are pooled across years, so colour changes over time reflect real change. Hovering shows a county tooltip; **clicking a county opens it in the County profile**. |
| **County profile** | One county at a time (map click, picker, or the address lookup; defaults to the top county in Intervention priority). SORI and the four prevalence measures with differences vs state and vs US; three charts (prevalence trend, cognitive burden vs. antipsychotic use across the state's counties, and an estimated summary of all health center dental and vision services in the county). Then **Nursing homes & dementia care**, **Health centers (HRSA)**, **State Medicaid adult benefits** and **Medicare Advantage plans**, described below. |
| **Longitudinal & policy view** (secondary view) | Prevalence trend with numbered policy and evidence milestones, SORI distribution over time, the policy timeline, and the counties whose SORI rose the most. The year slider in the header has ▶ playback. |
| **Predictive analytics** | Association model, Cognitive decline ~ Vision + Hearing + Complete tooth loss (OLS, coefficients, standardised β, R²), and a preventative-care scenario that estimates adults with cognitive difficulty potentially averted by reducing sensory impairment. |
| **Priority regions** | Counties ranked by SORI for intervention (Show Top 25 to 500), with a CSV export. |

**Sidebar:** health region (states), US address lookup (resolves to a county, pins it on the map and shows its national percentiles), map mode and overlay layers, pattern threshold (top 10 to 50%), and SORI weighting (equal or custom).

### County profile panels

- **Nursing homes & dementia care (CMS + ACS):** share of adults 65+ with a cognitive difficulty, nursing homes, certified beds, beds per 100 at-risk seniors, average star rating; an automatic reading that flags a *dementia-care priority* (above-median burden with above-average antipsychotic use) or a *capacity gap*; and a table of resident-weighted dementia-care indicators (antipsychotic, antianxiety/hypnotic use, restraints, depressive symptoms, ADL decline, falls with major injury, F744/F758/F605 citations, star rating, RN hours) laid out as County | vs State | State | vs US | US, with worse cells in red and better cells in green.
- **Health centers (HRSA):** site counts, UDS 2020 to 2025 services by organization (pick a service to compare organizations year by year with a sparkline and change since the first published year, or open one organization's full history), and a site map and list.
- **State Medicaid adult benefits (KFF):** whether adult dental, eyeglasses and hearing aids are covered, with copays, limits and prior approval.
- **Medicare Advantage plans (CMS PBP 2026):** individual plans offered in the county and the share offering dental, comprehensive dental, hearing aids and eyewear, against state and US county medians, plus the full plan list.

Every `?` tooltip that involves arithmetic gives the formula and a worked example, using the selected county's own numbers where possible. Every change label names its reference, for example `+1.2 pp (vs MA)`, `-14 (vs US)` or `+0.2 pp (vs 2022)`.

## Data sources

| Source | Used for | Vintage |
|---|---|---|
| [CDC PLACES](https://www.cdc.gov/places/) county data (Socrata API) | VISION, HEARING, TEETHLOST, COGNITION prevalence (age-adjusted) | BRFSS 2021 to 2023 (PLACES 2023 to 2025 releases) |
| US Census cartographic boundaries 1:20M | County polygons (EPSG:4326) | 2020 and 2023 |
| US Census Geocoder | Address to county lookup (live, on demand) | current |
| [KFF Medicaid Benefits Database](https://www.kff.org/medicaid/state-indicator/dental-services/) | State adult dental, eyeglasses, hearing-aid coverage | 2018 survey |
| [CMS Plan Benefit Package](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/benefits-data) | Medicare Advantage plans, benefits, service areas | contract year 2026 |
| [HRSA Data Downloads](https://data.hrsa.gov/data/download) | Health center service delivery sites; UDS awardee tables 5 and 6A | sites current; UDS 2020 to 2025 |
| [CMS Care Compare nursing homes](https://data.cms.gov/provider-data/topics/nursing-homes) | Provider information, MDS quality measures, health citations | current release (MDS 2025Q2 to 2026Q1) |
| [ACS 5-year table B18104](https://www.census.gov/programs-surveys/acs) | Adults 65+ with a cognitive difficulty | 2019 to 2023 |

## Methodology in brief

- **Measures:** `VISION`, `HEARING` and `TEETHLOST` (complete tooth loss, adults 65+, a mastication proxy) are the sensory measures; `COGNITION` is the outcome. Rates are age-adjusted; `TEETHLOST` is biennial and carried forward.
- **SORI (0 to 100):** each sensory rate becomes a z-score pooled over all years, the z-scores are averaged with the chosen weights, and the average is ranked against every county-year. Scores are therefore comparable across years. Tiers: Critical ≥ 90, High ≥ 75, Elevated ≥ 50.
- **Care access:** Medicare Advantage counts cover individual (non-employer) plans including special needs plans. UDS reports per organization, so the county summary allocates each organization's count by its share of sites in the county and marks years with unpublished counts. Nursing home measures are weighted by average daily residents; citations cover the last three inspection cycles.
- **Caveat:** all comparisons are ecological and cross-sectional. They show where to look and prioritise, not causal effects on individuals. ACS counts self- or proxy-reported cognitive difficulty, not diagnosed dementia.

See [`architecture.md`](architecture.md) for the full data model, algorithms and known limitations.

## Project layout

```
app.py                        Streamlit entry point: layout, tabs, orchestration, cached analytics
src/config.py                 Paths, measure metadata, palettes, policy timeline, indicator labels
src/data_loader.py            @st.cache_data loaders, per-year geometry, outlines, geocoder, map patterns
src/model.py                  SORI, OLS model, scenario, summaries, plan / UDS / nursing home aggregates
src/components.py             Sidebar, KPI cards, Pydeck map, Plotly charts, tables, County profile
scripts/preprocess.py         CDC PLACES + Census boundaries -> Parquet
scripts/preprocess_access.py  KFF, CMS PBP, HRSA sites + UDS, CMS nursing homes, ACS -> Parquet
data/processed/               Parquet snapshot read by the app (committed)
data/raw/                     Downloaded source files (git-ignored)
assets/favicon.png            App icon (favicon and title)
.streamlit/config.toml        Theme and server settings
```
