# Cognimap IQ

An interactive web GIS dashboard showing how cognitive decline relates to three sensory impairments: vision, hearing and mastication. It is built for public health policy makers who want evidence for moving budgets from reactive care toward low-cost preventative sensory care.

This prototype (PoC) uses **CDC PLACES** county estimates as a stand-in for Canadian longitudinal data (CCHS, CLSA). See [`architecture.md`](architecture.md) and [`claude.md`](claude.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/preprocess.py         # builds data/processed/*.parquet (one-time, ~1 min)
python scripts/preprocess_access.py  # care-access data for the county profile (~2 min)
streamlit run app.py
```

## Features

| View | What it shows |
|---|---|
| **Spatial risk map** (primary) | Two Pydeck/WebGL views. **Sensory Organ Risk Index (SORI)** shows a county choropleth. **Cognitive & sensory overlay** stacks layers you pick with checkboxes: cognitive decline as a blue fill, vision as red hatching rising to the upper right, hearing as purple hatching rising to the upper left, and complete tooth loss as black dots. Patterns mark counties at or above the 75th percentile of each measure. Colour classes are pooled across years, so colour changes over time reflect real change. |
| **County profile** | One county at a time: click a county on the map, or pick one (defaults to the first county in Intervention priority). It shows SORI and the four prevalence measures against the state average and a trend chart. **Coverage:** whether the state's Medicaid covers adult dental, eyeglasses and hearing aids (KFF Medicaid Benefits Database), and how many individual Medicare Advantage plans in the county offer dental, hearing-aid and eyewear benefits, with state and US medians (CMS Plan Benefit Package). **Health centers:** HRSA service delivery sites in the county on a map and in a list, plus each operating organization's dental and vision services from the UDS for 2020–2025: pick a service to compare organizations year by year (with a trend sparkline and change since the first published year), or pick an organization to see all of its services by year. |
| **Longitudinal & policy view** (secondary) | A year time slider with ▶ playback and animated colour transitions. Trend lines are marked with numbered policy and evidence milestones, alongside the score distribution over time and the counties whose score rose the most. |
| **Predictive analytics** | An association model, Cognition ~ Vision + Hearing + Tooth loss (R², coefficients, standardised β), plus a preventative-care scenario that estimates how many adults with cognitive difficulty might be averted. |
| **Priority regions** | A ranked list of counties for intervention, with a CSV export. |

Sidebar controls: health region (state), map view and overlay layers, and SORI weighting (equal or custom). All rates are age-adjusted prevalence.

## Methodology

- **Measures:** `VISION`, `HEARING` and `TEETHLOST` (complete tooth loss in adults 65+, used as a mastication proxy) are the sensory measures. `COGNITION` is the outcome.
- **Years:** 2021–2023 BRFSS survey years, taken from the PLACES 2023–2025 releases. `TEETHLOST` is only surveyed every two years, so the most recent value is carried forward.
- **Sensory Organ Risk Index (SORI, 0–100):** each sensory measure is converted to a z-score using all years together. The z-scores are combined as a weighted average, and that average is turned into a percentile rank across all years. Because of this, scores can be compared from one year to the next.
- **Care-access data:** KFF's Medicaid benefits survey is from 2018 and covers fee-for-service benefits for categorically needy adults. Medicare Advantage counts use the CMS PBP file for 2026, limited to individual (non-employer) MA plans including special needs plans. CMS lists service areas by SSA county code, so counties are matched to FIPS by state and name; independent cities are told apart from their namesake counties, and Connecticut planning regions have no match. HRSA UDS reports services per organization, not per site, so county pages show each organization's total next to how many of its sites are in the county. The public UDS file withholds some counts (most Table 5 dental and vision totals before 2022), 2019 and earlier files use a different layout and are not included, and UDS has no hearing or audiology service line.
- **Caveat:** these are ecological, cross-sectional associations between counties. They are meant for prioritising regions, not for estimating causal effects on individuals.

## Project layout

```
app.py                  Streamlit entry point (layout & orchestration)
src/config.py           Measure metadata, palettes, policy timeline
src/data_loader.py      @st.cache_data Parquet/GeoParquet loading, per-year geometry
src/model.py            Composite score, OLS association model, scenarios, summaries
src/components.py       Sidebar, KPI cards, Pydeck map, Plotly charts, tables
scripts/preprocess.py   CDC PLACES (Socrata) + Census boundaries -> Parquet (EPSG:4326)
scripts/preprocess_access.py  KFF Medicaid benefits, CMS PBP plans, HRSA sites + UDS -> Parquet
data/processed/         Local snapshot read by the app
```

County boundaries are the US Census 1:20M cartographic files. The 2020 and 2023 vintages are combined so that both the old Connecticut counties and the newer planning regions are covered.
