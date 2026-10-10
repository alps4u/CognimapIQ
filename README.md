# Cognimap IQ

An interactive web GIS dashboard showing how cognitive decline relates to three sensory impairments: vision, hearing and mastication. It is built for public health policy makers who want evidence for moving budgets from reactive care toward low-cost preventative sensory care.

This prototype (PoC) uses **CDC PLACES** county estimates as a stand-in for Canadian longitudinal data (CCHS, CLSA). See [`architecture.md`](architecture.md) and [`claude.md`](claude.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/preprocess.py      # builds data/processed/*.parquet (one-time, ~1 min)
streamlit run app.py
```

## Features

| View | What it shows |
|---|---|
| **Spatial risk map** (primary) | Two Pydeck/WebGL views. **Sensory Organ Risk Index (SORI)** shows a county choropleth. **Cognitive & sensory overlay** stacks layers you pick with checkboxes: cognitive decline as a blue fill, vision as red hatching rising to the upper right, hearing as purple hatching rising to the upper left, and complete tooth loss as black dots. Patterns mark counties at or above the 75th percentile of each measure. Colour classes are pooled across years, so colour changes over time reflect real change. |
| **Longitudinal & policy view** (secondary) | A year time slider with ▶ playback and animated colour transitions. Trend lines are marked with numbered policy and evidence milestones, alongside the score distribution over time and the counties whose score rose the most. |
| **Predictive analytics** | An association model, Cognition ~ Vision + Hearing + Tooth loss (R², coefficients, standardised β), plus a preventative-care scenario that estimates how many adults with cognitive difficulty might be averted. |
| **Priority regions** | A ranked list of counties for intervention, with a CSV export. |

Sidebar controls: health region (state), map view and overlay layers, and SORI weighting (equal or custom). All rates are age-adjusted prevalence.

## Methodology

- **Measures:** `VISION`, `HEARING` and `TEETHLOST` (complete tooth loss in adults 65+, used as a mastication proxy) are the sensory measures. `COGNITION` is the outcome.
- **Years:** 2021–2023 BRFSS survey years, taken from the PLACES 2023–2025 releases. `TEETHLOST` is only surveyed every two years, so the most recent value is carried forward.
- **Sensory Organ Risk Index (SORI, 0–100):** each sensory measure is converted to a z-score using all years together. The z-scores are combined as a weighted average, and that average is turned into a percentile rank across all years. Because of this, scores can be compared from one year to the next.
- **Caveat:** these are ecological, cross-sectional associations between counties. They are meant for prioritising regions, not for estimating causal effects on individuals.

## Project layout

```
app.py                  Streamlit entry point (layout & orchestration)
src/config.py           Measure metadata, palettes, policy timeline
src/data_loader.py      @st.cache_data Parquet/GeoParquet loading, per-year geometry
src/model.py            Composite score, OLS association model, scenarios, summaries
src/components.py       Sidebar, KPI cards, Pydeck map, Plotly charts, tables
scripts/preprocess.py   CDC PLACES (Socrata) + Census boundaries -> Parquet (EPSG:4326)
data/processed/         Local snapshot read by the app
```

County boundaries are the US Census 1:20M cartographic files. The 2020 and 2023 vintages are combined so that both the old Connecticut counties and the newer planning regions are covered.
