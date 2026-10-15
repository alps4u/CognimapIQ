"""In-memory analytics: Sensory Organ Risk Index (SORI), association model and scenarios.

All statistics here are ecological (county-level prevalence). They describe
spatial association between sensory impairment and cognitive decline and are
meant to prioritise regions for preventative care, not to estimate individual
causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    COMPOSITE,
    NH_CITATIONS,
    NH_MEASURES,
    NH_OTHER,
    OUTCOME_MEASURE,
    PLAN_BENEFITS,
    RISK_TIERS,
    SENSORY_MEASURES,
)


# --------------------------------------------------------------------------- #
# Sensory Organ Risk Index (SORI)
# --------------------------------------------------------------------------- #
def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {m: max(float(weights.get(m, 0.0)), 0.0) for m in SENSORY_MEASURES}
    total = sum(clean.values())
    if total <= 0:
        return {m: 1 / len(SENSORY_MEASURES) for m in SENSORY_MEASURES}
    return {m: w / total for m, w in clean.items()}


def add_composite_score(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Add z-scores, the weighted composite index and the 0–100 Sensory Organ Risk Index (SORI).

    Standardisation and percentile ranking are pooled across all years so a
    county's score is comparable over time (a rising score means its sensory
    burden rose relative to the full study period, not just relative to peers
    in the same year).
    """
    w = normalize_weights(weights)
    out = panel.copy()
    for m in SENSORY_MEASURES:
        mu, sd = out[m].mean(), out[m].std(ddof=0)
        out[f"z_{m}"] = (out[m] - mu) / sd if sd > 0 else 0.0
    zcols = [f"z_{m}" for m in SENSORY_MEASURES]
    complete = out[zcols].notna().all(axis=1)
    index = sum(out[f"z_{m}"] * w[m] for m in SENSORY_MEASURES)
    out["composite_index"] = index.where(complete)
    out[COMPOSITE] = out["composite_index"].rank(pct=True) * 100
    out["risk_tier"] = out[COMPOSITE].apply(risk_tier)
    return out


def risk_tier(score: float) -> str | None:
    if pd.isna(score):
        return None
    for lower, label in RISK_TIERS:
        if score >= lower:
            return label
    return RISK_TIERS[-1][1]


# --------------------------------------------------------------------------- #
# Association model: COGNITION ~ VISION + HEARING + TEETHLOST
# --------------------------------------------------------------------------- #
@dataclass
class ModelFit:
    year: int
    n: int
    intercept: float
    coef: dict[str, float]            # percentage points of COGNITION per 1 pp
    std_err: dict[str, float]
    std_beta: dict[str, float]        # standardised coefficients
    r2: float
    pearson_r: dict[str, float] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pred = self.intercept + sum(df[m] * self.coef[m] for m in SENSORY_MEASURES)
        return pred


def fit_model(df_year: pd.DataFrame, year: int) -> ModelFit | None:
    cols = [*SENSORY_MEASURES, OUTCOME_MEASURE]
    data = df_year[cols].dropna()
    if len(data) < len(SENSORY_MEASURES) + 5:
        return None
    X = data[SENSORY_MEASURES].to_numpy(dtype=float)
    y = data[OUTCOME_MEASURE].to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = len(y) - Xd.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    sx = X.std(axis=0, ddof=0)
    sy = y.std(ddof=0)
    return ModelFit(
        year=year,
        n=len(y),
        intercept=float(beta[0]),
        coef={m: float(b) for m, b in zip(SENSORY_MEASURES, beta[1:])},
        std_err={m: float(s) for m, s in zip(SENSORY_MEASURES, se[1:])},
        std_beta={m: float(b * s / sy) for m, b, s in zip(SENSORY_MEASURES, beta[1:], sx)},
        r2=float(r2),
        pearson_r={m: float(np.corrcoef(data[m], y)[0, 1]) for m in SENSORY_MEASURES},
    )


def add_model_outputs(df: pd.DataFrame, fit: ModelFit | None) -> pd.DataFrame:
    out = df.copy()
    if fit is None:
        out["predicted_cognition"] = np.nan
        out["residual"] = np.nan
        return out
    out["predicted_cognition"] = fit.predict(out)
    out["residual"] = out[OUTCOME_MEASURE] - out["predicted_cognition"]
    return out


# --------------------------------------------------------------------------- #
# Preventative-care scenario
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    baseline_rate: float          # population-weighted predicted prevalence (%)
    scenario_rate: float
    delta_pp: float               # scenario minus baseline (negative = improvement)
    adults_affected_baseline: float
    adults_averted: float
    per_county: pd.DataFrame


def run_scenario(
    df: pd.DataFrame, fit: ModelFit | None, reductions: dict[str, float]
) -> ScenarioResult | None:
    """Association-based what-if: relative reduction in each sensory prevalence.

    `reductions` maps measure -> fraction (0.2 = 20 % relative reduction in that
    county's prevalence). Only positive model coefficients are credited, so a
    negative association never yields a "benefit".
    """
    if fit is None:
        return None
    cols = [*SENSORY_MEASURES, "adult_population"]
    data = df.dropna(subset=cols).copy()
    data = data[data["adult_population"] > 0]
    if data.empty:
        return None
    change = sum(
        -data[m] * reductions.get(m, 0.0) * max(fit.coef[m], 0.0) for m in SENSORY_MEASURES
    )
    data["baseline_pred"] = fit.predict(data).clip(lower=0)
    data["scenario_pred"] = (data["baseline_pred"] + change).clip(lower=0)
    data["adults_averted"] = (
        (data["baseline_pred"] - data["scenario_pred"]) / 100 * data["adult_population"]
    )
    w = data["adult_population"].astype(float)
    base = float(np.average(data["baseline_pred"], weights=w))
    scen = float(np.average(data["scenario_pred"], weights=w))
    return ScenarioResult(
        baseline_rate=base,
        scenario_rate=scen,
        delta_pp=scen - base,
        adults_affected_baseline=float((data["baseline_pred"] / 100 * w).sum()),
        adults_averted=float(data["adults_averted"].sum()),
        per_county=data[["fips", "county", "state", "baseline_pred", "scenario_pred", "adults_averted"]],
    )


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask].astype(float)))


def summarize(df: pd.DataFrame) -> dict:
    """Population-weighted headline statistics for a selection of counties."""
    w = df["adult_population"]
    stats = {
        "counties": int(df["fips"].nunique()),
        "adults": float(w.sum(skipna=True)),
        "score_mean": float(df[COMPOSITE].mean()) if COMPOSITE in df else float("nan"),
        "high_risk_counties": int((df.get(COMPOSITE, pd.Series(dtype=float)) >= 75).sum()),
    }
    for m in [*SENSORY_MEASURES, OUTCOME_MEASURE]:
        stats[m] = weighted_mean(df[m], w)
    cog_rate = stats[OUTCOME_MEASURE]
    stats["adults_cognitive"] = float((df[OUTCOME_MEASURE] / 100 * w).sum(skipna=True))
    if COMPOSITE in df:
        pair = df[[COMPOSITE, OUTCOME_MEASURE]].dropna()
        stats["score_cognition_r"] = (
            float(pair.corr().iloc[0, 1]) if len(pair) > 2 else float("nan")
        )
    stats["cognition_rate"] = cog_rate
    return stats


def state_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per state and year, aggregated from counties as in `summarize`:
    adult-population-weighted prevalence and the simple average of county SORI.
    `fips` holds the two-digit state FIPS so state rows can key a map like counties."""
    df = panel.assign(statefp=panel["fips"].str[:2])
    keys = ["year", "statefp", "state", "state_name"]
    w = df["adult_population"].where(df["adult_population"] > 0)
    g = df.groupby(keys)
    out = g.agg(counties=("fips", "nunique"), adult_population=("adult_population", "sum"),
                **{COMPOSITE: (COMPOSITE, "mean")})
    for m in [*SENSORY_MEASURES, OUTCOME_MEASURE]:
        has = df[m].notna() & w.notna()
        num = (df[m] * w).where(has).groupby([df[k] for k in keys]).sum(min_count=1)
        den = w.where(has).groupby([df[k] for k in keys]).sum(min_count=1)
        out[m] = num / den
    out = out.reset_index().rename(columns={"statefp": "fips"})
    out["county"] = out["state_name"]  # display name, like a county row
    out["risk_tier"] = out[COMPOSITE].apply(risk_tier)
    return out


def priority_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Counties ordered for intervention: highest Sensory Organ Risk Index first."""
    return df.dropna(subset=[COMPOSITE]).sort_values(COMPOSITE, ascending=False)


def national_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted prevalence by year for each measure (long format)."""
    rows = []
    for year, grp in panel.groupby("year"):
        for m in [*SENSORY_MEASURES, OUTCOME_MEASURE]:
            rows.append({"year": year, "measure": m, "value": weighted_mean(grp[m], grp["adult_population"])})
    return pd.DataFrame(rows)


def county_change(panel: pd.DataFrame, col: str, start: int, end: int) -> pd.DataFrame:
    """Per-county change in `col` between two years."""
    a = panel.loc[panel["year"] == start, ["fips", col]].rename(columns={col: "start"})
    b = panel.loc[panel["year"] == end, ["fips", "county", "state", col]].rename(columns={col: "end"})
    out = b.merge(a, on="fips", how="inner")
    out["change"] = out["end"] - out["start"]
    return out


# --------------------------------------------------------------------------- #
# Care access: Medicare Advantage plan benefits and health centers
# --------------------------------------------------------------------------- #
def plan_benefit_shares(plans: pd.DataFrame, plan_counties: pd.DataFrame) -> pd.DataFrame:
    """Per county: number of individual MA plans and the share offering each benefit."""
    df = plan_counties.merge(plans[["bid_id", *PLAN_BENEFITS]], on="bid_id")
    out = df.groupby("fips")[list(PLAN_BENEFITS)].agg(["sum", "mean"])
    out.columns = [f"{b}_{'plans' if agg == 'sum' else 'share'}" for b, agg in out.columns]
    out["plans"] = df.groupby("fips").size()
    return out.reset_index()


def plan_access_summary(shares: pd.DataFrame, fips: str, state_fips: set[str]) -> pd.DataFrame | None:
    """Benefit rows for one county, with state and national medians of county shares."""
    row = shares[shares["fips"] == fips]
    if row.empty:
        return None
    r = row.iloc[0]
    state = shares[shares["fips"].isin(state_fips)]
    return pd.DataFrame([
        {
            "benefit": b,
            "plans": int(r[f"{b}_plans"]),
            "share": float(r[f"{b}_share"]),
            "state_median": float(state[f"{b}_share"].median()),
            "us_median": float(shares[f"{b}_share"].median()),
        }
        for b in PLAN_BENEFITS
    ])


def county_health_centers(sites: pd.DataFrame, fips: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Health center sites in a county, and the organisations operating them with
    how many of their sites fall in the county (UDS reports per organisation, so
    `sites_in_county` / `sites_total` gives the county's share of its sites)."""
    local = sites[sites["fips"] == fips]
    totals = sites.groupby("bhcmis_id").size().rename("sites_total")
    in_county = local.groupby("bhcmis_id").size().rename("sites_in_county")
    orgs = (
        pd.concat([in_county, totals.reindex(in_county.index)], axis=1).reset_index()
        .merge(local.drop_duplicates("bhcmis_id")[["bhcmis_id", "health_center"]], on="bhcmis_id")
    )
    return local, orgs.sort_values("sites_in_county", ascending=False).reset_index(drop=True)


def uds_service_history(orgs: pd.DataFrame, uds: pd.DataFrame, service: str) -> pd.DataFrame:
    """One UDS service for each organisation, one column per reporting year."""
    wide = (
        uds[uds["bhcmis_id"].isin(orgs["bhcmis_id"])]
        .pivot(index="bhcmis_id", columns="year", values=service)
        .reindex(columns=sorted(uds["year"].unique()))
    )
    return orgs.merge(wide, left_on="bhcmis_id", right_index=True, how="left")


def uds_org_history(uds: pd.DataFrame, bhcmis_id: str, services: list[str]) -> pd.DataFrame:
    """All selected UDS services for one organisation: services as rows, years as columns."""
    org = uds[uds["bhcmis_id"] == bhcmis_id].set_index("year")[services]
    return org.reindex(sorted(uds["year"].unique())).T


def uds_county_summary(
    orgs: pd.DataFrame, uds: pd.DataFrame, services: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimated county totals of UDS services by year (services as rows, years as
    columns) and how many of the county's organisations published each value.

    UDS counts are organisation-wide, so each organisation contributes its count
    times the share of its sites located in the county."""
    share = (orgs["sites_in_county"] / orgs["sites_total"]).set_axis(orgs["bhcmis_id"])
    data = uds[uds["bhcmis_id"].isin(share.index)].set_index(["bhcmis_id", "year"])[services]
    years = sorted(uds["year"].unique())
    weighted = data.mul(share, axis=0, level="bhcmis_id").astype(float)
    totals = weighted.groupby(level="year").sum(min_count=1).reindex(years).T
    reporting = data.notna().groupby(level="year").sum().reindex(years, fill_value=0).T
    return totals, reporting


# --------------------------------------------------------------------------- #
# Nursing homes: dementia-care indicators against older adults' cognitive burden
# --------------------------------------------------------------------------- #
NH_WEIGHTED = [*NH_MEASURES, *NH_OTHER]


def _nh_summary(df: pd.DataFrame) -> dict:
    """Capacity, resident-weighted quality measures and citations per facility."""
    w = df["residents"].fillna(df["beds"]).fillna(0)
    out = {"facilities": len(df), "beds": float(df["beds"].sum()), "residents": float(df["residents"].sum())}
    for c in NH_WEIGHTED:
        ok = df[c].notna() & (w > 0)
        out[c] = float((df.loc[ok, c] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
    for c in NH_CITATIONS:
        out[c] = float(df[c].sum() / len(df)) if len(df) else float("nan")
    out["cited_dementia_share"] = float((df["cit_dementia_care"] > 0).mean()) if len(df) else float("nan")
    return out


def nh_summaries(nh: pd.DataFrame, seniors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Nursing home summaries by county and by state, plus the national one, each
    with certified beds per 100 adults 65+ with a cognitive difficulty."""
    nh = nh.dropna(subset=["fips"])
    by_fips = pd.DataFrame({f: _nh_summary(g) for f, g in nh.groupby("fips")}).T
    by_fips = by_fips.join(seniors.set_index("fips")[["pop65", "cog65", "cog65_rate"]], how="outer")
    by_fips["facilities"] = by_fips["facilities"].fillna(0)
    by_fips["beds"] = by_fips["beds"].fillna(0)
    by_fips["beds_per_100_cog65"] = by_fips["beds"] / by_fips["cog65"].astype(float) * 100

    seniors_state = seniors.assign(statefp=seniors["fips"].str[:2])
    state_cog = seniors_state.groupby("statefp")[["pop65", "cog65"]].sum()
    by_state = pd.DataFrame({s: _nh_summary(g) for s, g in nh.groupby(nh["fips"].str[:2])}).T
    by_state = by_state.join(state_cog)
    by_state["cog65_rate"] = by_state["cog65"].astype(float) / by_state["pop65"].astype(float) * 100
    by_state["beds_per_100_cog65"] = by_state["beds"] / by_state["cog65"].astype(float) * 100

    us = _nh_summary(nh)
    us["cog65_rate"] = float(seniors["cog65"].sum() / seniors["pop65"].sum() * 100)
    us["beds_per_100_cog65"] = us["beds"] / float(seniors["cog65"].sum()) * 100
    us["county_median_cog65_rate"] = float(seniors["cog65_rate"].median())
    return by_fips, by_state, us
