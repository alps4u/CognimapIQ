"""Build the state policy milestones shown in the longitudinal & policy view.

Writes data/processed/policy_events.parquet: one row per state-level policy event,
combining two sources:

    KFF State Health Facts    ACA Medicaid expansion implementation dates (downloaded)
    CHCS / CareQuest / ADA HPI Medicaid adult dental benefit changes (curated below from
                              the Medicaid Adult Dental Coverage Checker reports and state
                              announcements; no machine-readable dated table exists)

National (US) milestones stay in src/config.py because they also mark the trend chart.

    python scripts/preprocess_policy.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_access import HEADERS, OUT_DIR, PLACES_PATH, ROOT, _kff_tables  # noqa: E402

KFF_EXPANSION_URL = ("https://www.kff.org/affordable-care-act/state-indicator/"
                     "state-activity-around-expanding-medicaid-under-the-affordable-care-act/")

CAREQUEST_2023 = "https://carequest.org/blog/five-key-takeaways-about-medicaid-adult-dental-benefits-in-2023/"
CAREQUEST_2023_CHANGES = "https://carequest.org/?p=2706"
CAREQUEST_2024_CHANGES = "https://carequest.org/?p=1898"

# (date, state, milestone, detail, source). Dates are ISO; "YYYY-MM" where the
# source gives only the month.
STATE_DENTAL_EVENTS = [
    ("2020-10-01", "DE", "Medicaid adult dental benefit created",
     "First adult dental benefit for full-benefit Medicaid adults 21+ (SPA 20-0001); "
     "delayed from April 2020 by the pandemic.",
     "https://news.delaware.gov/2020/09/30/dental-coverage-for-low-income-adults-in-delaware-begins-oct-1/"),
    ("2021-01-01", "WV", "Medicaid adult dental benefit created",
     "Limited adult benefit for diagnostic, preventive and restorative care, capped at $1,000 a year.",
     CAREQUEST_2023),
    ("2021-07-01", "OK", "Medicaid adult dental benefit reinstated",
     "Preventive, restorative and denture services restored for all adult beneficiaries.",
     CAREQUEST_2023),
    ("2021-07-01", "VA", "Comprehensive Medicaid adult dental benefit",
     "Exams, cleanings, fillings, root canals, gum treatment, dentures and oral surgery for "
     "full-benefit adults 21+ (more than 750,000 people).",
     "https://vamedicaid.dmas.virginia.gov/node/2304"),
    ("2022-07-01", "KS", "Medicaid adult dental benefit restored and expanded",
     "Fillings, crowns and periodontal treatment added; adult benefits had been eliminated in 2010.",
     CAREQUEST_2023),
    ("2022-07-01", "ME", "Extensive Medicaid adult dental benefit",
     "Most dental service categories plus oral surgery for all adults.",
     CAREQUEST_2023),
    ("2023-01-01", "HI", "Medicaid adult dental benefit expanded",
     "Diagnostic, preventive, restorative care and dentures; coverage had been limited to pain "
     "relief and extractions since 2008.",
     CAREQUEST_2023_CHANGES),
    ("2023-01-01", "KY", "Medicaid adult dental benefit expanded",
     "Cleanings, crowns, root canals and dentures added for all adults.",
     CAREQUEST_2023_CHANGES),
    ("2023-01-01", "MD", "Medicaid adult dental benefit for all adults",
     "Diagnostic, preventive, restorative, endodontic, periodontal and oral surgery services; "
     "previously limited to specific groups.",
     CAREQUEST_2023_CHANGES),
    ("2023-01-01", "TN", "Medicaid adult dental benefit for all adults",
     "Endodontic, periodontal and prosthodontic care extended to all adults (pregnant and "
     "postpartum adults covered from April 2022).",
     CAREQUEST_2023_CHANGES),
    ("2023-04-01", "MI", "Medicaid adult dental benefit redesigned",
     "X-rays, cleanings, fillings, crowns, root canals, periodontal care, dentures and extractions, "
     "with higher provider rates.",
     CAREQUEST_2023_CHANGES),
    ("2023-04-01", "NH", "Medicaid adult dental benefit created",
     "Diagnostic, preventive, restorative and oral surgery services ($1,500 annual maximum); "
     "previously emergency-only.",
     CAREQUEST_2023_CHANGES),
    ("2024-01", "NE", "Medicaid adult dental annual maximum lifted",
     "The $750 annual benefit maximum was removed.",
     CAREQUEST_2024_CHANGES),
    ("2024-01-01", "CT", "Medicaid periodontal coverage added",
     "Periodontal treatment for adults with treatable periodontal disease and specified chronic conditions.",
     CAREQUEST_2024_CHANGES),
    ("2024-01-01", "MN", "Medicaid dental for all adults",
     "Medically necessary dental coverage for all enrolled adults regardless of age or pregnancy.",
     CAREQUEST_2024_CHANGES),
    ("2024-01-01", "UT", "Postpartum Medicaid dental extended",
     "Postpartum dental coverage extended from 60 days to 12 months.",
     CAREQUEST_2024_CHANGES),
    ("2024-07-01", "GA", "Medicaid adult dental benefit expanded",
     "Exams, cleanings, fluoride, fillings, crowns, root canals and dentures for all adult enrollees.",
     CAREQUEST_2024_CHANGES),
    ("2024-07", "WV", "Medicaid adult dental annual maximum raised",
     "Annual benefit maximum increased to $2,000.",
     CAREQUEST_2024_CHANGES),
    ("2024-07-19", "NV", "Section 1115 waiver approved: dental for adults with diabetes",
     "Limited dental package for nonpregnant adults with diabetes through FQHCs and tribal "
     "health centers (approval date; services start later).",
     CAREQUEST_2024_CHANGES),
    ("2025-04-01", "UT", "Medicaid dental for all adults",
     "Dental benefits opened to all adults under a Section 1115 waiver approved January 8, 2025.",
     CAREQUEST_2024_CHANGES),
]


def build_dental_events() -> pd.DataFrame:
    df = pd.DataFrame(STATE_DENTAL_EVENTS, columns=["date", "state", "title", "detail", "source"])
    df["domain"] = "Mastication"
    return df


def build_expansion_events(name_to_abbr: dict[str, str]) -> pd.DataFrame:
    resp = requests.get(KFF_EXPANSION_URL, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    _, table = _kff_tables(resp.text)[0]  # current status tab
    header, rows = table[0], table[2:]
    df = pd.DataFrame([r + [None] * (len(header) - len(r)) for r in rows], columns=header)
    df = df.rename(columns={
        header[0]: "state_name",
        "Status of Medicaid Expansion Decision": "status",
        "Expansion Implementation Date": "impl_date",
        "Expansion Adopted Through Ballot Initiative": "ballot",
    })
    df = df[df["state_name"].isin(name_to_abbr) & (df["status"] == "Adopted")]
    df["date"] = [datetime.strptime(d, "%m/%d/%Y").date().isoformat() for d in df["impl_date"]]

    def detail(ballot: str) -> str:
        how = f"adopted by ballot initiative ({ballot[5:-1]})" if ballot.startswith("Yes") else "adopted by the state"
        return f"ACA Medicaid expansion to adults up to 138% FPL, {how}. Extends the state's Medicaid " \
               "benefits, including any adult dental, vision and hearing coverage, to newly eligible adults aged 19–64."

    print(f"  KFF Medicaid expansion: {len(df)} adopting states")
    return pd.DataFrame({
        "date": df["date"],
        "state": df["state_name"].map(name_to_abbr),
        "title": "ACA Medicaid expansion implemented",
        "detail": df["ballot"].map(detail),
        "source": KFF_EXPANSION_URL,
        "domain": "Access",
    })


def main() -> int:
    if not PLACES_PATH.exists():
        print("Run scripts/preprocess.py first (state names come from the PLACES snapshot).")
        return 1
    states = pd.read_parquet(PLACES_PATH, columns=["state", "state_name"]).drop_duplicates()
    name_to_abbr = dict(zip(states["state_name"], states["state"]))

    print("[1/2] KFF State Health Facts: Medicaid expansion")
    expansion = build_expansion_events(name_to_abbr)
    print("[2/2] CHCS / CareQuest / ADA HPI: Medicaid adult dental changes")
    dental = build_dental_events()
    print(f"  {len(dental)} curated events")

    out = (pd.concat([expansion, dental], ignore_index=True)
           .assign(scope="State")
           .sort_values(["date", "state"], ignore_index=True))
    out = out[["date", "scope", "state", "domain", "title", "detail", "source"]]
    path = OUT_DIR / "policy_events.parquet"
    out.to_parquet(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)} ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
