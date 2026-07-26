"""Merge unemployment and minimum wage sources into a clean analysis panel.

Produces both a state-month panel and a state-year panel (annual average
unemployment rate, year-end minimum wage) in data/processed/.

Treatment definition
--------------------
A state is *treated* from the first year its own minimum wage binds above
the federal floor, and treatment is modelled as **absorbing**: once a state
has legislated above the federal minimum we treat it as a minimum-wage
state from then on, even in years when a federal increase temporarily
catches up to it (as happened in 2007-2009). Staggered-adoption estimators
such as Callaway-Sant'Anna require an absorbing treatment, and the
alternative — letting states switch back when Congress moves — would
attribute federal policy changes to state cohorts.

`adoption_year` is that first year, or <NA> for never-treated states.
States already above the federal floor in the panel's first year are
always-treated: they have no clean pre-period, so `adoption_year` is set to
the first panel year and estimators that need a pre-period drop them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetch_minwage import load_minimum_wage

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Wages are meaningful to the cent; anything below this is float noise.
CENT_TOLERANCE = 0.005


def _log_wage(series):
    wage = pd.to_numeric(series, errors="coerce")
    return np.log(wage.where(wage > 0))


def build_state_month_panel(unemployment_path=None, minwage_path=None):
    unemployment_path = unemployment_path or RAW_DIR / "bls_unemployment.parquet"
    unemployment_path = Path(unemployment_path)
    if not unemployment_path.exists():
        raise FileNotFoundError(
            f"{unemployment_path} not found. Run `python -m src.data.fetch_bls` first."
        )
    unemployment = pd.read_parquet(unemployment_path)
    minwage = load_minimum_wage(minwage_path)

    panel = unemployment.merge(minwage, on=["state", "year", "month"], how="inner")
    panel["log_minimum_wage"] = _log_wage(panel["minimum_wage"])
    # Compare with a half-cent tolerance: source wages are only meaningful to
    # the cent, and a bare `>` turns float noise into spurious treatment.
    panel["above_federal"] = (
        panel["minimum_wage"] > panel["federal_minimum_wage"] + CENT_TOLERANCE
    )
    panel = panel.sort_values(["state", "year", "month"]).reset_index(drop=True)

    if panel.empty:
        raise ValueError("build_state_month_panel produced an empty panel")
    if panel.duplicated(subset=["state", "year", "month"]).any():
        raise ValueError("build_state_month_panel produced duplicate state-year-month rows")
    if panel["minimum_wage"].isna().any():
        raise ValueError("build_state_month_panel has missing minimum_wage values")
    if panel["unemployment_rate"].isna().any():
        raise ValueError("build_state_month_panel has missing unemployment_rate values")

    return panel


def build_state_year_panel(state_month_panel):
    yearly = (
        state_month_panel.groupby(["state", "year"])
        .agg(
            unemployment_rate=("unemployment_rate", "mean"),
            minimum_wage=("minimum_wage", "last"),  # year-end minimum wage
            federal_minimum_wage=("federal_minimum_wage", "last"),
            above_federal=("above_federal", "any"),
        )
        .reset_index()
    )
    yearly["log_minimum_wage"] = _log_wage(yearly["minimum_wage"])
    yearly = yearly.sort_values(["state", "year"]).reset_index(drop=True)
    return add_adoption_year(yearly)


def add_adoption_year(panel):
    """Add `adoption_year`, `treated`, and `event_time` to a state-year panel.

    See the module docstring for the treatment definition. Requires
    `state`, `year`, and `above_federal` columns.
    """
    required = {"state", "year", "above_federal"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)
    first_above = (
        panel[panel["above_federal"]]
        .groupby("state")["year"]
        .min()
    )
    panel["adoption_year"] = panel["state"].map(first_above).astype("Int64")

    # Absorbing treatment: on and after the state's own adoption year.
    panel["treated"] = (
        panel["adoption_year"].notna()
        & (panel["year"] >= panel["adoption_year"])
    )
    panel["event_time"] = (
        panel["year"] - panel["adoption_year"]
    ).astype("Int64")
    return panel


def summarize_cohorts(year_panel):
    """One row per adoption cohort: how many states, and their pre-period length."""
    first_year = int(year_panel["year"].min())
    treated = year_panel.dropna(subset=["adoption_year"])
    rows = (
        treated.groupby("adoption_year")["state"]
        .nunique()
        .reset_index(name="n_states")
    )
    rows["pre_periods"] = rows["adoption_year"].astype(int) - first_year
    rows["usable_for_event_study"] = rows["pre_periods"] > 0
    never = year_panel[year_panel["adoption_year"].isna()]["state"].nunique()
    return rows, never


def main():
    month_panel = build_state_month_panel()
    year_panel = build_state_year_panel(month_panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    month_panel.to_parquet(PROCESSED_DIR / "panel_state_month.parquet", index=False)
    year_panel.to_parquet(PROCESSED_DIR / "panel_state_year.parquet", index=False)
    print(
        f"Wrote {len(month_panel)} state-month rows and {len(year_panel)} "
        f"state-year rows ({year_panel['year'].min()}-{year_panel['year'].max()})"
    )

    cohorts, never = summarize_cohorts(year_panel)
    always = int((~cohorts["usable_for_event_study"]).mul(cohorts["n_states"]).sum())
    print(f"\n{never} never-treated states, {always} always-treated (no pre-period)")
    print("\nAdoption cohorts:")
    print(cohorts.to_string(index=False))


if __name__ == "__main__":
    main()
