"""Generate a synthetic state-month panel with a known ground-truth effect.

Lets every downstream method/notebook/app run and be sanity-checked before
real BLS/minimum-wage data is wired in. The panel mimics the real one's
shape (state, year, month, minimum_wage, unemployment_rate) plus a
`true_effect` column recording the ground truth so estimators can be graded
against it.

Design: staggered minimum wage adoption across states/years (some states
never raise above the federal floor = never-treated controls, for
Callaway-Sant'Anna). Unemployment is generated as state FE + year FE +
TRUE_EFFECT * log(minimum_wage) + noise, so a correctly specified DiD
should recover TRUE_EFFECT.
"""
import numpy as np
import pandas as pd

STATES = [f"S{i:02d}" for i in range(1, 21)]  # 20 synthetic states
YEARS = list(range(2005, 2024))
MONTHS = list(range(1, 13))
FEDERAL_MIN_WAGE = 7.25
TRUE_EFFECT = -0.15  # true elasticity: d(unemployment_rate) / d(log minimum wage)


def generate_panel(seed=42, true_effect=TRUE_EFFECT):
    rng = np.random.default_rng(seed)

    state_fe = {s: rng.normal(6.0, 1.2) for s in STATES}  # baseline unemployment level
    year_fe = {y: rng.normal(0, 0.5) for y in YEARS}  # business-cycle shocks common to all states

    # Two-thirds of states raise their minimum wage at a random point after 2010;
    # the rest stay at the federal floor (never-treated controls).
    treated_states = rng.choice(STATES, size=int(len(STATES) * 0.65), replace=False)
    adoption_year = {
        s: int(rng.integers(2011, 2022)) for s in treated_states
    }

    rows = []
    for state in STATES:
        wage = FEDERAL_MIN_WAGE
        for year in YEARS:
            if state in adoption_year and year >= adoption_year[state]:
                # ratchet up ~$0.75/year once adopted, small idiosyncratic increments after
                steps_since = year - adoption_year[state]
                wage = FEDERAL_MIN_WAGE + 0.75 * (steps_since + 1)
            for month in MONTHS:
                log_wage = np.log(wage)
                noise = rng.normal(0, 0.4)
                unemployment = (
                    state_fe[state]
                    + year_fe[year]
                    + true_effect * (log_wage - np.log(FEDERAL_MIN_WAGE))
                    + noise
                )
                rows.append({
                    "state": state,
                    "year": year,
                    "month": month,
                    "minimum_wage": round(wage, 2),
                    "unemployment_rate": round(max(unemployment, 1.0), 2),
                    "treated": state in adoption_year and year >= adoption_year.get(state, 10**9),
                    "adoption_year": adoption_year.get(state, np.nan),
                })

    df = pd.DataFrame(rows)
    df["log_minimum_wage"] = np.log(df["minimum_wage"])
    return df.sort_values(["state", "year", "month"]).reset_index(drop=True)


def generate_state_year_panel(seed=42, true_effect=TRUE_EFFECT):
    month_panel = generate_panel(seed=seed, true_effect=true_effect)
    yearly = (
        month_panel.groupby(["state", "year"])
        .agg(
            unemployment_rate=("unemployment_rate", "mean"),
            minimum_wage=("minimum_wage", "last"),
            treated=("treated", "max"),
            adoption_year=("adoption_year", "first"),
        )
        .reset_index()
    )
    yearly["log_minimum_wage"] = np.log(yearly["minimum_wage"])
    return yearly.sort_values(["state", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    panel = generate_state_year_panel()
    print(panel.head(10))
    print(f"\n{len(panel)} state-year rows, {panel['state'].nunique()} states, "
          f"true effect = {TRUE_EFFECT}")
