"""Two-way fixed effects DiD, in both a continuous and a binary treatment.

Standard errors clustered by state throughout, the conventional choice
for state-by-year panels with serially correlated treatment assignment
(Bertrand, Duflo & Mullainathan 2004).

Two specifications, because they have different problems and the
comparison against Callaway-Sant'Anna only means something against the
second.

`estimate_twfe` regresses unemployment on **log minimum wage**, a
continuous dose. Its coefficient is a semi-elasticity, and it is a
weighted average of dose-response slopes whose weights can be negative
(de Chaisemartin & D'Haultfœuille 2020; Callaway, Goodman-Bacon &
Sant'Anna 2021 on continuous treatment). That is a distinct problem from
the staggered-timing bias in the binary case — it does not go away with
a never-treated group and it is not what the Goodman-Bacon decomposition
describes.

`estimate_twfe_binary` regresses unemployment on the **absorbing binary
indicator** of being above the federal floor — the same treatment
Callaway-Sant'Anna, the event study and synthetic control estimate. Its
gap from the Callaway-Sant'Anna estimate is the staggered-timing bias,
like for like, in the same units. Comparing the continuous specification
against CS confounds that bias with the dose-weighting problem and with
a units conversion.
"""
import pandas as pd
from linearmodels.panel import PanelOLS

#: The absorbing binary treatment `build_panel.add_adoption_year` derives.
BINARY_TREATMENT = "treated"


def estimate_twfe(panel, outcome="unemployment_rate", treatment="log_minimum_wage"):
    """Fit unemployment ~ treatment with state + year fixed effects.

    `panel` must be a state-year (or state-month) DataFrame with columns
    `state`, `year`, outcome, and treatment. With the default
    `log_minimum_wage` the coefficient is a semi-elasticity — see the
    module docstring on why that is not directly comparable to the
    binary-treatment estimators. Returns the fitted PanelOLSResults.
    """
    df = panel.set_index(["state", "year"])
    model = PanelOLS.from_formula(
        f"{outcome} ~ {treatment} + EntityEffects + TimeEffects",
        data=df,
    )
    return model.fit(cov_type="clustered", cluster_entity=True)


def estimate_twfe_binary(panel, outcome="unemployment_rate",
                         treatment=BINARY_TREATMENT):
    """TWFE on the binary absorbing treatment, in percentage points.

    This is the specification the Goodman-Bacon decomposition is about,
    and the one whose difference from Callaway-Sant'Anna isolates
    staggered-timing bias rather than mixing it with dose weighting.
    """
    if treatment not in panel.columns:
        raise ValueError(
            f"panel has no {treatment!r} column; run "
            "src.data.build_panel.add_adoption_year first"
        )
    df = panel.copy()
    df[treatment] = df[treatment].fillna(False).astype(float)
    if df[treatment].nunique() < 2:
        raise ValueError(
            f"{treatment!r} does not vary; a binary TWFE needs both treated "
            "and untreated state-years"
        )
    return estimate_twfe(df, outcome=outcome, treatment=treatment)


def summarize_twfe(result):
    ci = result.conf_int()
    return pd.DataFrame({
        "coef": result.params,
        "std_error": result.std_errors,
        "pvalue": result.pvalues,
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
    })


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")

    continuous = estimate_twfe(panel)
    binary = estimate_twfe_binary(panel)
    print("\ncontinuous (log minimum wage, semi-elasticity):")
    print(summarize_twfe(continuous).loc[["log_minimum_wage"]])
    print("\nbinary (above federal floor, percentage points):")
    print(summarize_twfe(binary).loc[[BINARY_TREATMENT]])
