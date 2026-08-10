"""Two-way fixed effects DiD: unemployment ~ log(minimum_wage) + state FE + year FE.

Standard errors clustered by state, the conventional choice for
state-by-year panels with serially correlated treatment assignment
(Bertrand, Duflo & Mullainathan 2004).
"""
import pandas as pd
from linearmodels.panel import PanelOLS


def estimate_twfe(panel, outcome="unemployment_rate", treatment="log_minimum_wage"):
    """Fit unemployment ~ treatment with state + year fixed effects.

    `panel` must be a state-year (or state-month) DataFrame with columns
    `state`, `year`, outcome, and treatment. Returns the fitted
    PanelOLSResults object.
    """
    df = panel.set_index(["state", "year"])
    model = PanelOLS.from_formula(
        f"{outcome} ~ {treatment} + EntityEffects + TimeEffects",
        data=df,
    )
    return model.fit(cov_type="clustered", cluster_entity=True)


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
    from src.data.synthetic import generate_state_year_panel

    panel = generate_state_year_panel()
    result = estimate_twfe(panel)
    print(result.summary)
    print("\n", summarize_twfe(result))
