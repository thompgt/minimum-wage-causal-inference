"""Border-discontinuity design (Dube, Lester & Reich 2010 style).

Compares unemployment changes between contiguous state pairs with
different minimum wage policies, using each pair as its own local control
group. This is a robustness check on TWFE: it holds local economic
conditions closer to fixed than a nationwide state panel does, at the
cost of only using states that border a state with a different policy.
"""
import pandas as pd

# Real contiguous U.S. state-pair borders (a subset commonly used in the
# literature; extend as needed). Each pair is unordered.
US_STATE_BORDER_PAIRS = [
    ("CA", "OR"), ("CA", "NV"), ("CA", "AZ"),
    ("OR", "WA"), ("OR", "ID"), ("OR", "NV"),
    ("NV", "AZ"), ("NV", "UT"), ("NV", "ID"),
    ("NJ", "PA"), ("NJ", "NY"), ("NJ", "DE"),
    ("PA", "NY"), ("PA", "OH"), ("PA", "WV"), ("PA", "MD"),
    ("NY", "CT"), ("NY", "MA"), ("NY", "VT"),
    ("IL", "IN"), ("IL", "WI"), ("IL", "MO"), ("IL", "KY"), ("IL", "IA"),
    ("IN", "OH"), ("IN", "KY"), ("IN", "MI"),
    ("WA", "ID"),
]


def border_pair_diffs(panel, border_pairs, outcome="unemployment_rate",
                       treatment="minimum_wage", period_col="year"):
    """For each border pair and period, compute (state_a - state_b) for
    outcome and treatment. Returns a long DataFrame keyed by pair/period.

    A large treatment gap with little corresponding outcome gap is
    evidence against a large disemployment effect; the sign/magnitude
    relationship across many pairs is the actual estimand of interest
    (regress outcome_diff on treatment_diff, see `estimate_border_effect`).
    """
    indexed = panel.set_index(["state", period_col])
    rows = []
    for state_a, state_b in border_pairs:
        if state_a not in panel["state"].values or state_b not in panel["state"].values:
            continue
        periods = sorted(
            set(panel.loc[panel["state"] == state_a, period_col])
            & set(panel.loc[panel["state"] == state_b, period_col])
        )
        for period in periods:
            row_a = indexed.loc[(state_a, period)]
            row_b = indexed.loc[(state_b, period)]
            rows.append({
                "state_a": state_a,
                "state_b": state_b,
                period_col: period,
                "outcome_diff": row_a[outcome] - row_b[outcome],
                "treatment_diff": row_a[treatment] - row_b[treatment],
            })
    return pd.DataFrame(rows)


def estimate_border_effect(pair_diffs):
    """OLS of outcome_diff on treatment_diff across all border-pair-periods.

    The slope is the border-discontinuity estimate of the minimum wage's
    effect on unemployment, analogous to the TWFE coefficient but
    identified off within-border-pair variation only.
    """
    import statsmodels.api as sm

    X = sm.add_constant(pair_diffs["treatment_diff"])
    y = pair_diffs["outcome_diff"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": pair_diffs["state_a"] + "_" + pair_diffs["state_b"]},
    )
    return model


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel

    panel = generate_state_year_panel()
    # Synthetic states (S01..S20) have no real geography, so build
    # arbitrary "border" pairs for demonstration purposes only.
    states = sorted(panel["state"].unique())
    demo_pairs = list(zip(states[::2], states[1::2], strict=False))

    diffs = border_pair_diffs(panel, demo_pairs)
    print(diffs.head())
    model = estimate_border_effect(diffs)
    print(model.summary())
