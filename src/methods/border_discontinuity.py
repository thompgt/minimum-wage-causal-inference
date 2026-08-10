"""Border-discontinuity design (Dube, Lester & Reich 2010 style).

Compares unemployment changes between contiguous state pairs with
different minimum wage policies, using each pair as its own local control
group. This is a robustness check on TWFE: it holds local economic
conditions closer to fixed than a nationwide state panel does, at the
cost of only using states that border a state with a different policy.

Two things the naive version of this design gets wrong.

**Pair selection.** A hand-picked subset of borders is selected on the
same thing the design is trying to measure. The 28 pairs this module used
to ship were heavily CA/OR/NV and NY/NJ/PA — high-minimum-wage states —
so the sample was chosen on treatment. `US_STATE_BORDER_PAIRS` is now the
complete enumeration of contiguous jurisdiction pairs, derived from an
adjacency map rather than picked, so there is no selection rule left to
be sensitive to.

**Specification.** Regressing the outcome difference on the treatment
difference with only a constant lets the slope absorb permanent level
differences between pairs and nationwide shocks common to all of them.
Dube-Lester-Reich identify off *within-pair variation over time*, which
requires pair and period fixed effects. Standard errors are clustered
two-way on both states of the pair, because a state appears in as many
pairs as it has neighbours and those observations are not independent.
"""
import numpy as np
import pandas as pd

# Land-border adjacency for the 48 contiguous states plus DC. Written one
# way round and symmetrised below, so the pair list cannot drift out of
# sync with the adjacency it claims to encode. Four Corners contacts
# (AZ-CO, NM-UT) are included, as is standard in this literature.
_STATE_ADJACENCY = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AR": ["LA", "MO", "MS", "OK", "TN", "TX"],
    "AZ": ["CA", "CO", "NM", "NV", "UT"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DC": ["MD", "VA"],
    "DE": ["MD", "NJ", "PA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IA", "IN", "KY", "MO", "WI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MD": ["DC", "DE", "PA", "VA", "WV"],
    "ME": ["NH"],
    "MI": ["IN", "OH", "WI"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MO": ["AR", "IA", "IL", "KS", "KY", "NE", "OK", "TN"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NH": ["MA", "ME", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "ND", "NE", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MO", "MS", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NM", "NV", "WY"],
    "VA": ["DC", "KY", "MD", "NC", "TN", "WV"],
    "VT": ["MA", "NH", "NY"],
    "WA": ["ID", "OR"],
    "WI": ["IA", "IL", "MI", "MN"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}


def _enumerate_border_pairs(adjacency):
    """Every unordered adjacent pair, canonically ordered, sorted, deduplicated.

    Also asserts the adjacency map is symmetric — an asymmetric entry is a
    typo that would silently drop a border.
    """
    asymmetric = [
        (a, b)
        for a, neighbours in adjacency.items()
        for b in neighbours
        if a not in adjacency.get(b, [])
    ]
    if asymmetric:
        raise ValueError(f"adjacency map is not symmetric: {asymmetric}")
    pairs = {
        tuple(sorted((a, b)))
        for a, neighbours in adjacency.items()
        for b in neighbours
    }
    return sorted(pairs)


#: Every contiguous jurisdiction pair (48 states + DC), enumerated rather
#: than selected.
US_STATE_BORDER_PAIRS = _enumerate_border_pairs(_STATE_ADJACENCY)


def border_pair_diffs(panel, border_pairs, outcome="unemployment_rate",
                      treatment="minimum_wage", period_col="year"):
    """For each border pair and period, compute (state_a - state_b) for
    outcome and treatment. Returns a long DataFrame keyed by pair/period.

    Pairs are canonically ordered (state_a < state_b) so the sign of every
    difference is well defined and a state always sits on the same side of
    every pair it belongs to.

    A large treatment gap with little corresponding outcome gap is
    evidence against a large disemployment effect; the sign/magnitude
    relationship across many pairs is the actual estimand of interest
    (regress outcome_diff on treatment_diff, see `estimate_border_effect`).
    """
    indexed = panel.set_index(["state", period_col])
    available = set(panel["state"].unique())
    rows = []
    used_pairs = 0
    for pair in border_pairs:
        state_a, state_b = sorted(pair)
        if state_a not in available or state_b not in available:
            continue
        periods = sorted(
            set(panel.loc[panel["state"] == state_a, period_col])
            & set(panel.loc[panel["state"] == state_b, period_col])
        )
        if not periods:
            continue
        used_pairs += 1
        for period in periods:
            row_a = indexed.loc[(state_a, period)]
            row_b = indexed.loc[(state_b, period)]
            rows.append({
                "state_a": state_a,
                "state_b": state_b,
                "pair": f"{state_a}-{state_b}",
                period_col: period,
                "outcome_diff": row_a[outcome] - row_b[outcome],
                "treatment_diff": row_a[treatment] - row_b[treatment],
            })
    out = pd.DataFrame(rows)
    out.attrs.update(
        n_pairs_requested=len(border_pairs),
        n_pairs_used=used_pairs,
        period_col=period_col,
    )
    return out


def estimate_border_effect(pair_diffs, pair_fe=True, period_fe=True,
                           period_col=None, cluster="two-way"):
    """Regress the outcome difference on the treatment difference.

    With `pair_fe` and `period_fe` (the defaults) the slope is identified
    off variation *within* a border pair *over time*, net of shocks common
    to all pairs in a period — which is what the Dube-Lester-Reich
    argument actually rests on. A pooled OLS with only a constant, as this
    function used to run, lets the slope absorb permanent cross-pair level
    differences and national business-cycle swings alike.

    `cluster="two-way"` clusters on both states of the pair. A state
    borders several others, so its observations recur across pairs;
    clustering on the pair alone treats those as independent.
    """
    import statsmodels.api as sm

    if pair_diffs.empty:
        raise ValueError("no border-pair observations to estimate from")

    period_col = period_col or pair_diffs.attrs.get("period_col", "year")
    design = [pair_diffs[["treatment_diff"]].astype(float).reset_index(drop=True)]

    if pair_fe:
        design.append(
            pd.get_dummies(pair_diffs["pair"].reset_index(drop=True),
                           prefix="pair", drop_first=True, dtype=float)
        )
    if period_fe:
        design.append(
            pd.get_dummies(pair_diffs[period_col].reset_index(drop=True),
                           prefix="period", drop_first=True, dtype=float)
        )

    X = sm.add_constant(pd.concat(design, axis=1), has_constant="add")
    y = pair_diffs["outcome_diff"].astype(float).reset_index(drop=True)

    # Pair and period dummies eat degrees of freedom fast. Saying so beats a
    # ZeroDivisionError from deep inside the covariance estimator.
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"border design is saturated: {X.shape[0]} pair-periods for "
            f"{X.shape[1]} parameters. Supply more pairs or periods, or turn "
            "off pair_fe/period_fe."
        )

    if cluster == "two-way":
        groups = np.column_stack([
            pd.factorize(pair_diffs["state_a"])[0],
            pd.factorize(pair_diffs["state_b"])[0],
        ])
    elif cluster == "pair":
        groups = pd.factorize(pair_diffs["pair"])[0]
    else:
        raise ValueError(f"unknown cluster option {cluster!r}")

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    model.border_spec = {
        "pair_fe": pair_fe,
        "period_fe": period_fe,
        "cluster": cluster,
        "n_pairs": int(pair_diffs["pair"].nunique()),
        "n_obs": int(len(pair_diffs)),
    }
    return model


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")
    print(f"{len(US_STATE_BORDER_PAIRS)} contiguous pairs enumerated")

    diffs = border_pair_diffs(panel, US_STATE_BORDER_PAIRS,
                              treatment="log_minimum_wage")
    print(f"{diffs.attrs['n_pairs_used']} pairs present in the panel, "
          f"{len(diffs)} pair-years")

    for label, kwargs in [
        ("pooled OLS, pair-clustered", dict(pair_fe=False, period_fe=False,
                                            cluster="pair")),
        ("pair + period FE, two-way clustered", {}),
    ]:
        model = estimate_border_effect(diffs, **kwargs)
        ci = model.conf_int().loc["treatment_diff"]
        print(f"\n{label}: {model.params['treatment_diff']:+.4f} "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
