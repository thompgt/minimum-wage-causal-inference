"""Event-study specification: leads/lags of unemployment around adoption.

Identification
--------------
The regression needs a group that is *never* treated. Restricting the
sample to states with a finite `adoption_year` — as this module used to —
leaves only differential timing among treated states to identify the
coefficients, which means already-treated states serve as controls for
later adopters. That is precisely the comparison this repo's own
Callaway-Sant'Anna module exists to avoid, and under heterogeneous
effects it biases the leads and lags in a direction that is not signed.

So the sample here is:

* **never-treated states** — kept, with every lead/lag dummy at 0. They
  are the clean control group; the entity fixed effect absorbs their
  level and they help identify the common year effects.
* **staggered adopters** — kept, contributing the leads and lags.
* **always-treated states** — dropped. A state already above the federal
  floor in the panel's first year has no pre-period, so it can only ever
  land in the top-coded lag bin, where it contaminates that one
  coefficient with a group that has no identified counterfactual.

Standard errors are clustered by state throughout.
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def build_event_time(panel, min_lead=-5, max_lag=5,
                     include_never_treated=True, drop_always_treated=True):
    """Add `rel_time` (year - adoption_year), clipped to [min_lead, max_lag].

    Never-treated states keep `rel_time = <NA>`; they carry no lead/lag
    dummy and act as the untreated control group. Always-treated states
    (adoption in or before the panel's first year, hence no pre-period)
    are dropped by default.

    The returned frame records `n_never_treated`, `always_treated_states`
    and `n_adopters` in `.attrs` so callers can report what the sample
    actually is.
    """
    df = panel.copy()
    first_year = int(df["year"].min())

    adoption = pd.to_numeric(df["adoption_year"], errors="coerce")
    always_treated = sorted(df.loc[adoption <= first_year, "state"].unique())
    if drop_always_treated and always_treated:
        df = df[~df["state"].isin(always_treated)].copy()
        adoption = pd.to_numeric(df["adoption_year"], errors="coerce")

    never_treated = sorted(df.loc[adoption.isna(), "state"].unique())
    if not include_never_treated:
        df = df[adoption.notna()].copy()
        adoption = pd.to_numeric(df["adoption_year"], errors="coerce")
        never_treated = []

    rel = df["year"] - adoption
    df["rel_time"] = rel.clip(min_lead, max_lag).astype("Int64")
    df["never_treated"] = adoption.isna()

    df.attrs.update(
        n_never_treated=len(never_treated),
        never_treated_states=never_treated,
        always_treated_states=always_treated if drop_always_treated else [],
        n_adopters=int(df.loc[adoption.notna(), "state"].nunique()),
    )
    return df


def estimate_event_study(panel, outcome="unemployment_rate", min_lead=-5, max_lag=5,
                         omit=-1, include_never_treated=True,
                         drop_always_treated=True):
    """Fit an event-study regression with relative-time dummies (omitting `omit`).

    Returns `(result, summary)`. `summary` has one row per event time
    including the omitted reference at 0, and carries the dummy-column →
    event-time map plus the sample composition in `.attrs`, which
    `average_post_effect` and the pre-trend Wald test both need.
    """
    df = build_event_time(
        panel, min_lead, max_lag,
        include_never_treated=include_never_treated,
        drop_always_treated=drop_always_treated,
    )
    sample = dict(df.attrs)

    dummy_cols = []
    col_to_reltime = {}
    for t in range(min_lead, max_lag + 1):
        if t == omit:
            continue
        col = f"rel_m{abs(t)}" if t < 0 else f"rel_p{t}"
        # Never-treated rows compare <NA> == t, which is pd.NA; they must
        # read as 0 on every dummy, not drop out of the regression.
        df[col] = df["rel_time"].eq(t).fillna(False).astype(int)
        dummy_cols.append(col)
        col_to_reltime[col] = t

    df = df.set_index(["state", "year"])
    formula = f"{outcome} ~ " + " + ".join(dummy_cols) + " + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=df)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    summary = pd.DataFrame({
        "rel_time": [col_to_reltime[c] for c in dummy_cols],
        "coef": result.params.values,
        "ci_lower": result.conf_int()["lower"].values,
        "ci_upper": result.conf_int()["upper"].values,
    }).sort_values("rel_time").reset_index(drop=True)
    # Add the omitted reference period back in at coef=0 for plotting.
    summary = pd.concat([
        summary,
        pd.DataFrame([{"rel_time": omit, "coef": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}]),
    ]).sort_values("rel_time").reset_index(drop=True)

    summary.attrs.update(col_to_reltime=col_to_reltime, omit=omit, **sample)
    return result, summary


def contrast_columns(summary, min_rel_time=None, max_rel_time=None):
    """Dummy column names whose event time falls in [min_rel_time, max_rel_time]."""
    mapping = summary.attrs.get("col_to_reltime")
    if not mapping:
        raise ValueError(
            "summary has no col_to_reltime in .attrs; it did not come from "
            "estimate_event_study"
        )
    cols = [
        col for col, t in mapping.items()
        if (min_rel_time is None or t >= min_rel_time)
        and (max_rel_time is None or t <= max_rel_time)
    ]
    if not cols:
        raise ValueError("no event-time coefficients in the requested window")
    return sorted(cols, key=lambda c: mapping[c])


def linear_contrast(result, columns, weights=None, alpha=0.05):
    """Point estimate and CI for a weighted sum of fitted coefficients.

    Averaging the *endpoints* of several coefficients' confidence
    intervals — which is what this repo's comparison table used to do —
    ignores their covariance and produces an interval with no coverage
    guarantee. A linear combination has its own standard error,
    sqrt(w'Vw), and that is what is computed here.
    """
    from scipy.stats import norm

    params = result.params
    if weights is None:
        weights = np.full(len(columns), 1.0 / len(columns))
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(columns):
        raise ValueError("weights and columns must be the same length")

    w = np.zeros(len(params))
    for col, weight in zip(columns, weights, strict=True):
        w[params.index.get_loc(col)] = weight

    cov = np.asarray(result.cov)
    estimate = float(w @ params.to_numpy())
    se = float(np.sqrt(max(w @ cov @ w, 0.0)))
    z = float(norm.ppf(1 - alpha / 2))
    return {
        "estimate": estimate,
        "std_error": se,
        "ci_lower": estimate - z * se,
        "ci_upper": estimate + z * se,
        "n_coefficients": len(columns),
        "columns": list(columns),
    }


def average_post_effect(result, summary, min_rel_time=0, alpha=0.05):
    """The average post-adoption event-study coefficient, as a real contrast.

    This is the number the method-comparison table reports for the event
    study: an equally weighted average of the post-adoption coefficients
    with a standard error that accounts for the covariance between them.
    """
    cols = contrast_columns(summary, min_rel_time=min_rel_time)
    return linear_contrast(result, cols, alpha=alpha)


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")
    result, summary = estimate_event_study(panel)
    print(
        f"\nsample: {summary.attrs['n_adopters']} adopters, "
        f"{summary.attrs['n_never_treated']} never-treated controls, "
        f"{len(summary.attrs['always_treated_states'])} always-treated dropped"
    )
    print(summary.to_string(index=False))
    post = average_post_effect(result, summary)
    print(
        f"\npost-period average: {post['estimate']:+.3f} "
        f"[{post['ci_lower']:+.3f}, {post['ci_upper']:+.3f}] "
        f"(se {post['std_error']:.3f}, {post['n_coefficients']} coefficients)"
    )
