"""Abadie synthetic control, implemented directly on scipy.

Builds a counterfactual for one treated unit as a weighted average of
untreated donor units, with weights constrained to the simplex (each
non-negative, summing to one). That constraint is what separates synthetic
control from ordinary regression: it forbids extrapolation, so the
counterfactual stays inside the convex hull of observed donor paths, and it
usually yields a sparse, interpretable set of donors.

    min_w  (X1 - X0 w)' V (X1 - X0 w)    s.t.  w >= 0,  sum(w) = 1

`X1` and `X0` are predictor vectors/matrices for the treated unit and the
donors. By default the predictors are **every pre-treatment outcome
value**. That choice makes the V-weighting step unnecessary: with lagged
outcomes as the only predictors, minimising the objective above under
V = I is exactly minimising pre-treatment mean squared prediction error,
which is what the nested V search is trying to achieve anyway. Pass extra
`covariates` and the nested search turns on, optimising a diagonal V to
minimise pre-treatment MSPE.

Inference follows Abadie, Diamond & Hainmueller (2010): there is only one
treated unit, so classical standard errors do not apply. Instead
`placebo_test` re-runs the whole procedure pretending each donor in turn
was treated, and compares the treated unit's post/pre RMSPE ratio against
that permutation distribution. The resulting p-value is the treated unit's
rank in that distribution.

Reference: Abadie, A., Diamond, A. and Hainmueller, J. (2010), "Synthetic
Control Methods for Comparative Case Studies", JASA 105(490), 493-505.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _pivot(panel, treated_unit, unit, time, outcome):
    wide = panel.pivot_table(index=unit, columns=time, values=outcome, aggfunc="mean")
    wide = wide[wide.notna().all(axis=1)]
    if treated_unit not in wide.index:
        raise ValueError(
            f"treated unit {treated_unit!r} is not in the panel with a complete "
            f"{outcome} series"
        )
    return wide.sort_index(axis=1)


def _solve_weights(X1, X0, V=None):
    """Simplex-constrained least squares: min (X1 - X0 w)' V (X1 - X0 w)."""
    n_donors = X0.shape[1]
    if V is None:
        V = np.ones(X1.shape[0])

    def objective(w):
        resid = X1 - X0 @ w
        return float(resid @ (V * resid))

    def gradient(w):
        resid = X1 - X0 @ w
        return -2.0 * X0.T @ (V * resid)

    result = minimize(
        objective,
        x0=np.full(n_donors, 1.0 / n_donors),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(result.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(n_donors, 1.0 / n_donors)


def _solve_v(X1, X0, Y1_pre, Y0_pre):
    """Diagonal V minimising pre-treatment MSPE of the outcome path."""
    k = X1.shape[0]

    def mspe(v_raw):
        v = np.abs(v_raw)
        if v.sum() <= 0:
            return np.inf
        v = v / v.sum()
        w = _solve_weights(X1, X0, V=v)
        resid = Y1_pre - Y0_pre @ w
        return float(np.mean(resid ** 2))

    result = minimize(
        mspe,
        x0=np.full(k, 1.0 / k),
        method="Nelder-Mead",
        options={"maxiter": 200 * k, "fatol": 1e-10, "xatol": 1e-8},
    )
    v = np.abs(result.x)
    return v / v.sum() if v.sum() > 0 else np.full(k, 1.0 / k)


def estimate_synthetic_control(
    panel,
    treated_unit,
    treatment_time,
    outcome="unemployment_rate",
    unit="state",
    time="year",
    covariates=None,
    donors=None,
):
    """Fit a synthetic control for `treated_unit` from an untreated donor pool.

    `covariates` is an optional list of column names averaged over the
    pre-treatment window and added to the predictors alongside the lagged
    outcomes. `donors` restricts the donor pool; by default every other
    unit in the panel is eligible, so filter out units also treated near
    `treatment_time` before calling if that matters for your design.

    Returns a dict with `paths`, `weights`, `gaps`, `pre_rmspe`,
    `post_rmspe`, `rmspe_ratio`, and `balance`.
    """
    wide = _pivot(panel, treated_unit, unit, time, outcome)
    times = wide.columns.to_numpy(dtype=float)
    pre_mask = times < treatment_time
    if pre_mask.sum() < 2:
        raise ValueError(
            f"need at least 2 pre-treatment periods before {treatment_time}, "
            f"got {int(pre_mask.sum())}"
        )
    if pre_mask.all():
        raise ValueError(f"no post-treatment periods at or after {treatment_time}")

    pool = [u for u in wide.index if u != treated_unit]
    if donors is not None:
        allowed = set(donors)
        pool = [u for u in pool if u in allowed]
    if not pool:
        raise ValueError("donor pool is empty")

    Y1 = wide.loc[treated_unit].to_numpy(dtype=float)
    Y0 = wide.loc[pool].to_numpy(dtype=float).T  # periods x donors

    # Predictors: pre-treatment outcomes, plus pre-treatment covariate means.
    X1 = Y1[pre_mask].copy()
    X0 = Y0[pre_mask].copy()
    predictor_names = [f"{outcome}_{int(t)}" for t in times[pre_mask]]
    if covariates:
        pre_panel = panel[panel[time] < treatment_time]
        means = pre_panel.groupby(unit)[list(covariates)].mean()
        X1 = np.concatenate([X1, means.loc[treated_unit].to_numpy(dtype=float)])
        X0 = np.vstack([X0, means.loc[pool].to_numpy(dtype=float).T])
        predictor_names += list(covariates)

    # Standardise so predictors on different scales weigh comparably.
    scale = np.concatenate([X1[:, None], X0], axis=1).std(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    X1s, X0s = X1 / scale, X0 / scale[:, None]

    if covariates:
        V = _solve_v(X1s, X0s, Y1[pre_mask], Y0[pre_mask])
    else:
        # Lagged outcomes only: V = I already minimises pre-treatment MSPE.
        V = np.ones(X1s.shape[0])

    w = _solve_weights(X1s, X0s, V=V)
    synthetic = Y0 @ w
    gaps = Y1 - synthetic

    pre_rmspe = float(np.sqrt(np.mean(gaps[pre_mask] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps[~pre_mask] ** 2)))

    paths = pd.DataFrame({
        time: np.concatenate([times, times]),
        unit: [treated_unit] * len(times) + [f"synthetic_{treated_unit}"] * len(times),
        outcome: np.concatenate([Y1, synthetic]),
        "type": ["actual"] * len(times) + ["synthetic"] * len(times),
    })

    weights = pd.Series(w, index=pool, name="weight").sort_values(ascending=False)
    balance = pd.DataFrame(
        {"treated": X1, "synthetic": X0 @ w, "donor_mean": X0.mean(axis=1)},
        index=predictor_names,
    )

    return {
        "paths": paths,
        "weights": weights,
        "gaps": pd.Series(gaps, index=times.astype(int), name="gap"),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "rmspe_ratio": post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf,
        "balance": balance,
        "treatment_time": treatment_time,
        "n_donors": len(pool),
    }


def placebo_test(
    panel,
    treated_unit,
    treatment_time,
    outcome="unemployment_rate",
    unit="state",
    time="year",
    covariates=None,
    donors=None,
    pre_rmspe_cutoff=None,
):
    """Abadie permutation inference: refit pretending each donor was treated.

    Donors that the method fits poorly before treatment produce huge
    post/pre ratios for reasons unrelated to any effect, so
    `pre_rmspe_cutoff` (a multiple of the treated unit's pre-RMSPE) drops
    them, as in Abadie et al. Returns (DataFrame, p_value).
    """
    actual = estimate_synthetic_control(
        panel, treated_unit, treatment_time, outcome, unit, time, covariates, donors
    )

    wide = _pivot(panel, treated_unit, unit, time, outcome)
    pool = [u for u in wide.index if u != treated_unit]
    if donors is not None:
        allowed = set(donors)
        pool = [u for u in pool if u in allowed]

    rows = [{
        "unit": treated_unit,
        "pre_rmspe": actual["pre_rmspe"],
        "post_rmspe": actual["post_rmspe"],
        "rmspe_ratio": actual["rmspe_ratio"],
        "is_treated": True,
    }]
    for placebo_unit in pool:
        placebo_donors = [u for u in pool if u != placebo_unit]
        try:
            fit = estimate_synthetic_control(
                panel, placebo_unit, treatment_time, outcome, unit, time,
                covariates, donors=placebo_donors,
            )
        except ValueError:
            continue
        rows.append({
            "unit": placebo_unit,
            "pre_rmspe": fit["pre_rmspe"],
            "post_rmspe": fit["post_rmspe"],
            "rmspe_ratio": fit["rmspe_ratio"],
            "is_treated": False,
        })

    results = pd.DataFrame(rows)
    kept = results
    if pre_rmspe_cutoff is not None:
        limit = pre_rmspe_cutoff * actual["pre_rmspe"]
        kept = results[results["is_treated"] | (results["pre_rmspe"] <= limit)]

    ratios = kept["rmspe_ratio"].to_numpy()
    treated_ratio = actual["rmspe_ratio"]
    p_value = float((ratios >= treated_ratio).sum() / len(ratios))

    results = results.sort_values("rmspe_ratio", ascending=False).reset_index(drop=True)
    results["included"] = results["unit"].isin(set(kept["unit"]))
    return results, p_value


#: Default pre-fit quality gate: drop an adopter whose pre-treatment RMSPE
#: exceeds this multiple of the median adopter's. A synthetic control that
#: does not track its unit before treatment has no counterfactual to
#: difference against, and its "effect" is fit error.
DEFAULT_MAX_PRE_RMSPE_RATIO = 2.0


def _fisher_combine(p_values):
    """Fisher's method: combine independent p-values into one chi-square test.

    The per-unit Abadie p-values are discrete and share a donor pool, so
    this is an approximation rather than an exact test — reported with its
    inputs so a reader can see what it rests on.
    """
    from scipy.stats import chi2

    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1.0)
    stat = float(-2.0 * np.log(p).sum())
    df = 2 * len(p)
    return {"statistic": stat, "df": df, "p_value": float(chi2.sf(stat, df))}


def aggregate_synthetic_control(
    panel,
    outcome="unemployment_rate",
    unit="state",
    time="year",
    cohort="adoption_year",
    covariates=None,
    min_pre=3,
    min_post=2,
    n_boot=1000,
    alpha=0.05,
    seed=0,
    max_pre_rmspe_ratio=DEFAULT_MAX_PRE_RMSPE_RATIO,
    permutation_inference=True,
):
    """Average synthetic control effect across every staggered adopter.

    Fits a separate synthetic control for each treated unit against the
    never-treated donor pool, then averages the mean post-treatment gap
    across the units that pass a pre-fit quality gate. This is what makes
    synthetic control comparable to the panel-wide estimators: a single
    case study estimates one state's ATT, not the average one.

    **Pre-fit filter.** `pre_rmspe` was previously recorded and never
    used, so an adopter whose synthetic control tracked it badly
    contributed to the ATT exactly as much as one it tracked well. Units
    whose pre-RMSPE exceeds `max_pre_rmspe_ratio` times the median
    adopter's are now dropped, and the dict reports how many and which.
    Pass `None` to keep everything.

    **Inference.** Two numbers, and they answer different questions:

    * `ci_lower`/`ci_upper` come from resampling the per-unit point
      effects with replacement. That treats each unit's effect as a known
      constant, so it captures dispersion *across* adopters but not the
      uncertainty *within* each fit, and it ignores the fact that all
      units draw on the same donor pool. It is a conditional interval,
      labelled as such in `ci_kind`, and it is narrower than a full
      accounting would be.
    * `permutation` runs Abadie inference per unit — refit pretending
      each donor was treated, rank the unit's post/pre RMSPE ratio in
      that distribution — and combines the per-unit p-values with
      Fisher's method. That is the inference this design actually
      supports. Set `permutation_inference=False` to skip it when speed
      matters.
    """
    never = panel[panel[cohort].isna()][unit].unique().tolist()
    if not never:
        raise ValueError("no never-treated units to serve as donors")

    adopters = (
        panel[panel[cohort].notna()]
        .drop_duplicates(subset=[unit])[[unit, cohort]]
        .itertuples(index=False)
    )
    times = np.sort(panel[time].unique())

    rows = []
    for name, adoption in adopters:
        adoption = float(adoption)
        if (times < adoption).sum() < min_pre or (times >= adoption).sum() < min_post:
            continue
        try:
            fit = estimate_synthetic_control(
                panel, name, adoption, outcome, unit, time, covariates, donors=never
            )
        except ValueError:
            continue
        post = fit["gaps"][fit["gaps"].index >= adoption]
        rows.append({
            unit: name,
            cohort: int(adoption),
            "effect": float(post.mean()),
            "pre_rmspe": fit["pre_rmspe"],
            "post_rmspe": fit["post_rmspe"],
            "rmspe_ratio": fit["rmspe_ratio"],
            "n_post": int(len(post)),
        })

    effects = pd.DataFrame(rows)
    if effects.empty:
        raise ValueError(
            f"no treated unit has >= {min_pre} pre and >= {min_post} post periods"
        )

    # --- pre-fit quality gate -------------------------------------------
    n_fitted = len(effects)
    median_pre_rmspe = float(effects["pre_rmspe"].median())
    if max_pre_rmspe_ratio is None:
        rmspe_cutoff = None
        effects["included"] = True
    else:
        rmspe_cutoff = max_pre_rmspe_ratio * median_pre_rmspe
        effects["included"] = effects["pre_rmspe"] <= rmspe_cutoff
    dropped = effects.loc[~effects["included"], unit].tolist()
    kept = effects[effects["included"]]
    if kept.empty:
        raise ValueError(
            f"the pre-RMSPE gate dropped all {n_fitted} adopters; "
            f"cutoff {rmspe_cutoff:.3f}"
        )

    values = kept["effect"].to_numpy()
    att = float(values.mean())

    # --- conditional interval across units ------------------------------
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    se = float(draws.std(ddof=1))
    z = _normal_quantile(1 - alpha / 2)

    result = {
        "att": att,
        "se": se,
        "ci_lower": att - z * se,
        "ci_upper": att + z * se,
        "ci_kind": "conditional-on-fitted-effects",
        "ci_note": (
            "resamples the per-unit point effects as if each were known exactly; "
            "ignores within-fit uncertainty and the shared donor pool, so it is "
            "narrower than a full accounting. See `permutation` for inference "
            "this design supports."
        ),
        "effects": effects.sort_values("effect").reset_index(drop=True),
        "n_units": len(values),
        "n_fitted": n_fitted,
        "n_dropped_pre_rmspe": len(dropped),
        "dropped_units": dropped,
        "median_pre_rmspe": median_pre_rmspe,
        "pre_rmspe_cutoff": rmspe_cutoff,
        "n_donors": len(never),
        "permutation": None,
    }

    if permutation_inference:
        result["permutation"] = _permutation_inference(
            panel, kept, outcome, unit, time, cohort, covariates, never
        )
    return result


def _permutation_inference(panel, kept, outcome, unit, time, cohort, covariates,
                           donors):
    """Per-unit Abadie permutation p-values, combined with Fisher's method."""
    per_unit = []
    for row in kept.itertuples(index=False):
        name = getattr(row, unit)
        adoption = float(getattr(row, cohort))
        try:
            _, p_value = placebo_test(
                panel, name, adoption, outcome, unit, time, covariates,
                donors=donors,
            )
        except ValueError:
            continue
        per_unit.append({unit: name, "p_value": float(p_value)})

    if not per_unit:
        return None
    table = pd.DataFrame(per_unit)
    combined = _fisher_combine(table["p_value"])
    return {
        "per_unit": table.sort_values("p_value").reset_index(drop=True),
        "combined": combined,
        "n_units": len(table),
        "n_significant_at_05": int((table["p_value"] <= 0.05).sum()),
        "method": "Abadie permutation over the never-treated donor pool, "
                  "per-unit p-values combined with Fisher's method",
    }


def _normal_quantile(p):
    from scipy.stats import norm

    return float(norm.ppf(p))


def run_synthetic_control(panel, treated_state, treatment_year, **kwargs):
    """Backwards-compatible flat output matching the old R script's CSV."""
    required = {"state", "year", "unemployment_rate"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    if treated_state not in panel["state"].unique():
        raise ValueError(f"treated_state {treated_state!r} not found in panel")
    res = estimate_synthetic_control(panel, treated_state, treatment_year, **kwargs)
    return res["paths"][["year", "state", "unemployment_rate", "type"]]


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")

    never = panel[panel["adoption_year"].isna()]["state"].unique().tolist()
    res = estimate_synthetic_control(
        panel, "WA" if "WA" in panel["state"].values else panel["state"].iloc[0],
        2016, donors=never,
    )
    print(f"\npre-RMSPE {res['pre_rmspe']:.3f}  post-RMSPE {res['post_rmspe']:.3f}")
    print("\nDonor weights:")
    print(res["weights"][res["weights"] > 0.01].to_string())
