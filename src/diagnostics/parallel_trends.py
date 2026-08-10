"""Pre-trend and placebo diagnostics for the DiD/event-study design."""
import numpy as np
import pandas as pd

from src.diagnostics.robustness import DEFAULT_MAX_FAILURE_RATE, check_refit_failures


def pretrend_joint_test(event_study_summary):
    """Wald-style check: are pre-period event-study coefficients jointly ~0?

    Lightweight version using the reported per-coefficient CIs: flags any
    pre-period (rel_time < 0) coefficient whose CI excludes zero. For a
    formal joint F-test, refit with `statsmodels.stats.contrast` on the
    full covariance matrix; this function is a quick screening pass.
    """
    pre = event_study_summary[event_study_summary["rel_time"] < 0]
    violations = pre[(pre["ci_lower"] > 0) | (pre["ci_upper"] < 0)]
    return {
        "n_pre_periods": len(pre),
        "n_significant_violations": len(violations),
        "violating_periods": violations["rel_time"].tolist(),
        "passes": len(violations) == 0,
    }


def placebo_test(panel, estimate_fn, outcome="unemployment_rate",
                 treatment="log_minimum_wage", n_placebos=50, seed=0,
                 max_failure_rate=DEFAULT_MAX_FAILURE_RATE):
    """Re-run the estimator with randomly reassigned (fake) treatment timing.

    If the true effect is real, placebo estimates should center near zero
    and the actual estimate should be an outlier relative to this null
    distribution.

    Draws that fail to fit are counted, not skipped: the share of placebo
    draws exceeding the real estimate is only interpretable against a
    known denominator, so `n_attempted`, `n_failed`, `failure_rate` and
    `failures` are attached to the returned Series' `.attrs` and
    `RefitFailureError` is raised past `max_failure_rate`.
    """
    rng = np.random.default_rng(seed)
    placebo_coefs = []
    failures = []
    states = panel["state"].unique()
    # Whole-state treatment paths, looked up by (donor state, year). Assigning
    # state A the entire minimum wage history of state B keeps each path
    # internally coherent while breaking its link to A's labour market.
    lookup = panel.set_index(["state", "year"])[treatment]

    for draw in range(n_placebos):
        shuffle_map = dict(zip(states, rng.permutation(states), strict=True))
        shuffled = panel.copy()
        donor_index = pd.MultiIndex.from_arrays(
            [shuffled["state"].map(shuffle_map), shuffled["year"]]
        )
        shuffled[treatment] = lookup.reindex(donor_index).to_numpy()
        if shuffled[treatment].isna().any():
            failures.append(
                f"draw {draw}: donor state has no observation for some year "
                "(panel is unbalanced in the reassigned treatment)"
            )
            continue
        try:
            result = estimate_fn(shuffled)
        except Exception as exc:  # noqa: BLE001 - counted and surfaced below
            failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")
            continue
        placebo_coefs.append(result.params[treatment])

    out = pd.Series(placebo_coefs, name="placebo_coef", dtype=float)
    rate = check_refit_failures(
        "placebo_test", len(failures), n_placebos, max_failure_rate, failures
    )
    out.attrs.update(
        n_attempted=n_placebos, n_failed=len(failures),
        failure_rate=rate, failures=failures,
    )
    return out


def placebo_share_at_least_as_large(placebos, real_coef):
    """Share of placebo draws at least as large in magnitude as the real one.

    Reported alongside the denominator it is computed over, because the
    number that matters is the fraction of *attempted* draws — a bare
    percentage over an unknown number of survivors is not a p-value.
    """
    n = len(placebos)
    if n == 0:
        raise ValueError("no placebo draws to compare against")
    share = float((placebos.abs() >= abs(real_coef)).mean())
    return {
        "share": share,
        "n_used": n,
        "n_attempted": int(placebos.attrs.get("n_attempted", n)),
        "n_failed": int(placebos.attrs.get("n_failed", 0)),
    }


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel
    from src.methods.event_study import estimate_event_study
    from src.methods.twfe_did import estimate_twfe

    panel = generate_state_year_panel()
    _, summary = estimate_event_study(panel)
    print(pretrend_joint_test(summary))

    real_result = estimate_twfe(panel)
    real_coef = real_result.params["log_minimum_wage"]
    placebos = placebo_test(panel, estimate_twfe, n_placebos=20)
    print(f"\nReal coef: {real_coef:.4f}")
    print(f"Placebo mean: {placebos.mean():.4f}, std: {placebos.std():.4f}")
