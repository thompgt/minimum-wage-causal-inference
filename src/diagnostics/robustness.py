"""Robustness checks: bootstrap CIs and leave-one-state-out sensitivity.

Both routines refit the estimator many times, and a refit can fail — a
bootstrap draw that samples the same state repeatedly can leave a
collinear design, a leave-one-out subset can drop the only state in a
year. Silently skipping those failures shrinks the effective sample
without saying so, which makes any quantile or share computed over the
survivors describe an unknown denominator. Every routine here therefore
counts its failures, exposes them on the result, and raises once they
exceed `max_failure_rate` rather than quietly returning a thinner sample.
"""
import numpy as np
import pandas as pd

# A handful of failed refits out of hundreds is tolerable and worth
# reporting; a fifth of them means the specification, not the resample,
# is the problem.
DEFAULT_MAX_FAILURE_RATE = 0.1


class RefitFailureError(RuntimeError):
    """Too many refits failed for the surviving sample to mean anything."""


def check_refit_failures(kind, n_failed, n_attempted, max_failure_rate, examples):
    if n_attempted == 0:
        raise RefitFailureError(f"{kind}: nothing was attempted")
    rate = n_failed / n_attempted
    if rate > max_failure_rate:
        shown = "; ".join(str(e) for e in examples[:3])
        raise RefitFailureError(
            f"{kind}: {n_failed} of {n_attempted} refits failed "
            f"({rate:.0%} > {max_failure_rate:.0%} allowed). First failures: {shown}"
        )
    return rate


def bootstrap_state_cluster(panel, estimate_fn, treatment="log_minimum_wage",
                            n_boot=200, seed=0,
                            max_failure_rate=DEFAULT_MAX_FAILURE_RATE):
    """Cluster (state-level) bootstrap: resample states with replacement,
    refit, and return the distribution of the treatment coefficient.

    Clustering the resampling at the state level (rather than row level)
    respects within-state serial correlation, consistent with the
    clustered SEs used elsewhere in this project.

    The returned Series carries `n_attempted`, `n_failed`, `failure_rate`
    and `failures` in `.attrs`, so a caller quoting a quantile knows how
    many draws it is over. Raises `RefitFailureError` past
    `max_failure_rate`.
    """
    rng = np.random.default_rng(seed)
    states = panel["state"].unique()
    coefs = []
    failures = []
    for draw in range(n_boot):
        sampled_states = rng.choice(states, size=len(states), replace=True)
        parts = []
        for i, s in enumerate(sampled_states):
            chunk = panel[panel["state"] == s].copy()
            chunk["state"] = f"{s}_boot{i}"  # avoid duplicate-entity collisions
            parts.append(chunk)
        resampled = pd.concat(parts, ignore_index=True)
        try:
            result = estimate_fn(resampled)
        except Exception as exc:  # noqa: BLE001 - counted and surfaced below
            failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")
            continue
        coefs.append(result.params[treatment])

    out = pd.Series(coefs, name="bootstrap_coef", dtype=float)
    rate = check_refit_failures(
        "bootstrap_state_cluster", len(failures), n_boot, max_failure_rate, failures
    )
    out.attrs.update(
        n_attempted=n_boot, n_failed=len(failures), failure_rate=rate, failures=failures
    )
    return out


def leave_one_state_out(panel, estimate_fn, treatment="log_minimum_wage",
                        max_failure_rate=DEFAULT_MAX_FAILURE_RATE):
    """Refit dropping each state in turn; flags states whose removal
    swings the coefficient by an outsized amount (undue influence check).

    A state whose refit fails is reported in `.attrs["failed_states"]`
    rather than dropped in silence — "no state swings the estimate" is a
    much weaker claim if some states were never actually tried.
    """
    full_result = estimate_fn(panel)
    full_coef = full_result.params[treatment]

    states = panel["state"].unique()
    rows = []
    failures = []
    failed_states = []
    for state in states:
        subset = panel[panel["state"] != state]
        try:
            result = estimate_fn(subset)
        except Exception as exc:  # noqa: BLE001 - counted and surfaced below
            failed_states.append(state)
            failures.append(f"drop {state}: {type(exc).__name__}: {exc}")
            continue
        coef = result.params[treatment]
        rows.append({
            "dropped_state": state,
            "coef_without_state": coef,
            "delta_from_full": coef - full_coef,
        })

    rate = check_refit_failures(
        "leave_one_state_out", len(failures), len(states), max_failure_rate, failures
    )
    out = (
        pd.DataFrame(rows)
        .sort_values("delta_from_full", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    out.attrs.update(
        full_coef=float(full_coef),
        n_attempted=len(states),
        n_failed=len(failures),
        failure_rate=rate,
        failed_states=failed_states,
        failures=failures,
    )
    return out


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel
    from src.methods.twfe_did import estimate_twfe

    panel = generate_state_year_panel()
    boot = bootstrap_state_cluster(panel, estimate_twfe, n_boot=50)
    print(f"Bootstrap: mean={boot.mean():.4f}, std={boot.std():.4f}, "
          f"95% CI=({boot.quantile(0.025):.4f}, {boot.quantile(0.975):.4f})")
    print(f"  over {len(boot)} of {boot.attrs['n_attempted']} draws "
          f"({boot.attrs['n_failed']} failed)")

    loo = leave_one_state_out(panel, estimate_twfe)
    print(f"\nMost influential states ({loo.attrs['n_failed']} refits failed):")
    print(loo.head())
