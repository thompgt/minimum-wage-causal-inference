"""Pre-trend and placebo diagnostics for the DiD/event-study design."""
import numpy as np
import pandas as pd


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
                  treatment="log_minimum_wage", n_placebos=50, seed=0):
    """Re-run the estimator with randomly reassigned (fake) treatment timing.

    If the true effect is real, placebo estimates should center near zero
    and the actual estimate should be an outlier relative to this null
    distribution.
    """
    rng = np.random.default_rng(seed)
    placebo_coefs = []
    states = panel["state"].unique()

    for _ in range(n_placebos):
        shuffled = panel.copy()
        shuffle_map = dict(zip(states, rng.permutation(states)))
        shuffled[treatment] = shuffled.apply(
            lambda r: panel.loc[
                (panel["state"] == shuffle_map[r["state"]]) & (panel["year"] == r["year"]),
                treatment,
            ].values[0],
            axis=1,
        )
        try:
            result = estimate_fn(shuffled)
            placebo_coefs.append(result.params[treatment])
        except Exception:
            continue

    return pd.Series(placebo_coefs, name="placebo_coef")


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
