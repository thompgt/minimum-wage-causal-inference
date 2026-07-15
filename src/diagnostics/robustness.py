"""Robustness checks: bootstrap CIs and leave-one-state-out sensitivity."""
import numpy as np
import pandas as pd


def bootstrap_state_cluster(panel, estimate_fn, treatment="log_minimum_wage",
                             n_boot=200, seed=0):
    """Cluster (state-level) bootstrap: resample states with replacement,
    refit, and return the distribution of the treatment coefficient.

    Clustering the resampling at the state level (rather than row level)
    respects within-state serial correlation, consistent with the
    clustered SEs used elsewhere in this project.
    """
    rng = np.random.default_rng(seed)
    states = panel["state"].unique()
    coefs = []
    for _ in range(n_boot):
        sampled_states = rng.choice(states, size=len(states), replace=True)
        parts = []
        for i, s in enumerate(sampled_states):
            chunk = panel[panel["state"] == s].copy()
            chunk["state"] = f"{s}_boot{i}"  # avoid duplicate-entity collisions
            parts.append(chunk)
        resampled = pd.concat(parts, ignore_index=True)
        try:
            result = estimate_fn(resampled)
            coefs.append(result.params[treatment])
        except Exception:
            continue
    return pd.Series(coefs, name="bootstrap_coef")


def leave_one_state_out(panel, estimate_fn, treatment="log_minimum_wage"):
    """Refit dropping each state in turn; flags states whose removal
    swings the coefficient by an outsized amount (undue influence check).
    """
    full_result = estimate_fn(panel)
    full_coef = full_result.params[treatment]

    rows = []
    for state in panel["state"].unique():
        subset = panel[panel["state"] != state]
        try:
            result = estimate_fn(subset)
            coef = result.params[treatment]
        except Exception:
            continue
        rows.append({
            "dropped_state": state,
            "coef_without_state": coef,
            "delta_from_full": coef - full_coef,
        })
    return pd.DataFrame(rows).sort_values(
        "delta_from_full", key=abs, ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel
    from src.methods.twfe_did import estimate_twfe

    panel = generate_state_year_panel()
    boot = bootstrap_state_cluster(panel, estimate_twfe, n_boot=50)
    print(f"Bootstrap: mean={boot.mean():.4f}, std={boot.std():.4f}, "
          f"95% CI=({boot.quantile(0.025):.4f}, {boot.quantile(0.975):.4f})")

    loo = leave_one_state_out(panel, estimate_twfe)
    print("\nMost influential states:")
    print(loo.head())
