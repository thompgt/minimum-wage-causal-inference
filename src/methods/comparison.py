"""Put every estimator on one scale so the comparison figure means something.

The estimators do not natively answer the same question. TWFE and the
border-discontinuity design regress unemployment on *log minimum wage*, so
their coefficients are semi-elasticities: percentage points of unemployment
per log point of minimum wage. Callaway-Sant'Anna, the event study, and
synthetic control estimate the effect of a *binary* treatment -- being a
state whose minimum wage binds above the federal floor -- so their estimates
are already in percentage points of unemployment.

Plotting those on a shared axis without conversion, as this repo previously
did, compares numbers that mean different things. Everything here is
converted to the **average treatment effect on the treated, in percentage
points**: the effect of the minimum wage policy treated states actually
enacted. The conversion multiplies a semi-elasticity by the average log gap
between the state and federal minimum among treated state-years, which for
the 2000-2022 panel is about 0.174 (a ~19% average premium over federal).

`scale` in the output records whether a row was converted, so a reader can
tell which estimates rest on that step.
"""
import numpy as np
import pandas as pd

from src.methods.border_discontinuity import border_pair_diffs, estimate_border_effect
from src.methods.callaway_santanna import estimate_att_gt
from src.methods.event_study import estimate_event_study
from src.methods.synthetic_control import aggregate_synthetic_control
from src.methods.twfe_did import estimate_twfe


def treated_log_wage_gap(panel, treatment="minimum_wage", federal="federal_minimum_wage"):
    """Mean log(state minimum) - log(federal minimum) over treated state-years.

    This is the size of the policy whose effect the binary-treatment
    estimators are measuring, and so the factor that converts a
    semi-elasticity into that same treatment effect.
    """
    if "treated" not in panel.columns:
        raise ValueError("panel needs a `treated` column; run build_panel first")
    treated = panel[panel["treated"].fillna(False).astype(bool)]
    if treated.empty:
        raise ValueError("panel has no treated state-years")
    if federal in panel.columns:
        gap = np.log(treated[treatment]) - np.log(treated[federal])
    else:
        gap = np.log(treated[treatment]) - np.log(treated[treatment].min())
    return float(gap.mean())


def build_comparison(
    panel,
    border_pairs=None,
    n_boot=500,
    include_synthetic_control=True,
    seed=0,
):
    """Run every estimator and return one row per method, all in pp of the ATT.

    `border_pairs` are contiguous state pairs for the border design; if
    omitted, the design is skipped rather than run on arbitrary pairs.
    Returns a DataFrame: method, estimate, ci_lower, ci_upper, scale, note.
    """
    factor = treated_log_wage_gap(panel)
    rows = []

    def add(method, est, lo, hi, scale, note):
        rows.append({
            "method": method, "estimate": est, "ci_lower": lo,
            "ci_upper": hi, "scale": scale, "note": note,
        })

    # --- TWFE: semi-elasticity, converted ------------------------------
    try:
        twfe = estimate_twfe(panel)
        ci = twfe.conf_int()
        add(
            "TWFE DiD",
            twfe.params["log_minimum_wage"] * factor,
            ci.loc["log_minimum_wage", "lower"] * factor,
            ci.loc["log_minimum_wage", "upper"] * factor,
            "converted",
            f"semi-elasticity x {factor:.3f} mean treated log wage gap",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the output table
        add("TWFE DiD", np.nan, np.nan, np.nan, "failed", str(exc))

    # --- Event study: binary treatment, already in pp -------------------
    try:
        _, es = estimate_event_study(panel)
        post = es[es["rel_time"] >= 0]
        add(
            "Event study\n(post-period avg)",
            post["coef"].mean(),
            post["ci_lower"].mean(),
            post["ci_upper"].mean(),
            "native",
            "mean of post-adoption event-time coefficients",
        )
    except Exception as exc:  # noqa: BLE001
        add("Event study\n(post-period avg)", np.nan, np.nan, np.nan, "failed", str(exc))

    # --- Callaway-Sant'Anna: binary treatment, already in pp ------------
    try:
        cs = estimate_att_gt(
            panel, aggregations=("simple",), n_boot=n_boot, seed=seed
        )["simple"].iloc[0]
        add(
            "Callaway-Sant'Anna",
            cs["att"], cs["ci_lower"], cs["ci_upper"],
            "native",
            f"overall ATT, {n_boot} cluster bootstrap draws",
        )
    except Exception as exc:  # noqa: BLE001
        add("Callaway-Sant'Anna", np.nan, np.nan, np.nan, "failed", str(exc))

    # --- Synthetic control: binary treatment, already in pp -------------
    if include_synthetic_control:
        try:
            sc = aggregate_synthetic_control(panel, n_boot=n_boot, seed=seed)
            add(
                "Synthetic control\n(avg over adopters)",
                sc["att"], sc["ci_lower"], sc["ci_upper"],
                "native",
                f"{sc['n_units']} adopters vs {sc['n_donors']} never-treated donors",
            )
        except Exception as exc:  # noqa: BLE001
            add("Synthetic control\n(avg over adopters)", np.nan, np.nan, np.nan,
                "failed", str(exc))

    # --- Border discontinuity: semi-elasticity, converted ---------------
    if border_pairs:
        try:
            diffs = border_pair_diffs(panel, border_pairs, treatment="log_minimum_wage")
            border = estimate_border_effect(diffs)
            b_ci = border.conf_int()
            add(
                "Border\ndiscontinuity",
                border.params["treatment_diff"] * factor,
                b_ci.loc["treatment_diff", 0] * factor,
                b_ci.loc["treatment_diff", 1] * factor,
                "converted",
                f"{len(border_pairs)} contiguous state pairs",
            )
        except Exception as exc:  # noqa: BLE001
            add("Border\ndiscontinuity", np.nan, np.nan, np.nan, "failed", str(exc))

    out = pd.DataFrame(rows)
    out.attrs["log_wage_gap"] = factor
    return out


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel
    from src.methods.border_discontinuity import US_STATE_BORDER_PAIRS

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")
    table = build_comparison(panel, border_pairs=US_STATE_BORDER_PAIRS, n_boot=300)
    print(f"\nconversion factor: {table.attrs['log_wage_gap']:.4f}\n")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(table.to_string(index=False))
