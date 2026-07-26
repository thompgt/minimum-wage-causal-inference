"""Generate the analytical figures embedded in README.md.

Runs the repo's own estimators (src/methods, src/diagnostics) against the
panel returned by `src.data.loader.load_state_year_panel` — the real
processed panel if `data/processed/panel_state_year.parquet` exists,
otherwise the seeded synthetic panel. Writes PNGs to docs/images/.

Usage (from the project root):
    python -m scripts.make_figures
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_state_year_panel
from src.data.synthetic import TRUE_EFFECT
from src.diagnostics.parallel_trends import placebo_test
from src.diagnostics.robustness import leave_one_state_out
from src.methods.border_discontinuity import US_STATE_BORDER_PAIRS
from src.methods.callaway_santanna import estimate_att_gt
from src.methods.comparison import build_comparison
from src.methods.event_study import estimate_event_study
from src.methods.twfe_did import estimate_twfe

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"

# Categorical slots, assigned in fixed order (never cycled).
C_TREATED = "#2a78d6"
C_CONTROL = "#eb6834"
C_ACCENT = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(fig, name):
    fig.tight_layout()
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def fig_treated_vs_control(panel):
    """Mean unemployment path, ever-treated vs never-treated states."""
    df = panel.copy()
    df["ever_treated"] = df["adoption_year"].notna()
    means = df.groupby(["year", "ever_treated"])["unemployment_rate"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(means.index, means[True], color=C_TREATED, linewidth=2, label="Ever treated")
    ax.plot(means.index, means[False], color=C_CONTROL, linewidth=2, label="Never treated")
    last = means.index.max()
    ax.annotate("Ever treated", (last, means[True].iloc[-1]), color=C_TREATED,
                fontsize=9, xytext=(6, 0), textcoords="offset points", va="center")
    ax.annotate("Never treated", (last, means[False].iloc[-1]), color=C_CONTROL,
                fontsize=9, xytext=(6, 0), textcoords="offset points", va="center")
    _style(ax, "Mean unemployment rate: treated vs. never-treated states",
           "Year", "Unemployment rate (%)")
    ax.set_xlim(means.index.min(), last + 4)
    ax.set_xticks(range(int(means.index.min()), int(last) + 1, 3))
    _save(fig, "treated_vs_control.png")


def fig_parallel_trends(panel):
    """Event-time aligned outcome, treated states only, relative to adoption."""
    df = panel[panel["adoption_year"].notna()].copy()
    df["rel_time"] = (df["year"] - df["adoption_year"]).astype(int)
    df = df[df["rel_time"].between(-6, 6)]
    agg = df.groupby("rel_time")["unemployment_rate"].agg(["mean", "sem"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pre = agg[agg.index < 0]
    post = agg[agg.index >= 0]
    ax.fill_between(agg.index, agg["mean"] - 1.96 * agg["sem"],
                    agg["mean"] + 1.96 * agg["sem"], color=C_TREATED, alpha=0.15,
                    linewidth=0)
    ax.plot(pre.index, pre["mean"], color=C_TREATED, linewidth=2, marker="o",
            markersize=5, label="Pre-adoption")
    ax.plot(post.index, post["mean"], color=C_ACCENT, linewidth=2, marker="o",
            markersize=5, label="Post-adoption")
    ax.axvline(-0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("Adoption", (-0.5, ax.get_ylim()[1]), color=MUTED, fontsize=9,
                xytext=(4, -12), textcoords="offset points")
    _style(ax, "Outcome path around minimum wage adoption (treated states)",
           "Years relative to adoption", "Mean unemployment rate (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, "parallel_trends.png")


def fig_event_study(panel):
    """Event-study coefficients with 95% CIs (src/methods/event_study.py)."""
    _, summary = estimate_event_study(panel)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.axvline(-0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.errorbar(summary["rel_time"], summary["coef"],
                yerr=[summary["coef"] - summary["ci_lower"],
                      summary["ci_upper"] - summary["coef"]],
                fmt="o", markersize=7, color=C_TREATED, ecolor=C_TREATED,
                elinewidth=2, capsize=4, markeredgecolor="#fcfcfb",
                markeredgewidth=2)
    _style(ax, "Event study: unemployment relative to year before adoption\n"
               "(TWFE, 95% CI, state-clustered SEs)",
           "Years relative to adoption", "Coefficient (pp)")
    ax.set_xticks(sorted(summary["rel_time"].unique()))
    ax.annotate("reference\nperiod", (-1, 0), color=MUTED, fontsize=8,
                xytext=(0, -30), textcoords="offset points", ha="center")
    _save(fig, "event_study.png")
    return summary


def fig_method_comparison(panel, is_synthetic=False):
    """Every estimator's ATT on one scale, with 95% CIs.

    Scale conversion lives in src/methods/comparison.py: the semi-elasticity
    estimators are multiplied by the average treated log wage gap so they
    answer the same question as the binary-treatment ones.
    """
    df = build_comparison(
        panel, border_pairs=US_STATE_BORDER_PAIRS, n_boot=500
    ).dropna(subset=["estimate"])
    factor = df.attrs.get("log_wage_gap", float("nan"))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    colors = [C_ACCENT if s == "converted" else C_TREATED for s in df["scale"]]
    for yi, row, color in zip(y, df.itertuples(), colors):
        ax.errorbar(row.estimate, yi,
                    xerr=[[row.estimate - row.ci_lower], [row.ci_upper - row.estimate]],
                    fmt="o", markersize=8, color=color, ecolor=color,
                    elinewidth=2, capsize=4, markeredgecolor="#fcfcfb",
                    markeredgewidth=2)
        ax.annotate(f"{row.estimate:+.2f}", (row.estimate, yi), color=INK,
                    fontsize=9, xytext=(0, 12), textcoords="offset points",
                    ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_ylim(-0.6, len(df) - 0.4)

    handles = [
        plt.Line2D([], [], color=C_TREATED, marker="o", linestyle="",
                   label="Binary treatment (native scale)"),
        plt.Line2D([], [], color=C_ACCENT, marker="o", linestyle="",
                   label=f"Semi-elasticity x {factor:.3f} (converted)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK,
              loc="lower right")

    if is_synthetic:
        truth = TRUE_EFFECT * factor
        ax.axvline(truth, color=C_CONTROL, linewidth=2, linestyle="--")
        ax.annotate(f"ground truth ({truth:+.2f})", (truth, len(df) - 0.5),
                    color=C_CONTROL, fontsize=8, ha="center",
                    xytext=(0, 4), textcoords="offset points")

    _style(ax, "Effect of state minimum wage policy on unemployment, by method\n"
               "Average treatment effect on the treated, 95% CI, same panel throughout",
           "Effect on unemployment rate (pp)", "")
    ax.grid(axis="y", visible=False)
    _save(fig, "method_comparison.png")
    return df


def fig_cs_event_study(panel, n_boot=500):
    """Callaway-Sant'Anna dynamic aggregation, with a sup-t uniform band."""
    res = estimate_att_gt(panel, aggregations=("dynamic",), n_boot=n_boot)
    d = res["dynamic"]
    # Restrict to event times supported by more than one cohort; the tails
    # rest on a single state and say more about that state than the policy.
    d = d[(d["n_cohorts"] > 1) & d["event_time"].between(-8, 12)]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.axvline(-0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.fill_between(d["event_time"], d["band_lower"], d["band_upper"],
                    color=C_TREATED, alpha=0.12, linewidth=0)
    pre = d[d["event_time"] < 0]
    post = d[d["event_time"] >= 0]
    for part, color in ((pre, MUTED), (post, C_TREATED)):
        ax.errorbar(part["event_time"], part["att"],
                    yerr=[part["att"] - part["ci_lower"],
                          part["ci_upper"] - part["att"]],
                    fmt="o", markersize=6, color=color, ecolor=color,
                    elinewidth=1.8, capsize=3, markeredgecolor="#fcfcfb",
                    markeredgewidth=1.5)
    ax.annotate("pre-adoption\n(placebo)", (pre["event_time"].min(), 0),
                color=MUTED, fontsize=8, xytext=(0, -34),
                textcoords="offset points", ha="left")
    _style(ax, "Callaway-Sant'Anna event study\n"
               "ATT by years since adoption; pointwise 95% CI, shaded sup-t "
               "uniform band",
           "Years relative to adoption", "ATT on unemployment rate (pp)")
    _save(fig, "cs_event_study.png")
    return d


def fig_synthetic_control(panel, treated="MO", n_placebos=None):
    """Classic Abadie pair: actual vs synthetic path, and the placebo gaps."""
    from src.methods.synthetic_control import estimate_synthetic_control

    never = panel[panel["adoption_year"].isna()]["state"].unique().tolist()
    adoption = int(panel.loc[panel["state"] == treated, "adoption_year"].iloc[0])
    fit = estimate_synthetic_control(panel, treated, adoption, donors=never)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    paths = fit["paths"]
    actual = paths[paths["type"] == "actual"]
    synth = paths[paths["type"] == "synthetic"]
    ax1.plot(actual["year"], actual["unemployment_rate"], color=C_TREATED,
             linewidth=2.2, label=treated)
    ax1.plot(synth["year"], synth["unemployment_rate"], color=C_CONTROL,
             linewidth=2.2, linestyle="--", label=f"synthetic {treated}")
    ax1.axvline(adoption, color=MUTED, linewidth=1, linestyle=":")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)
    _style(ax1, f"{treated}: actual vs. synthetic counterfactual\n"
                f"pre-adoption RMSPE {fit['pre_rmspe']:.3f}",
           "Year", "Unemployment rate (%)")

    # Placebo gaps: refit treating each donor as if it had adopted.
    ax2.axhline(0, color="#c3c2b7", linewidth=1)
    for donor in never:
        others = [d for d in never if d != donor]
        try:
            pl = estimate_synthetic_control(panel, donor, adoption, donors=others)
        except ValueError:
            continue
        ax2.plot(pl["gaps"].index, pl["gaps"].to_numpy(), color=MUTED,
                 linewidth=0.9, alpha=0.45)
    ax2.plot(fit["gaps"].index, fit["gaps"].to_numpy(), color=C_TREATED,
             linewidth=2.4)
    ax2.axvline(adoption, color=MUTED, linewidth=1, linestyle=":")
    ax2.annotate(treated, (fit["gaps"].index[-1], fit["gaps"].iloc[-1]),
                 color=C_TREATED, fontsize=9, xytext=(4, 0),
                 textcoords="offset points", va="center")
    _style(ax2, f"Placebo gaps: {treated} vs. {len(never)} never-treated donors\n"
                "each grey line is a donor treated as if it had adopted",
           "Year", "Gap vs. synthetic (pp)")

    _save(fig, "synthetic_control.png")
    return fit


def fig_placebo(panel, n_placebos=60):
    """Placebo distribution from randomly reassigned treatment timing."""
    real = estimate_twfe(panel).params["log_minimum_wage"]
    placebos = placebo_test(panel, estimate_twfe, n_placebos=n_placebos)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(placebos, bins=18, color=C_CONTROL, alpha=0.75, edgecolor="#fcfcfb",
            linewidth=1.2, label=f"Placebo estimates (n={len(placebos)})")
    ax.axvline(real, color=C_TREATED, linewidth=2.5,
               label=f"Actual estimate ({real:+.3f})")
    pct = (placebos.abs() >= abs(real)).mean()
    _style(ax, "Placebo test: actual estimate vs. randomized treatment timing\n"
               f"{pct:.0%} of placebo draws are at least this large in magnitude",
           "TWFE coefficient on log(minimum wage)", "Placebo draws")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    _save(fig, "placebo_distribution.png")


def fig_leave_one_out(panel):
    """Leave-one-state-out sensitivity of the TWFE coefficient."""
    full = estimate_twfe(panel).params["log_minimum_wage"]
    loo = leave_one_state_out(panel, estimate_twfe).sort_values("coef_without_state")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(loo))
    ax.bar(x, loo["coef_without_state"], color=C_TREATED, width=0.7)
    ax.axhline(full, color=C_CONTROL, linewidth=2,
               label=f"Full-sample estimate ({full:+.3f})")
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(loo["dropped_state"], rotation=90, fontsize=8)
    _style(ax, "Leave-one-state-out: TWFE coefficient with each state dropped",
           "State dropped", "Coefficient on log(minimum wage)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.grid(axis="x", visible=False)
    _save(fig, "leave_one_state_out.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, is_synthetic = load_state_year_panel()
    print(f"Panel: {len(panel)} rows, {panel['state'].nunique()} states, "
          f"{'SYNTHETIC' if is_synthetic else 'real'} data")
    if is_synthetic:
        print(f"Ground-truth effect baked into the synthetic panel: {TRUE_EFFECT}")

    fig_treated_vs_control(panel)
    fig_parallel_trends(panel)
    fig_event_study(panel)
    fig_cs_event_study(panel)
    fig_synthetic_control(panel)
    fig_method_comparison(panel, is_synthetic)
    fig_placebo(panel)
    fig_leave_one_out(panel)
    print("done")


if __name__ == "__main__":
    main()
