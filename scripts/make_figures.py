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
from src.methods.border_discontinuity import border_pair_diffs, estimate_border_effect
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
    """Point estimate + 95% CI for each estimator that runs without R."""
    rows = []

    twfe = estimate_twfe(panel)
    ci = twfe.conf_int()
    rows.append({
        "method": "TWFE DiD",
        "coef": twfe.params["log_minimum_wage"],
        "lo": ci.loc["log_minimum_wage", "lower"],
        "hi": ci.loc["log_minimum_wage", "upper"],
    })

    # Post-adoption average of the event-study coefficients.
    _, es = estimate_event_study(panel)
    post = es[es["rel_time"] >= 0]
    rows.append({
        "method": "Event study\n(post-period avg)",
        "coef": post["coef"].mean(),
        "lo": post["ci_lower"].mean(),
        "hi": post["ci_upper"].mean(),
    })

    # Border discontinuity. Synthetic states have no geography, so pair them
    # arbitrarily (as src/methods/border_discontinuity.py's own demo does).
    states = sorted(panel["state"].unique())
    pairs = list(zip(states[::2], states[1::2]))
    diffs = border_pair_diffs(panel, pairs, treatment="log_minimum_wage")
    border = estimate_border_effect(diffs)
    b_ci = border.conf_int()
    rows.append({
        "method": "Border\ndiscontinuity",
        "coef": border.params["treatment_diff"],
        "lo": b_ci.loc["treatment_diff", 0],
        "hi": b_ci.loc["treatment_diff", 1],
    })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.errorbar(df["coef"], y,
                xerr=[df["coef"] - df["lo"], df["hi"] - df["coef"]],
                fmt="o", markersize=8, color=C_TREATED, ecolor=C_TREATED,
                elinewidth=2, capsize=4, markeredgecolor="#fcfcfb",
                markeredgewidth=2)
    for yi, (c, m) in zip(y, zip(df["coef"], df["method"])):
        ax.annotate(f"{c:+.3f}", (c, yi), color=INK, fontsize=9,
                    xytext=(0, 12), textcoords="offset points", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_ylim(-0.6, len(df) - 0.4)

    # Keep the informative estimates readable: clip the axis to the tighter
    # CIs and label any interval that runs off scale rather than hiding it.
    inner = pd.concat([df["lo"].abs(), df["hi"].abs()]).nsmallest(4).max()
    limit = max(inner * 1.35, 0.3)
    ax.set_xlim(-limit, limit)
    for yi, r in zip(y, df.itertuples()):
        if r.lo < -limit or r.hi > limit:
            ax.annotate(f"95% CI [{r.lo:+.2f}, {r.hi:+.2f}] — off scale",
                        (0, yi), color=MUTED, fontsize=8, ha="center",
                        xytext=(0, -18), textcoords="offset points")
    if is_synthetic:
        ax.axvline(TRUE_EFFECT, color=C_ACCENT, linewidth=2, linestyle="--",
                   label=f"Synthetic ground truth ({TRUE_EFFECT})")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    _style(ax, "Estimated effect by identification strategy (95% CI)\n"
               "Coefficient on log(minimum wage), same panel for every method",
           "Effect on unemployment rate (pp)", "")
    ax.grid(axis="y", visible=False)
    _save(fig, "method_comparison.png")
    return df


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
    fig_method_comparison(panel, is_synthetic)
    fig_placebo(panel)
    fig_leave_one_out(panel)
    print("done")


if __name__ == "__main__":
    main()
