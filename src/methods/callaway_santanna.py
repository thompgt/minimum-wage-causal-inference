"""Callaway & Sant'Anna (2021) staggered-adoption DiD, implemented directly.

Two-way fixed effects is biased when treatment timing is staggered and
effects vary over time: already-treated units get used as controls for
later adopters, and their own evolving treatment effects enter the
comparison with negative weight. Callaway-Sant'Anna avoids this by never
comparing a treated unit to an already-treated one. It estimates a separate
2x2 DiD for every (cohort g, period t) pair -- the *group-time average
treatment effect* ATT(g, t) -- and only then aggregates.

With no covariates the building block is the unconditional DiD

    ATT(g, t) = E[Y_t - Y_{g-1} | G = g] - E[Y_t - Y_{g-1} | control]

where the control group is either the never-treated units or the
not-yet-treated ones (units whose own adoption is still in the future).
This module uses a **universal base period**: every comparison is anchored
to g-1, the last period before cohort g is treated. Cells with t < g are
therefore pre-treatment placebo estimates and should be near zero if
parallel trends holds.

Inference is a **pairs cluster bootstrap over units**: units are resampled
with replacement, the whole estimator is refit on each resample, and
standard errors are the bootstrap standard deviation. This clusters at the
unit level, as the design requires. `simultaneous_band` additionally
returns a sup-t critical value, so the event-study band covers all event
times jointly rather than one at a time.

Reference: Callaway, B. and Sant'Anna, P. (2021), "Difference-in-Differences
with multiple time periods", Journal of Econometrics 225(2), 200-230.
"""
import numpy as np
import pandas as pd

CONTROL_GROUPS = ("nevertreated", "notyettreated")


def _reshape(panel, outcome, unit, time, cohort):
    """Wide (n_units x n_periods) outcome matrix plus per-unit cohort vector."""
    required = {unit, time, outcome, cohort}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    wide = panel.pivot_table(index=unit, columns=time, values=outcome, aggfunc="mean")
    complete = wide.notna().all(axis=1)
    if not complete.any():
        raise ValueError("no unit is observed in every period")
    dropped = wide.index[~complete].tolist()
    wide = wide[complete]

    cohorts = (
        panel.drop_duplicates(subset=[unit])
        .set_index(unit)[cohort]
        .reindex(wide.index)
    )
    G = pd.to_numeric(cohorts, errors="coerce").to_numpy(dtype=float)
    return wide.to_numpy(dtype=float), G, wide.index.to_numpy(), wide.columns.to_numpy(), dropped


def _cells(G, times):
    """The (g, t) pairs the design can identify, excluding the base period."""
    first = times[0]
    cohort_values = np.unique(G[np.isfinite(G)])
    # A cohort needs at least one pre-period, so g must be after the first
    # observed period; always-treated units are dropped here.
    usable = [g for g in cohort_values if g > first]
    return [(float(g), float(t)) for g in usable for t in times if t != g - 1], usable


def _att_gt_vector(Y, G, cells, time_pos, control_group):
    """ATT(g, t) for each cell, NaN where a cell has no treated or no controls."""
    out = np.full(len(cells), np.nan)
    for i, (g, t) in enumerate(cells):
        base = time_pos.get(g - 1)
        now = time_pos.get(t)
        if base is None or now is None:
            continue

        treated = G == g
        if control_group == "nevertreated":
            control = ~np.isfinite(G)
        else:  # notyettreated: never-treated, plus cohorts not yet adopted
            control = ~np.isfinite(G) | (G > max(t, g))
            control &= G != g

        if treated.sum() == 0 or control.sum() == 0:
            continue

        delta = Y[:, now] - Y[:, base]
        out[i] = delta[treated].mean() - delta[control].mean()
    return out


def _weights(G, cells, kind):
    """Cohort-size weights for one aggregation, as an (n_aggregates, n_cells) matrix.

    Returns (labels, weight matrix). Each row sums to 1 over the cells it
    uses; cells outside an aggregate get weight 0.
    """
    sizes = {g: float((G == g).sum()) for g, _ in cells}
    gs = np.array([c[0] for c in cells])
    ts = np.array([c[1] for c in cells])
    n = np.array([sizes[g] for g in gs])
    post = ts >= gs

    if kind == "simple":
        labels = ["overall"]
        masks = [post]
        w = n
    elif kind == "dynamic":
        events = np.unique(ts - gs)
        labels = [float(e) for e in events]
        masks = [(ts - gs) == e for e in events]
        w = n
    elif kind == "group":
        groups = np.unique(gs)
        labels = [float(g) for g in groups]
        masks = [(gs == g) & post for g in groups]
        w = np.ones_like(n)  # uniform over that cohort's post periods
    elif kind == "calendar":
        periods = np.unique(ts[post])
        labels = [float(t) for t in periods]
        masks = [(ts == t) & post for t in periods]
        w = n
    else:
        raise ValueError(
            f"unknown aggregation {kind!r}; expected one of "
            "simple, dynamic, group, calendar"
        )

    matrix = np.zeros((len(masks), len(cells)))
    for i, mask in enumerate(masks):
        row = np.where(mask, w, 0.0)
        matrix[i] = row
    return labels, matrix


def _apply_weights(att, matrix):
    """Weighted mean per row, renormalising over the cells that are non-NaN."""
    valid = np.isfinite(att)
    filled = np.where(valid, att, 0.0)
    w = matrix * valid
    denom = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, (w * filled).sum(axis=1) / denom, np.nan)
    return out


def estimate_att_gt(
    panel,
    outcome="unemployment_rate",
    unit="state",
    time="year",
    cohort="adoption_year",
    control_group="nevertreated",
    aggregations=("simple", "dynamic", "group"),
    n_boot=1000,
    alpha=0.05,
    seed=0,
):
    """Estimate ATT(g, t) and the requested aggregations with bootstrap SEs.

    Returns a dict with:
      `att_gt`      DataFrame of group, time, event_time, att, se, ci_lower, ci_upper
      `<agg>`       one DataFrame per requested aggregation
      `n_units`, `n_boot`, `control_group`, `dropped_units`

    Never-treated units are those with a missing `cohort` value.
    Always-treated units (adopting in or before the first observed period)
    have no pre-period and are excluded.
    """
    if control_group not in CONTROL_GROUPS:
        raise ValueError(
            f"control_group must be one of {CONTROL_GROUPS}, got {control_group!r}"
        )

    Y, G, units, times, dropped = _reshape(panel, outcome, unit, time, cohort)
    times = np.sort(times.astype(float))
    time_pos = {float(t): i for i, t in enumerate(times)}
    cells, usable = _cells(G, times)
    if not cells:
        raise ValueError(
            "no identifiable (cohort, period) cells -- every treated unit is "
            "always-treated, or there are no treated units"
        )

    att = _att_gt_vector(Y, G, cells, time_pos, control_group)
    agg_specs = {k: _weights(G, cells, k) for k in aggregations}
    agg_point = {k: _apply_weights(att, m) for k, (_, m) in agg_specs.items()}

    # Pairs cluster bootstrap: resample units, refit everything.
    rng = np.random.default_rng(seed)
    n = len(units)
    boot_att = np.full((n_boot, len(cells)), np.nan)
    boot_agg = {k: np.full((n_boot, m.shape[0]), np.nan) for k, (_, m) in agg_specs.items()}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Yb, Gb = Y[idx], G[idx]
        ab = _att_gt_vector(Yb, Gb, cells, time_pos, control_group)
        boot_att[b] = ab
        for k in agg_specs:
            _, mb = _weights(Gb, cells, k)
            boot_agg[k][b] = _apply_weights(ab, mb)

    z = _normal_quantile(1 - alpha / 2)
    att_se = _nan_std(boot_att)
    att_gt = pd.DataFrame({
        "group": [c[0] for c in cells],
        "time": [c[1] for c in cells],
        "event_time": [c[1] - c[0] for c in cells],
        "att": att,
        "se": att_se,
        "ci_lower": att - z * att_se,
        "ci_upper": att + z * att_se,
    }).sort_values(["group", "time"]).reset_index(drop=True)

    result = {
        "att_gt": att_gt,
        "n_units": int(n),
        "n_treated_cohorts": len(usable),
        "n_boot": int(n_boot),
        "control_group": control_group,
        "dropped_units": dropped,
    }
    cell_groups = np.array([c[0] for c in cells])
    for k, (labels, matrix) in agg_specs.items():
        point = agg_point[k]
        se = _nan_std(boot_agg[k])
        crit = _sup_t_critical(boot_agg[k], point, se, alpha)
        # How much each aggregate actually rests on. Event times far from
        # zero are often a single cohort, and that belongs in the output.
        used = (matrix > 0) & np.isfinite(att)
        n_cohorts = [len(np.unique(cell_groups[row])) for row in used]
        n_units = [
            int(sum((G == g).sum() for g in np.unique(cell_groups[row])))
            for row in used
        ]
        result[k] = pd.DataFrame({
            _label_column(k): labels,
            "att": point,
            "se": se,
            "ci_lower": point - z * se,
            "ci_upper": point + z * se,
            "band_lower": point - crit * se,
            "band_upper": point + crit * se,
            "n_cohorts": n_cohorts,
            "n_units": n_units,
        })
    return result


def _label_column(kind):
    return {
        "simple": "aggregation",
        "dynamic": "event_time",
        "group": "group",
        "calendar": "time",
    }[kind]


def _nan_std(draws):
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(draws).sum(axis=0)
        se = np.nanstd(draws, axis=0, ddof=1)
    return np.where(counts > 1, se, np.nan)


def _normal_quantile(p):
    from scipy.stats import norm

    return float(norm.ppf(p))


def _sup_t_critical(draws, point, se, alpha):
    """Critical value for a band covering every element simultaneously."""
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.abs(draws - point) / np.where(se > 0, se, np.nan)
    sup = np.nanmax(np.where(np.isfinite(t), t, np.nan), axis=1)
    sup = sup[np.isfinite(sup)]
    if sup.size == 0:
        return _normal_quantile(1 - alpha / 2)
    return float(np.quantile(sup, 1 - alpha))


def run_callaway_santanna(panel, **kwargs):
    """Backwards-compatible flat output: ATT(g,t) rows plus an `overall` row.

    Mirrors the CSV contract the previous R implementation wrote, so callers
    that just want a single table keep working.
    """
    res = estimate_att_gt(panel, aggregations=("simple",), **kwargs)
    gt = res["att_gt"][["group", "time", "att", "se", "ci_lower", "ci_upper"]].copy()
    overall = res["simple"].iloc[0]
    gt.loc[len(gt)] = {
        "group": np.nan,
        "time": np.nan,
        "att": overall["att"],
        "se": overall["se"],
        "ci_lower": overall["ci_lower"],
        "ci_upper": overall["ci_upper"],
    }
    return gt


if __name__ == "__main__":
    from src.data.loader import load_state_year_panel

    panel, is_synthetic = load_state_year_panel()
    print(f"panel: {'synthetic' if is_synthetic else 'real'}, {len(panel)} rows")
    res = estimate_att_gt(panel, n_boot=200)
    print(f"\n{res['n_units']} units, {res['n_treated_cohorts']} cohorts, "
          f"control = {res['control_group']}")
    print("\nOverall ATT:")
    print(res["simple"].to_string(index=False))
    print("\nEvent study:")
    print(res["dynamic"].to_string(index=False))
