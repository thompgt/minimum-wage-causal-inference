"""Tests for the Callaway-Sant'Anna estimator.

The important ones are recovery tests: build a panel whose true ATT is
known by construction and check the estimator returns it. The last test is
the reason the method exists at all -- it constructs a case where TWFE is
biased by staggered timing with heterogeneous effects, and shows CS is not.
"""
import numpy as np
import pandas as pd
import pytest

from src.methods.callaway_santanna import estimate_att_gt, run_callaway_santanna


def make_panel(
    cohort_effects,
    n_never=20,
    n_per_cohort=10,
    years=range(2000, 2020),
    noise=0.05,
    seed=0,
):
    """Panel with unit FE, year FE, and a known additive effect once treated.

    `cohort_effects` maps adoption_year -> true ATT for that cohort.
    """
    rng = np.random.default_rng(seed)
    years = list(years)
    rows = []

    def emit(unit, adoption):
        unit_fe = rng.normal(5, 1)
        for y in years:
            year_fe = 0.1 * (y - years[0])
            effect = (
                cohort_effects[adoption]
                if adoption is not None and y >= adoption
                else 0.0
            )
            rows.append({
                "state": unit,
                "year": y,
                "unemployment_rate": unit_fe + year_fe + effect + rng.normal(0, noise),
                "adoption_year": adoption,
            })

    for i in range(n_never):
        emit(f"N{i:02d}", None)
    for g, _ in cohort_effects.items():
        for i in range(n_per_cohort):
            emit(f"T{g}_{i:02d}", g)

    df = pd.DataFrame(rows)
    df["adoption_year"] = df["adoption_year"].astype("Int64")
    return df


def test_recovers_known_constant_effect():
    true_att = -0.8
    panel = make_panel({2008: true_att, 2012: true_att})
    res = estimate_att_gt(panel, n_boot=100, aggregations=("simple",))
    assert res["simple"]["att"].iloc[0] == pytest.approx(true_att, abs=0.05)


def test_pre_period_effects_are_zero():
    panel = make_panel({2008: -0.8, 2012: -0.8})
    res = estimate_att_gt(panel, n_boot=100, aggregations=("dynamic",))
    dynamic = res["dynamic"]
    pre = dynamic[(dynamic["event_time"] < 0) & (dynamic["n_cohorts"] > 1)]
    assert pre["att"].abs().max() < 0.1


def test_recovers_cohort_specific_effects():
    panel = make_panel({2008: -1.5, 2012: 0.5})
    res = estimate_att_gt(panel, n_boot=100, aggregations=("group",))
    by_group = res["group"].set_index("group")["att"]
    assert by_group[2008.0] == pytest.approx(-1.5, abs=0.05)
    assert by_group[2012.0] == pytest.approx(0.5, abs=0.05)


def test_unbiased_where_twfe_is_biased():
    """Staggered timing + heterogeneous effects: the case TWFE gets wrong."""
    from linearmodels.panel import PanelOLS

    effects = {2005: -2.0, 2015: 0.5}
    panel = make_panel(effects, years=range(2000, 2020), noise=0.02)
    panel["treated"] = (
        panel["adoption_year"].notna() & (panel["year"] >= panel["adoption_year"])
    ).astype(int)

    # Truth: equal cohort sizes, so the average post-treatment effect is the
    # cohort-size-weighted mean of each cohort's effect over its own post
    # periods, which CS's `simple` aggregation targets.
    cs = estimate_att_gt(panel, n_boot=100, aggregations=("simple",))
    cs_att = cs["simple"]["att"].iloc[0]

    twfe = PanelOLS.from_formula(
        "unemployment_rate ~ treated + EntityEffects + TimeEffects",
        data=panel.set_index(["state", "year"]),
    ).fit()
    twfe_att = float(twfe.params["treated"])

    # CS lands between the two cohort effects; TWFE is dragged outside that
    # range by using the early cohort as a control for the late one.
    assert -2.0 < cs_att < 0.5
    assert abs(twfe_att - cs_att) > 0.2


def test_always_treated_units_are_excluded():
    panel = make_panel({2000: -1.0, 2010: -1.0})  # 2000 == first period
    res = estimate_att_gt(panel, n_boot=50, aggregations=("group",))
    assert set(res["group"]["group"]) == {2010.0}
    assert res["n_treated_cohorts"] == 1


def test_not_yet_treated_controls_agree_with_never_treated():
    panel = make_panel({2008: -0.8, 2012: -0.8})
    never = estimate_att_gt(
        panel, n_boot=50, aggregations=("simple",), control_group="nevertreated"
    )["simple"]["att"].iloc[0]
    notyet = estimate_att_gt(
        panel, n_boot=50, aggregations=("simple",), control_group="notyettreated"
    )["simple"]["att"].iloc[0]
    assert never == pytest.approx(notyet, abs=0.05)


def test_rejects_unknown_control_group():
    panel = make_panel({2008: -0.5})
    with pytest.raises(ValueError, match="control_group must be one of"):
        estimate_att_gt(panel, control_group="everyone")


def test_rejects_unknown_aggregation():
    panel = make_panel({2008: -0.5})
    with pytest.raises(ValueError, match="unknown aggregation"):
        estimate_att_gt(panel, n_boot=10, aggregations=("nonsense",))


def test_validates_required_columns():
    bad = pd.DataFrame({"state": ["S01"], "year": [2020]})
    with pytest.raises(ValueError, match="missing required columns"):
        estimate_att_gt(bad)


def test_raises_when_no_cohort_has_a_pre_period():
    panel = make_panel({2000: -1.0})  # only cohort is always-treated
    with pytest.raises(ValueError, match="no identifiable"):
        estimate_att_gt(panel, n_boot=10)


def test_flat_output_contract():
    """run_callaway_santanna keeps the columns the old R script emitted."""
    panel = make_panel({2008: -0.8, 2012: -0.8})
    out = run_callaway_santanna(panel, n_boot=50)
    assert list(out.columns) == ["group", "time", "att", "se", "ci_lower", "ci_upper"]
    overall = out.iloc[-1]
    assert pd.isna(overall["group"]) and pd.isna(overall["time"])
    assert overall["att"] == pytest.approx(-0.8, abs=0.05)


def test_bootstrap_is_seeded():
    panel = make_panel({2008: -0.8})
    a = estimate_att_gt(panel, n_boot=50, seed=7)["att_gt"]["se"]
    b = estimate_att_gt(panel, n_boot=50, seed=7)["att_gt"]["se"]
    pd.testing.assert_series_equal(a, b)
