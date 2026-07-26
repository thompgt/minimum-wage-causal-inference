"""Tests for the synthetic control estimator.

The core checks are that the simplex constraint actually holds, that a
donor combination which reproduces the treated unit pre-treatment is
found, and that a known post-treatment shock comes back out of the gaps.
"""
import numpy as np
import pandas as pd
import pytest

from src.methods.synthetic_control import (
    estimate_synthetic_control,
    placebo_test,
    run_synthetic_control,
)

YEARS = list(range(2000, 2020))
TREAT_YEAR = 2012


def make_panel(shock=0.0, n_donors=6, seed=0, mix=None):
    """Treated unit is an exact convex mix of donors, plus a post shock."""
    rng = np.random.default_rng(seed)
    donor_paths = {
        f"D{i}": 4 + rng.normal(0, 1) + np.cumsum(rng.normal(0, 0.3, len(YEARS)))
        for i in range(n_donors)
    }
    if mix is None:
        mix = {"D0": 0.6, "D1": 0.4}
    treated = sum(w * donor_paths[d] for d, w in mix.items())
    treated = treated + shock * (np.array(YEARS) >= TREAT_YEAR)

    rows = []
    for name, path in donor_paths.items():
        for y, v in zip(YEARS, path):
            rows.append({"state": name, "year": y, "unemployment_rate": v,
                         "minimum_wage": 7.25})
    for y, v in zip(YEARS, treated):
        rows.append({"state": "T", "year": y, "unemployment_rate": v,
                     "minimum_wage": 8.5})
    return pd.DataFrame(rows)


def test_weights_lie_on_the_simplex():
    res = estimate_synthetic_control(make_panel(), "T", TREAT_YEAR)
    w = res["weights"]
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_recovers_the_generating_donor_mix():
    res = estimate_synthetic_control(make_panel(), "T", TREAT_YEAR)
    w = res["weights"]
    assert w["D0"] == pytest.approx(0.6, abs=0.05)
    assert w["D1"] == pytest.approx(0.4, abs=0.05)
    assert res["pre_rmspe"] < 1e-3


def test_recovers_known_post_treatment_shock():
    shock = 1.5
    res = estimate_synthetic_control(make_panel(shock=shock), "T", TREAT_YEAR)
    post = res["gaps"][res["gaps"].index >= TREAT_YEAR]
    assert post.mean() == pytest.approx(shock, abs=0.1)


def test_no_shock_means_no_post_gap():
    res = estimate_synthetic_control(make_panel(shock=0.0), "T", TREAT_YEAR)
    post = res["gaps"][res["gaps"].index >= TREAT_YEAR]
    assert post.abs().max() < 0.05


def test_single_perfect_donor_gets_all_the_weight():
    res = estimate_synthetic_control(
        make_panel(mix={"D3": 1.0}), "T", TREAT_YEAR
    )
    assert res["weights"]["D3"] == pytest.approx(1.0, abs=0.02)


def test_synthetic_beats_the_plain_donor_average_pre_treatment():
    panel = make_panel()
    res = estimate_synthetic_control(panel, "T", TREAT_YEAR)
    wide = panel.pivot_table(index="state", columns="year", values="unemployment_rate")
    pre = [y for y in YEARS if y < TREAT_YEAR]
    naive = wide.drop(index="T")[pre].mean(axis=0).to_numpy()
    naive_rmspe = float(np.sqrt(np.mean((wide.loc["T", pre].to_numpy() - naive) ** 2)))
    assert res["pre_rmspe"] < naive_rmspe


def test_covariates_path_runs_and_stays_on_simplex():
    res = estimate_synthetic_control(
        make_panel(), "T", TREAT_YEAR, covariates=["minimum_wage"]
    )
    assert res["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert "minimum_wage" in res["balance"].index


def test_donor_pool_is_respected():
    res = estimate_synthetic_control(
        make_panel(), "T", TREAT_YEAR, donors=["D2", "D3", "D4"]
    )
    assert set(res["weights"].index) == {"D2", "D3", "D4"}
    assert res["n_donors"] == 3


def test_placebo_test_returns_a_valid_p_value():
    results, p = placebo_test(make_panel(shock=2.0), "T", TREAT_YEAR)
    assert 0 < p <= 1
    assert results["is_treated"].sum() == 1
    # A large shock should put the treated unit at or near the top.
    assert results.iloc[0]["unit"] == "T"


def test_rejects_missing_treated_unit():
    with pytest.raises(ValueError, match="not in the panel"):
        estimate_synthetic_control(make_panel(), "NOPE", TREAT_YEAR)


def test_rejects_empty_donor_pool():
    with pytest.raises(ValueError, match="donor pool is empty"):
        estimate_synthetic_control(make_panel(), "T", TREAT_YEAR, donors=[])


def test_rejects_too_few_pre_periods():
    with pytest.raises(ValueError, match="at least 2 pre-treatment periods"):
        estimate_synthetic_control(make_panel(), "T", 2001)


def test_rejects_no_post_periods():
    with pytest.raises(ValueError, match="no post-treatment periods"):
        estimate_synthetic_control(make_panel(), "T", 2100)


def test_flat_output_contract():
    """run_synthetic_control keeps the columns the old R script emitted."""
    out = run_synthetic_control(make_panel(), "T", TREAT_YEAR)
    assert list(out.columns) == ["year", "state", "unemployment_rate", "type"]
    assert set(out["type"]) == {"actual", "synthetic"}
    assert len(out) == 2 * len(YEARS)


def test_flat_output_validates_columns():
    bad = pd.DataFrame({"state": ["S01"], "year": [2020]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_synthetic_control(bad, "S01", 2015)


def test_flat_output_validates_treated_state_present():
    with pytest.raises(ValueError, match="not found in panel"):
        run_synthetic_control(make_panel(), "S99", TREAT_YEAR)
