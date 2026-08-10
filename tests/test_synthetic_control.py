"""Tests for the synthetic control estimator.

The core checks are that the simplex constraint actually holds, that a
donor combination which reproduces the treated unit pre-treatment is
found, and that a known post-treatment shock comes back out of the gaps.
"""
import numpy as np
import pandas as pd
import pytest

from src.methods.synthetic_control import (
    DEFAULT_MAX_PRE_RMSPE_RATIO,
    _fisher_combine,
    aggregate_synthetic_control,
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
        for y, v in zip(YEARS, path, strict=True):
            rows.append({"state": name, "year": y, "unemployment_rate": v,
                         "minimum_wage": 7.25})
    for y, v in zip(YEARS, treated, strict=True):
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


# --- aggregation across adopters ------------------------------------------

def make_staggered_panel(seed=11, n_donors=6, n_adopters=4, bad_fit_unit=True):
    """Never-treated donors plus several staggered adopters.

    One adopter (`BAD`) is deliberately unreproducible from the donor pool
    — a large idiosyncratic wander — so the pre-RMSPE gate has something
    to catch.
    """
    rng = np.random.default_rng(seed)
    rows = []
    donor_paths = {}
    for i in range(n_donors):
        path = 5 + rng.normal(0, 0.5) + np.cumsum(rng.normal(0, 0.2, len(YEARS)))
        donor_paths[f"D{i}"] = path
        for y, v in zip(YEARS, path, strict=True):
            rows.append({"state": f"D{i}", "year": y, "unemployment_rate": v,
                         "minimum_wage": 7.25, "adoption_year": pd.NA})

    for a in range(n_adopters):
        adoption = 2008 + 2 * a
        mix = {"D0": 0.5, "D1": 0.3, "D2": 0.2}
        path = sum(w * donor_paths[d] for d, w in mix.items())
        path = path + 0.6 * (np.array(YEARS) >= adoption)
        for y, v in zip(YEARS, path, strict=True):
            rows.append({"state": f"A{a}", "year": y, "unemployment_rate": v,
                         "minimum_wage": 9.0, "adoption_year": adoption})

    if bad_fit_unit:
        wander = 5 + np.cumsum(rng.normal(0, 2.5, len(YEARS)))
        for y, v in zip(YEARS, wander, strict=True):
            rows.append({"state": "BAD", "year": y, "unemployment_rate": v,
                         "minimum_wage": 9.0, "adoption_year": 2012})

    df = pd.DataFrame(rows)
    df["adoption_year"] = df["adoption_year"].astype("Int64")
    return df


def test_aggregate_reports_the_prefit_gate_it_applied():
    panel = make_staggered_panel()
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=False)
    assert res["n_fitted"] >= res["n_units"]
    assert res["n_dropped_pre_rmspe"] == res["n_fitted"] - res["n_units"]
    assert res["pre_rmspe_cutoff"] == pytest.approx(
        DEFAULT_MAX_PRE_RMSPE_RATIO * res["median_pre_rmspe"]
    )


def test_prefit_gate_drops_a_unit_the_donors_cannot_reproduce():
    panel = make_staggered_panel(bad_fit_unit=True)
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=False)
    assert "BAD" in res["dropped_units"]
    assert res["n_dropped_pre_rmspe"] >= 1


def test_prefit_gate_can_be_disabled():
    panel = make_staggered_panel()
    gated = aggregate_synthetic_control(panel, n_boot=100,
                                        permutation_inference=False)
    ungated = aggregate_synthetic_control(panel, n_boot=100,
                                          max_pre_rmspe_ratio=None,
                                          permutation_inference=False)
    assert ungated["n_dropped_pre_rmspe"] == 0
    assert ungated["n_units"] > gated["n_units"]
    # A badly fitting unit contributes its fit error to the ATT, which is
    # exactly why the gate exists.
    assert ungated["att"] != gated["att"]


def test_effects_table_marks_which_units_were_included():
    panel = make_staggered_panel()
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=False)
    effects = res["effects"]
    assert "included" in effects.columns
    assert effects["included"].sum() == res["n_units"]
    assert set(effects.loc[~effects["included"], "state"]) == set(res["dropped_units"])


def test_ci_is_labelled_conditional():
    """The interval ignores within-fit uncertainty; it must say so."""
    panel = make_staggered_panel()
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=False)
    assert res["ci_kind"] == "conditional-on-fitted-effects"
    assert "within-fit uncertainty" in res["ci_note"]
    assert res["ci_lower"] < res["att"] < res["ci_upper"]


def test_permutation_inference_runs_per_unit_and_combines():
    panel = make_staggered_panel()
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=True)
    perm = res["permutation"]
    assert perm is not None
    assert perm["n_units"] == res["n_units"]
    assert set(perm["per_unit"].columns) == {"state", "p_value"}
    assert perm["per_unit"]["p_value"].between(0, 1).all()
    assert 0.0 <= perm["combined"]["p_value"] <= 1.0
    assert perm["combined"]["df"] == 2 * perm["n_units"]


def test_permutation_inference_is_skippable():
    panel = make_staggered_panel()
    res = aggregate_synthetic_control(panel, n_boot=100, permutation_inference=False)
    assert res["permutation"] is None


def test_fisher_combine_is_uniform_on_a_single_pvalue():
    """Fisher's method on one p-value returns that p-value back."""
    for p in [0.01, 0.2, 0.5, 0.9]:
        assert _fisher_combine([p])["p_value"] == pytest.approx(p, rel=1e-9)


def test_fisher_combine_gets_smaller_as_evidence_accumulates():
    one = _fisher_combine([0.05])["p_value"]
    many = _fisher_combine([0.05] * 5)["p_value"]
    assert many < one


def test_aggregate_requires_never_treated_donors():
    panel = make_staggered_panel()
    panel["adoption_year"] = panel["adoption_year"].fillna(2005).astype("Int64")
    with pytest.raises(ValueError, match="no never-treated units"):
        aggregate_synthetic_control(panel, permutation_inference=False)
