"""Tests for the module that produces the repo's headline table.

`comparison.py` is what the README, the figures and the app all quote, and
it was the only estimator-facing module with no tests at all. Two things
are worth pinning: the conversion factor, which silently rescales two of
the five rows, and the row contract, because every estimator is wrapped in
a bare `except` that turns a crash into a row rather than a failure. A
broken estimator is supposed to be visible in the table, not invisible.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_state_year_panel
from src.methods import comparison as cmp_mod
from src.methods.comparison import build_comparison, treated_log_wage_gap

EXPECTED_COLUMNS = ["method", "estimate", "ci_lower", "ci_upper", "scale", "note"]
VALID_SCALES = {"native", "converted", "conditional", "failed"}


@pytest.fixture(scope="module")
def panel():
    return generate_state_year_panel(seed=11)


# --- the conversion factor ------------------------------------------------

def test_gap_is_the_mean_log_ratio_over_treated_rows_only():
    """Hand-computable: two treated rows at 2x and 4x the federal floor."""
    panel = pd.DataFrame({
        "state": ["A", "A", "B", "B"],
        "year": [2020, 2021, 2020, 2021],
        "minimum_wage": [10.0, 20.0, 5.0, 5.0],
        "federal_minimum_wage": [5.0, 5.0, 5.0, 5.0],
        "treated": [True, True, False, False],
    })
    expected = (np.log(2.0) + np.log(4.0)) / 2
    assert treated_log_wage_gap(panel) == pytest.approx(expected)


def test_gap_ignores_untreated_rows_however_high_their_wage():
    """An untreated state's wage must not move the factor."""
    base = pd.DataFrame({
        "state": ["A", "B"],
        "year": [2020, 2020],
        "minimum_wage": [10.0, 5.0],
        "federal_minimum_wage": [5.0, 5.0],
        "treated": [True, False],
    })
    louder = base.copy()
    louder.loc[1, "minimum_wage"] = 99.0
    assert treated_log_wage_gap(base) == pytest.approx(treated_log_wage_gap(louder))


def test_gap_needs_a_treated_column():
    panel = pd.DataFrame({"minimum_wage": [10.0], "federal_minimum_wage": [5.0]})
    with pytest.raises(ValueError, match="needs a `treated` column"):
        treated_log_wage_gap(panel)


def test_gap_needs_at_least_one_treated_row():
    panel = pd.DataFrame({
        "minimum_wage": [5.0], "federal_minimum_wage": [5.0], "treated": [False],
    })
    with pytest.raises(ValueError, match="no treated state-years"):
        treated_log_wage_gap(panel)


def test_gap_falls_back_to_the_panel_minimum_without_a_federal_column():
    panel = pd.DataFrame({
        "state": ["A", "B"], "year": [2020, 2020],
        "minimum_wage": [10.0, 5.0], "treated": [True, True],
    })
    expected = (np.log(10.0 / 5.0) + np.log(5.0 / 5.0)) / 2
    assert treated_log_wage_gap(panel) == pytest.approx(expected)


# --- the row contract -----------------------------------------------------

@pytest.fixture(scope="module")
def table(panel):
    # Synthetic states are S01..S20, which no border pair names, so the
    # border row is skipped rather than run on nothing.
    return build_comparison(panel, n_boot=20, include_synthetic_control=False)


def test_table_has_the_documented_columns(table):
    assert list(table.columns) == EXPECTED_COLUMNS


def test_every_scale_is_one_of_the_declared_kinds(table):
    assert set(table["scale"]) <= VALID_SCALES


def test_converted_rows_are_exactly_the_semi_elasticity_ones(table):
    """`scale` is the reader's only signal that a row was rescaled."""
    converted = set(table.loc[table["scale"] == "converted", "method"])
    assert converted == {"TWFE DiD\n(log minimum wage)"}


def test_the_conversion_factor_is_attached_to_the_table(table, panel):
    assert table.attrs["log_wage_gap"] == pytest.approx(treated_log_wage_gap(panel))


def test_converted_row_is_the_raw_coefficient_times_the_factor(table, panel):
    from src.methods.twfe_did import estimate_twfe

    raw = estimate_twfe(panel).params["log_minimum_wage"]
    row = table[table["method"] == "TWFE DiD\n(log minimum wage)"].iloc[0]
    assert row["estimate"] == pytest.approx(raw * table.attrs["log_wage_gap"])


def test_binary_row_is_not_rescaled(table, panel):
    """It is already in percentage points; converting it would double-count."""
    from src.methods.twfe_did import BINARY_TREATMENT, estimate_twfe_binary

    raw = estimate_twfe_binary(panel).params[BINARY_TREATMENT]
    row = table[table["method"] == "TWFE DiD\n(binary treatment)"].iloc[0]
    assert row["scale"] == "native"
    assert row["estimate"] == pytest.approx(raw)


def test_intervals_bracket_their_estimates(table):
    fitted = table[table["scale"] != "failed"]
    assert (fitted["ci_lower"] <= fitted["estimate"]).all()
    assert (fitted["estimate"] <= fitted["ci_upper"]).all()


def test_synthetic_control_is_opt_out(panel, table):
    assert not table["method"].str.startswith("Synthetic control").any()


def test_border_row_is_skipped_rather_than_run_on_nothing(table):
    assert not table["method"].str.startswith("Border").any()


# --- the failure contract -------------------------------------------------
#
# Every estimator is wrapped so one broken method does not take the table
# with it. That is only defensible if the failure is legible in the output.

def test_a_failing_estimator_becomes_a_row_not_an_exception(panel, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("singular matrix, allegedly")

    monkeypatch.setattr(cmp_mod, "estimate_att_gt", explode)
    table = build_comparison(panel, n_boot=20, include_synthetic_control=False)
    row = table[table["method"] == "Callaway-Sant'Anna"].iloc[0]
    assert row["scale"] == "failed"
    assert "singular matrix" in row["note"]
    assert np.isnan(row["estimate"])
    assert np.isnan(row["ci_lower"]) and np.isnan(row["ci_upper"])


def test_one_failure_does_not_suppress_the_other_rows(panel, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(cmp_mod, "estimate_att_gt", explode)
    table = build_comparison(panel, n_boot=20, include_synthetic_control=False)
    assert (table["scale"] != "failed").sum() >= 2


def test_the_factor_failing_is_not_swallowed(panel):
    """A bad conversion factor corrupts two rows, so it must raise."""
    broken = panel.drop(columns="treated")
    with pytest.raises(ValueError, match="needs a `treated` column"):
        build_comparison(broken, n_boot=20, include_synthetic_control=False)
