"""Tests for the placebo test the README's strongest caveat rests on.

The claim "38% of randomly-timed placebo draws are at least this large" is
the one number in the README that argues against everything else in it, and
nothing tested how it was produced. Two properties matter: the reassignment
has to actually break the link between a state and its own treatment path,
and the share has to be reported over a denominator that counts failed
draws rather than quietly dropping them.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_state_year_panel
from src.diagnostics.parallel_trends import (
    placebo_share_at_least_as_large,
    placebo_test,
)
from src.diagnostics.robustness import RefitFailureError
from src.methods.twfe_did import estimate_twfe


@pytest.fixture(scope="module")
def panel():
    return generate_state_year_panel(seed=5)


def test_draw_count_and_determinism(panel):
    first = placebo_test(panel, estimate_twfe, n_placebos=6, seed=3)
    second = placebo_test(panel, estimate_twfe, n_placebos=6, seed=3)
    assert len(first) == 6
    pd.testing.assert_series_equal(first, second)


def test_a_different_seed_gives_a_different_draw(panel):
    a = placebo_test(panel, estimate_twfe, n_placebos=6, seed=1)
    b = placebo_test(panel, estimate_twfe, n_placebos=6, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_reassignment_uses_whole_donor_treatment_paths(panel):
    """The point of the design: a state gets *another state's* history.

    If draws just reshuffled treatment within a state, the placebo
    distribution would inherit the real design's variation and the test
    would be vacuous. Reassigning whole paths keeps each path internally
    coherent while breaking its link to the state's own labour market.
    """
    real = estimate_twfe(panel).params["log_minimum_wage"]
    placebos = placebo_test(panel, estimate_twfe, n_placebos=12, seed=0)
    # Some permutation is the identity on at most a vanishing share of
    # draws; the rest must not reproduce the real coefficient.
    assert (np.abs(placebos.to_numpy() - real) > 1e-9).sum() >= len(placebos) - 1


def test_failure_accounting_is_attached(panel):
    placebos = placebo_test(panel, estimate_twfe, n_placebos=5, seed=0)
    assert placebos.attrs["n_attempted"] == 5
    assert placebos.attrs["n_failed"] == len(placebos.attrs["failures"])
    assert placebos.attrs["failure_rate"] == pytest.approx(
        placebos.attrs["n_failed"] / 5
    )


def test_too_many_failures_raise_rather_than_shrink_the_sample(panel):
    """A surviving 3-of-50 placebo distribution is not a null distribution."""
    def always_fails(_):
        raise RuntimeError("refit blew up")

    with pytest.raises(RefitFailureError, match="refits failed"):
        placebo_test(panel, always_fails, n_placebos=5, seed=0)


def test_a_tolerated_failure_rate_is_recorded_not_hidden(panel):
    calls = {"n": 0}

    def fails_once(df):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("one bad draw")
        return estimate_twfe(df)

    placebos = placebo_test(panel, fails_once, n_placebos=10, seed=0,
                            max_failure_rate=0.5)
    assert len(placebos) == 9
    assert placebos.attrs["n_attempted"] == 10
    assert placebos.attrs["n_failed"] == 1
    assert "one bad draw" in placebos.attrs["failures"][0]


# --- the share, and its denominator ---------------------------------------

def make_placebos(values, n_attempted=None, n_failed=0):
    s = pd.Series(values, dtype=float, name="placebo_coef")
    s.attrs.update(n_attempted=n_attempted or len(values), n_failed=n_failed)
    return s


def test_share_compares_magnitudes_not_signs():
    """A large negative placebo is as extreme as a large positive one."""
    placebos = make_placebos([-5.0, 0.1, 0.1, 0.1])
    assert placebo_share_at_least_as_large(placebos, 1.0)["share"] == pytest.approx(0.25)


def test_share_counts_ties_as_at_least_as_large():
    placebos = make_placebos([1.0, 0.0])
    assert placebo_share_at_least_as_large(placebos, 1.0)["share"] == pytest.approx(0.5)


def test_share_reports_the_denominator_it_was_computed_over():
    """Without n_attempted a bare percentage is not interpretable."""
    placebos = make_placebos([0.5, 0.5, 2.0], n_attempted=10, n_failed=7)
    out = placebo_share_at_least_as_large(placebos, 1.0)
    assert out["n_used"] == 3
    assert out["n_attempted"] == 10
    assert out["n_failed"] == 7
    assert out["share"] == pytest.approx(1 / 3)


def test_no_draws_is_an_error_not_a_zero_share():
    with pytest.raises(ValueError, match="no placebo draws"):
        placebo_share_at_least_as_large(make_placebos([]), 1.0)
