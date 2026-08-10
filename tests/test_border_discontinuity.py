import numpy as np
import pandas as pd
import pytest

from src.methods.border_discontinuity import (
    _STATE_ADJACENCY,
    US_STATE_BORDER_PAIRS,
    _enumerate_border_pairs,
    border_pair_diffs,
    estimate_border_effect,
)


def make_panel():
    return pd.DataFrame({
        "state": ["A", "A", "B", "B", "C", "C"],
        "year": [2020, 2021, 2020, 2021, 2020, 2021],
        "unemployment_rate": [5.0, 5.5, 4.0, 4.2, 6.0, 6.3],
        "minimum_wage": [8.0, 9.0, 7.25, 7.25, 10.0, 11.0],
    })


def make_wide_panel(n_states=8, n_years=12, seed=3):
    """Enough states and years that a pair+period FE design is identified."""
    rng = np.random.default_rng(seed)
    states = [f"S{i:02d}" for i in range(n_states)]
    years = list(range(2010, 2010 + n_years))
    rows = []
    for i, state in enumerate(states):
        for year in years:
            rows.append({
                "state": state,
                "year": year,
                "unemployment_rate": 5.0 + 0.1 * i + rng.normal(0, 0.2),
                "minimum_wage": 7.25 + 0.15 * i + 0.1 * (year - 2010),
            })
    return pd.DataFrame(rows)


def wide_pairs(n_states=5):
    """Every pair among the first `n_states`, so each state recurs in several.

    A chain (S00-S01, S01-S02, ...) would not exercise two-way clustering:
    with one pair per state on each side, clustering on state_a collapses
    to clustering on the pair. Real borders are not a chain — a state has
    as many neighbours as it has neighbours — so the fixture is a clique.
    """
    states = [f"S{i:02d}" for i in range(n_states)]
    return [
        (states[i], states[j])
        for i in range(n_states)
        for j in range(i + 1, n_states)
    ]


def test_border_pair_diffs_computes_correct_signs():
    panel = make_panel()
    diffs = border_pair_diffs(panel, [("A", "B")])
    row_2020 = diffs[diffs["year"] == 2020].iloc[0]
    assert row_2020["outcome_diff"] == 5.0 - 4.0
    assert row_2020["treatment_diff"] == 8.0 - 7.25


def test_border_pair_diffs_skips_missing_states():
    panel = make_panel()
    diffs = border_pair_diffs(panel, [("A", "ZZZ")])
    assert diffs.empty


def test_border_pair_diffs_only_uses_overlapping_periods():
    panel = make_panel()
    panel_trimmed = panel[~((panel["state"] == "C") & (panel["year"] == 2021))]
    diffs = border_pair_diffs(panel_trimmed, [("A", "C")])
    assert set(diffs["year"]) == {2020}


def test_pairs_are_canonically_ordered_whichever_way_they_are_given():
    """A state must sit on the same side of a pair however the pair is written."""
    panel = make_panel()
    forward = border_pair_diffs(panel, [("A", "B")])
    reversed_ = border_pair_diffs(panel, [("B", "A")])
    assert (forward["state_a"] == "A").all()
    assert (reversed_["state_a"] == "A").all()
    pd.testing.assert_frame_equal(forward, reversed_)


# --- the enumeration replacing the hand-picked subset ---------------------

def test_adjacency_map_is_symmetric():
    """_enumerate_border_pairs raises on an asymmetric entry; assert it holds."""
    assert _enumerate_border_pairs(_STATE_ADJACENCY) == US_STATE_BORDER_PAIRS


def test_asymmetric_adjacency_is_rejected():
    with pytest.raises(ValueError, match="not symmetric"):
        _enumerate_border_pairs({"AA": ["BB"], "BB": []})


def test_every_pair_is_sorted_and_unique():
    assert all(a < b for a, b in US_STATE_BORDER_PAIRS)
    assert len(set(US_STATE_BORDER_PAIRS)) == len(US_STATE_BORDER_PAIRS)


def test_enumeration_is_not_selected_on_high_minimum_wage_states():
    """The old 28-pair subset was concentrated in CA/OR/NV and NY/NJ/PA."""
    assert len(US_STATE_BORDER_PAIRS) > 100
    states = {s for pair in US_STATE_BORDER_PAIRS for s in pair}
    # Every contiguous jurisdiction borders something, so all 49 appear.
    assert len(states) == 49
    for landlocked_by_one_neighbour in ["ME", "SC", "RI"]:
        assert any(landlocked_by_one_neighbour in p for p in US_STATE_BORDER_PAIRS)


def test_known_borders_present_and_known_non_borders_absent():
    for pair in [("CA", "OR"), ("DC", "MD"), ("ME", "NH"), ("AZ", "CO")]:
        assert tuple(sorted(pair)) in US_STATE_BORDER_PAIRS
    for pair in [("CA", "NY"), ("ME", "FL"), ("MI", "MN"), ("AK", "WA")]:
        assert tuple(sorted(pair)) not in US_STATE_BORDER_PAIRS


# --- the specification ----------------------------------------------------

def test_estimate_border_effect_returns_fitted_model():
    diffs = border_pair_diffs(make_wide_panel(), wide_pairs())
    model = estimate_border_effect(diffs)
    assert "treatment_diff" in model.params.index
    assert model.border_spec["pair_fe"] and model.border_spec["period_fe"]


def test_fixed_effects_are_actually_in_the_design():
    diffs = border_pair_diffs(make_wide_panel(), wide_pairs())
    model = estimate_border_effect(diffs)
    assert any(name.startswith("pair_") for name in model.params.index)
    assert any(name.startswith("period_") for name in model.params.index)


def test_pooled_and_fixed_effects_specifications_differ():
    """If they agreed, the fixed effects would not be doing anything."""
    diffs = border_pair_diffs(make_wide_panel(), wide_pairs())
    pooled = estimate_border_effect(diffs, pair_fe=False, period_fe=False,
                                    cluster="pair")
    within = estimate_border_effect(diffs)
    assert not np.isclose(
        pooled.params["treatment_diff"], within.params["treatment_diff"]
    )


def test_two_way_clustering_changes_the_standard_error():
    diffs = border_pair_diffs(make_wide_panel(), wide_pairs())
    two_way = estimate_border_effect(diffs, cluster="two-way")
    by_pair = estimate_border_effect(diffs, cluster="pair")
    assert not np.isclose(
        two_way.bse["treatment_diff"], by_pair.bse["treatment_diff"]
    )


def test_unknown_cluster_option_is_rejected():
    diffs = border_pair_diffs(make_wide_panel(), wide_pairs())
    with pytest.raises(ValueError, match="unknown cluster option"):
        estimate_border_effect(diffs, cluster="state")


def test_saturated_design_raises_a_readable_error():
    """Two pairs over two years cannot support pair and period fixed effects."""
    diffs = border_pair_diffs(make_panel(), [("A", "B"), ("B", "C")])
    with pytest.raises(ValueError, match="saturated"):
        estimate_border_effect(diffs)


def test_empty_pair_diffs_raises():
    with pytest.raises(ValueError, match="no border-pair observations"):
        estimate_border_effect(pd.DataFrame(columns=["treatment_diff"]))
