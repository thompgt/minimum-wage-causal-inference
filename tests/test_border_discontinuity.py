import pandas as pd

from src.methods.border_discontinuity import border_pair_diffs, estimate_border_effect


def make_panel():
    return pd.DataFrame({
        "state": ["A", "A", "B", "B", "C", "C"],
        "year": [2020, 2021, 2020, 2021, 2020, 2021],
        "unemployment_rate": [5.0, 5.5, 4.0, 4.2, 6.0, 6.3],
        "minimum_wage": [8.0, 9.0, 7.25, 7.25, 10.0, 11.0],
    })


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


def test_estimate_border_effect_returns_fitted_model():
    panel = make_panel()
    diffs = border_pair_diffs(panel, [("A", "B"), ("B", "C")])
    model = estimate_border_effect(diffs)
    assert "treatment_diff" in model.params.index
