"""Tests for the single entry point every surface uses to get a panel.

The whole repo runs before any data is fetched because this module falls
back to a synthetic panel with a ground-truth effect baked in. That is
only safe if callers can tell which one they got, so the `is_synthetic`
flag is the thing worth pinning.
"""
import pandas as pd

from src.data.loader import PANEL_FILENAME, load_state_year_panel


def test_falls_back_to_synthetic_when_no_processed_panel(tmp_path):
    panel, is_synthetic = load_state_year_panel(processed_dir=tmp_path)
    assert is_synthetic
    assert not panel.empty


def test_reads_the_real_panel_when_it_exists(tmp_path):
    real = pd.DataFrame({
        "state": ["AA", "AA"], "year": [2020, 2021],
        "unemployment_rate": [5.0, 5.5], "minimum_wage": [9.0, 9.5],
    })
    real.to_parquet(tmp_path / PANEL_FILENAME, index=False)

    panel, is_synthetic = load_state_year_panel(processed_dir=tmp_path)
    assert not is_synthetic
    pd.testing.assert_frame_equal(panel, real)


def test_the_synthetic_flag_is_not_cosmetic(tmp_path):
    """The two branches must return different data, or the flag means nothing."""
    from src.data.synthetic import STATES

    synthetic, was_synthetic = load_state_year_panel(processed_dir=tmp_path)
    assert was_synthetic
    assert set(synthetic["state"]) == set(STATES)

    pd.DataFrame({
        "state": ["AA"], "year": [2020],
        "unemployment_rate": [5.0], "minimum_wage": [9.0],
    }).to_parquet(tmp_path / PANEL_FILENAME, index=False)
    real, is_synthetic = load_state_year_panel(processed_dir=tmp_path)
    assert not is_synthetic
    assert set(real["state"]) == {"AA"}


def test_the_default_location_is_still_used_when_nothing_is_passed():
    """No argument must not mean "always synthetic"."""
    panel, is_synthetic = load_state_year_panel()
    assert isinstance(panel, pd.DataFrame)
    assert isinstance(is_synthetic, bool)
