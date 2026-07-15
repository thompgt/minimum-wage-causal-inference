import shutil

import pandas as pd
import pytest

from src.methods.r_bridge import run_callaway_santanna, run_r_script, run_synthetic_control


def test_run_callaway_santanna_validates_required_columns():
    bad_panel = pd.DataFrame({"state": ["S01"], "year": [2020]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_callaway_santanna(bad_panel)


def test_run_synthetic_control_validates_required_columns():
    bad_panel = pd.DataFrame({"state": ["S01"], "year": [2020]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_synthetic_control(bad_panel, "S01", 2015)


def test_run_synthetic_control_validates_treated_state_present():
    panel = pd.DataFrame({
        "state": ["S01", "S02"],
        "year": [2020, 2020],
        "unemployment_rate": [5.0, 6.0],
        "minimum_wage": [7.25, 8.0],
    })
    with pytest.raises(ValueError, match="not found in panel"):
        run_synthetic_control(panel, "S99", 2015)


@pytest.mark.skipif(shutil.which("Rscript") is not None, reason="Rscript is available")
def test_run_r_script_raises_clear_error_without_rscript():
    with pytest.raises(RuntimeError, match="Rscript not found"):
        run_r_script("callaway_santanna.R", pd.DataFrame({"a": [1]}))
