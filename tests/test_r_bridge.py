import shutil

import pandas as pd
import pytest

from src.methods.r_bridge import run_callaway_santanna, run_r_script


def test_run_callaway_santanna_validates_required_columns():
    bad_panel = pd.DataFrame({"state": ["S01"], "year": [2020]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_callaway_santanna(bad_panel)


@pytest.mark.skipif(shutil.which("Rscript") is not None, reason="Rscript is available")
def test_run_r_script_raises_clear_error_without_rscript():
    with pytest.raises(RuntimeError, match="Rscript not found"):
        run_r_script("callaway_santanna.R", pd.DataFrame({"a": [1]}))
