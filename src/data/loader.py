"""Single entry point the app/notebooks use to get a panel: real processed
data if it exists, otherwise the synthetic panel (with a visible warning).

The `(panel, is_synthetic)` pair is the contract: every caller has to be
able to say which of the two it got, because the synthetic panel has a
ground-truth effect baked in and the real one does not. Nothing here
falls back silently.
"""
from pathlib import Path

from src.data.synthetic import generate_state_year_panel

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PANEL_FILENAME = "panel_state_year.parquet"


def load_state_year_panel(processed_dir=None):
    """Returns (panel, is_synthetic).

    `processed_dir` overrides where the real panel is looked for; it exists
    so this branch can be tested without a real panel on disk, and so a
    caller can point at an alternative build.
    """
    real_path = Path(processed_dir or PROCESSED_DIR) / PANEL_FILENAME
    if real_path.exists():
        import pandas as pd
        return pd.read_parquet(real_path), False
    return generate_state_year_panel(), True
