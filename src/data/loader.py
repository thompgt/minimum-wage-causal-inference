"""Single entry point the app/notebooks use to get a panel: real processed
data if it exists, otherwise the synthetic panel (with a visible warning)."""
from pathlib import Path

from src.data.synthetic import generate_state_year_panel

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_state_year_panel():
    """Returns (panel, is_synthetic)."""
    real_path = PROCESSED_DIR / "panel_state_year.parquet"
    if real_path.exists():
        import pandas as pd
        return pd.read_parquet(real_path), False
    return generate_state_year_panel(), True
