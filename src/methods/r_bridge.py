"""Call the R-based estimators (callaway_santanna.R, synthetic_control.R)
from Python via subprocess, round-tripping data through temporary CSVs.

Requires Rscript on PATH and the packages in install.R installed
(`Rscript install.R` from the project root).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

METHODS_DIR = Path(__file__).resolve().parent


def _check_rscript_available():
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R (https://cran.r-project.org/) "
            "and run `Rscript install.R` from the project root."
        )


def run_r_script(script_name, input_df, extra_args=None, timeout=300):
    """Write input_df to a temp CSV, run an R script that reads/writes CSVs,
    and return the output as a DataFrame.

    `script_name` is relative to src/methods/, e.g. "callaway_santanna.R".
    """
    _check_rscript_available()
    script_path = METHODS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        output_path = Path(tmpdir) / "output.csv"
        input_df.to_csv(input_path, index=False)

        cmd = ["Rscript", str(script_path), str(input_path), str(output_path)]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{script_name} failed (exit {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError(
                f"{script_name} exited 0 but did not write {output_path.name}"
            )
        return pd.read_csv(output_path)


def run_callaway_santanna(panel):
    """panel: state-year DataFrame with state, year, unemployment_rate, adoption_year."""
    required = {"state", "year", "unemployment_rate", "adoption_year"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    return run_r_script("callaway_santanna.R", panel[list(required)])


def run_synthetic_control(panel, treated_state, treatment_year):
    """panel: state-year DataFrame with state, year, unemployment_rate, minimum_wage.

    Runs a synthetic control case study for `treated_state`, using its
    `treatment_year` minimum wage increase as the intervention date.
    """
    required = {"state", "year", "unemployment_rate", "minimum_wage"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    if treated_state not in panel["state"].unique():
        raise ValueError(f"treated_state {treated_state!r} not found in panel")
    return run_r_script(
        "synthetic_control.R",
        panel[list(required)],
        extra_args=[treated_state, str(treatment_year)],
    )


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel

    panel = generate_state_year_panel()
    out = run_callaway_santanna(panel)
    print(out)
