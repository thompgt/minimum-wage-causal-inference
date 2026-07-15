# Minimum Wage vs. Unemployment: Causal Inference

Estimates the causal effect of U.S. state minimum wage increases on
unemployment, using multiple causal inference methods so the results'
sensitivity to method choice is visible rather than hidden behind a single
number.

## Methods implemented

- Two-way fixed effects DiD (`src/methods/twfe_did.py`)
- Event-study / pre-trend check (`src/methods/event_study.py`)
- Callaway-Sant'Anna staggered-adoption DiD (`src/methods/callaway_santanna.R`)
- Synthetic control case studies (`src/methods/synthetic_control.R`)
- Border-discontinuity (contiguous county pairs) (`src/methods/border_discontinuity.py`)

## Setup

### Python

```
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### R

Requires R installed and on PATH. Then:

```
Rscript install.R
```

### API key

BLS API pulls need a free key from https://www.bls.gov/developers/. Copy
`.env.example` to `.env` and set `BLS_API_KEY`.

## Usage

**Without real data**: everything below runs out of the box against a
seeded synthetic panel (`src/data/synthetic.py`) with a known ground-truth
effect, so you can explore the pipeline before wiring up BLS/minimum-wage
data.

**With real data**:
1. Set `BLS_API_KEY` (see above) and add a state minimum wage history CSV
   to `data/raw/state_minimum_wage.csv` (see `src/data/fetch_minwage.py`
   docstring for the required columns and where to find the data).
2. Build the panel:
   ```
   python -m src.data.fetch_bls
   python -m src.data.build_panel
   ```
   `src/data/loader.py` (used by the app and notebooks) automatically
   prefers `data/processed/panel_state_year.parquet` once it exists.

Then:
- Explore the pipeline and each method in `notebooks/` (run in order,
  01 through 05). To execute from the CLI:
  ```
  jupyter kernelspec list   # confirm/register a kernel for this venv, e.g.:
  python -m ipykernel install --user --name mwci --display-name "Python (mwci)"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mwci notebooks/01_data_pipeline_demo.ipynb
  ```
- Run the interactive app:
  ```
  streamlit run app/Home.py
  ```
- Run tests:
  ```
  pytest
  ```

## Project layout

See `src/data` (data acquisition + panel construction + synthetic
fallback), `src/methods` (causal estimators, Python and R), `src/diagnostics`
(pre-trend/robustness checks), `notebooks/` (methodology walkthroughs),
`app/` (Streamlit UI).

## Known gaps

- **R-dependent estimators are unverified on this machine** (Callaway-
  Sant'Anna and synthetic control) — R isn't installed here. The Python
  side (`r_bridge.py`, input validation, error handling) is tested; the
  `.R` scripts themselves need a run with R + the `did`/`Synth` packages
  installed to confirm end-to-end.
- **Real BLS/minimum-wage data hasn't been fetched** — no API key was
  available while building this. All notebooks/app/tests currently run
  against the synthetic panel.
- The event-study and Callaway-Sant'Anna estimators need an `adoption_year`
  column (staggered treatment timing), which the synthetic panel has but
  the real `build_panel.py` pipeline does not yet compute from the
  minimum wage panel — add that derivation when wiring in real data.
