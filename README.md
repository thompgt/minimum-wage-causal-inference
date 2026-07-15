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

1. Build the panel:
   ```
   python -m src.data.build_panel
   ```
2. Explore the pipeline and each method in `notebooks/` (run in order,
   01 through 05).
3. Run the interactive app:
   ```
   streamlit run app/Home.py
   ```

## Project layout

See `src/data` (data acquisition + panel construction), `src/methods`
(causal estimators, Python and R), `src/diagnostics` (pre-trend/robustness
checks), `notebooks/` (methodology walkthroughs), `app/` (Streamlit UI).
