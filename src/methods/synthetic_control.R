#!/usr/bin/env Rscript
# Synthetic control case study: build a synthetic counterfactual for one
# treated state from a weighted combination of untreated donor states.
#
# Usage: Rscript synthetic_control.R <input_csv> <output_csv> <treated_state> <treatment_year>
#
# Input CSV must have columns: state, year, unemployment_rate, minimum_wage
# (a state-year panel covering the full pre/post window for the case study).
#
# Output CSV has columns: year, state, unemployment_rate, type
# (type is "actual" or "synthetic"), covering the treated state's actual
# path and the synthetic counterfactual's path across all years.

suppressPackageStartupMessages({
  library(Synth)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript synthetic_control.R <input_csv> <output_csv> <treated_state> <treatment_year>")
}
input_path <- args[1]
output_path <- args[2]
treated_state <- args[3]
treatment_year <- as.integer(args[4])

df <- fread(input_path)
df[, state_id := .GRP, by = state]
id_lookup <- unique(df[, .(state, state_id)])
treated_id <- id_lookup[state == treated_state, state_id]
donor_ids <- id_lookup[state != treated_state, state_id]

dp <- dataprep(
  foo = as.data.frame(df),
  predictors = c("minimum_wage"),
  predictors.op = "mean",
  time.predictors.prior = sort(unique(df[year < treatment_year, year])),
  dependent = "unemployment_rate",
  unit.variable = "state_id",
  time.variable = "year",
  treatment.identifier = treated_id,
  controls.identifier = donor_ids,
  time.optimize.ssr = sort(unique(df[year < treatment_year, year])),
  unit.names.variable = "state",
  time.plot = sort(unique(df$year))
)

sc <- synth(dp)

years <- sort(unique(df$year))
actual <- data.table(
  year = years, state = treated_state,
  unemployment_rate = as.vector(dp$Y1plot), type = "actual"
)
synthetic <- data.table(
  year = years, state = paste0("synthetic_", treated_state),
  unemployment_rate = as.vector(dp$Y0plot %*% sc$solution.w), type = "synthetic"
)

out <- rbind(actual, synthetic)
fwrite(out, output_path)
cat(sprintf("Wrote %d rows to %s\n", nrow(out), output_path))
