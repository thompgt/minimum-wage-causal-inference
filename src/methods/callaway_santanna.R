#!/usr/bin/env Rscript
# Callaway & Sant'Anna (2021) staggered-adoption DiD estimator.
#
# Usage: Rscript callaway_santanna.R <input_csv> <output_csv>
#
# Input CSV must have columns: state, year, unemployment_rate, adoption_year
# (adoption_year = NA/blank for never-treated units, per the `did` package's
# convention of coding never-treated as first.treat = 0).
#
# Output CSV has columns: group, time, att, se, ci_lower, ci_upper
# (group-time average treatment effects on the treated, from att_gt()),
# plus a final row with event = "overall" for the aggregated ATT.

suppressPackageStartupMessages({
  library(did)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript callaway_santanna.R <input_csv> <output_csv>")
}
input_path <- args[1]
output_path <- args[2]

df <- fread(input_path)
df[, state_id := .GRP, by = state]  # did package wants a numeric unit id
df[, first_treat := ifelse(is.na(adoption_year), 0, adoption_year)]

cs_result <- att_gt(
  yname = "unemployment_rate",
  tname = "year",
  idname = "state_id",
  gname = "first_treat",
  data = df,
  control_group = "nevertreated",
  clustervars = "state_id"
)

gt_out <- data.table(
  group = cs_result$group,
  time = cs_result$t,
  att = cs_result$att,
  se = cs_result$se
)
gt_out[, ci_lower := att - 1.96 * se]
gt_out[, ci_upper := att + 1.96 * se]

agg_simple <- aggte(cs_result, type = "simple")
overall_row <- data.table(
  group = NA_real_, time = NA_real_,
  att = agg_simple$overall.att, se = agg_simple$overall.se,
  ci_lower = agg_simple$overall.att - 1.96 * agg_simple$overall.se,
  ci_upper = agg_simple$overall.att + 1.96 * agg_simple$overall.se
)

out <- rbind(gt_out, overall_row)
fwrite(out, output_path)
cat(sprintf("Wrote %d rows to %s\n", nrow(out), output_path))
