pkgs <- c("did", "Synth", "tidysynth", "fixest", "data.table", "arrow")
installed <- rownames(installed.packages())
for (p in pkgs) {
  if (!(p %in% installed)) install.packages(p, repos = "https://cloud.r-project.org")
}
