# Final Project 1 figures, from aggregate results only.
# Run from this directory (r-viz/), or use here::here() if run elsewhere.

library(tidyverse)

aggregate_dir <- "../results/aggregate"

# Placeholder -- filled in once results/aggregate/*.csv actually exist.
# Expected schema (see PROJECT_GUIDE.md Part 5, issue #18):
#   task, condition, graph_variant, metric, value, ci_low, ci_high, seed
