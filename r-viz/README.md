# r-viz/

R + ggplot2, for the final Project 1 report figures only. Reads exclusively from
`../results/aggregate/*.csv` -- never touches raw data, never touches per-patient
files. All the actual graph construction and network-stat computation stays in
Python (`../src/ehrgraph/`) -- this is a downstream consumer of already-aggregate
output, same governance boundary as everywhere else in this repo.

## Setup (one time)

```r
install.packages("renv")
renv::init()
renv::install(c("tidyverse", "here"))
renv::snapshot()
```

`renv` is R's equivalent of `uv` -- pins exact package versions in `renv.lock`,
committed to git, so this is reproducible on any machine the same way the Python
side is.

## Usage

Open `figures.R` (or the RStudio/VS Code R project) and run top to bottom once
`results/aggregate/` has real data in it (starts after issue #18 freezes the
schema, filled in progressively as later issues land).
