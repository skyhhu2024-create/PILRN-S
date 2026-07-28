# Released results

This directory contains derived evaluation products needed to verify the
reported tables, statistical comparisons, and figures without rerunning the
licensed discharge data.

The canonical station-level file is `station_seed_metrics.csv` (eight methods,
five seeds, and 164 scored stations). `method_summary.csv`,
`seed_summary.csv`, the `section_5_*.csv` files, and the paired-evidence tables
are derived from that file. `leakage_audit_batches.csv` records the five
all-month audit batches reported in the Supplementary Information. The
`figure_source_data/` directory contains compact, derived panel data.

Current manuscript numbering differs from two compatibility filenames:
`fig09_round_seed_metrics.csv` and `fig09_component_summary.csv` support
Figure 10, while the `fig10_density_*.csv` files support Figure 11. The root
README provides the complete manuscript-to-release mapping.

`figure_source_data/FigS2_source.csv` uses the same log-symmetric >30%
catchment-area disagreement rule as the manuscript and embedded figure:
`ratio > 1.3` or `ratio < 1/1.3`. Among the 189 stations with usable
observed-to-HydroRIVERS mean-flow ratios, this yields 113 verified-consistent
linkages and 76 mismatches.

No GRDC discharge time series are redistributed. In particular, the observed
hydrographs underlying Figure 6 must be obtained directly from GRDC under its
terms of use. Raw ERA5-Land files, model checkpoints, training caches, and
complete prediction archives are also excluded.
