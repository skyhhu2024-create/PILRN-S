# PILRN-S — Compact signal-level river-network routing for monthly streamflow estimation in partially gauged networks

Code, audited configuration data, and results accompanying:

> Chen, M., Li, H., Yong, B., Liao, W., Wang, H. — *Compact signal-level
> river-network routing suffices to fuse frozen regional priors with concurrent
> observations for monthly streamflow estimation in partially gauged networks*


PILRN-S (Positive-weight Interpretable Learnable River-Network router operating
at the Signal level) corrects a frozen Regional-LSTM runoff-depth prior by
propagating availability-masked runoff-depth signals through
descriptor-conditioned positive edge weights and a learned prior–neighbour
gate. Three propagation rounds are fixed by an outcome-independent graph rule
(≥95% nearest-visible-donor reachability). Under a strict leave-station-out
protocol (five seeds × ten fixed folds, 164 quality-controlled Rhine Basin
gauges, observed discharge as the only reference), optimizing only the
1,074-parameter Stage-2 fusion correction attains median NSE 0.656 / KGE 0.697
— statistically indistinguishable from full 87,763-parameter joint
optimization and comparable to a 53,761-parameter embedding-level GAT —
while producing fewer negative-NSE gauges and lower seed sensitivity.

The two information sources have distinct roles. The frozen Regional LSTM
provides coverage and stability; concurrent gauge observations provide local,
event-specific evidence; and the learned gate determines how strongly that
evidence should revise the prior for each target and month.

## Repository layout

```
code/       complete experiment pipeline (Rebuild_v2)
            - config_v2.py                paths, features, split definitions
            - prepare_forcing.py          ERA5-Land -> monthly forcing matrix
            - rebuild_network_v3.py       metadata-constrained station-network rebuild
            - qc_stations.py              hydrological screening diagnostics
            - graph_v2.py                 edge descriptors and hand-specified weights
            - train_regional_lstm.py      Stage-1 Regional LSTM
            - train_pilrn_v2.py           Stage-2 router (PILRN-S and variants), GAT
            - train_coupled_v2.py         coupled from-scratch / warm-start controls
            - baseline_idw_v2.py          nearest+DAR and IDW+DAR
            - baseline_rf_v2.py           random forest
            - run_all_v3.py               final sequential reproduction controller
            - run_ablation_*.py / run_component_sweep_*.py
                                              supporting SI evidence
            - run_final_*.py              five-seed x ten-fold production runs
            - run_round_sweep_*.py        1-8 round propagation-depth sweep
            - run_density_*.py            50% / 25% training-gauge-density experiments
            - audit_leakage_physical_v4.py  all-month model-execution leakage audit
            - evaluate_final_manuscript_v5.py  unified post-processing (Table 3)
data/       redistributable derived configuration
            - river_network_nodes_v3.csv / river_network_edges_v3.csv
              (307 candidate-node records, including 8 flagged unmappable;
              299 active mapped nodes, 298 physical links, and 596 directed
              message-passing edges after bidirectional duplication)
            - kfold_station_assignment.csv     ten fixed leave-station-out folds
            - grdc_official_metadata.csv       official areas/coordinates used for QC
            - area_correction_report.csv       matched vs. official areas (82/207 >30%)
            - station_qc.csv, station_exclude.csv
results/    verification of every number in the paper without a GRDC licence
            - station_seed_metrics.csv         8 methods x 5 seeds x 164 stations
            - method_summary.csv, seed_summary.csv, section_5_*.csv (Table 3)
            - paired_significance_paperconv_v6.csv  (SI Table S13: CIs, Wilcoxon)
            - stability_evidence.csv, paired_*_evidence.csv (Tables S9-S10)
            - leakage_audit_batches.csv   released all-month audit batch summary
            - figure_source_data/              per-panel source CSVs
            - README.md                   scope and redistribution notes
```

## Data access (not redistributed here)

| Data | Source | Notes |
|---|---|---|
| Monthly discharge | [GRDC](https://grdc.bafg.de) | station list in `data/kfold_station_assignment.csv`; redistribution restricted |
| Meteorological forcing (11 variables) | [ERA5-Land monthly](https://cds.climate.copernicus.eu) | 45.5–52.5°N, 5.0–12.5°E, 1990–2025 |
| River network | [HydroRIVERS v1.0](https://www.hydrosheds.org) | derived graph shipped in `data/` |

## Reproducing the experiments

The full manuscript-final workflow is available as one sequential,
resume-safe command:

```bash
python code/run_all_v3.py
```

Despite its compatibility filename, this controller dispatches the final v5
configuration: three rounds, no multiplicative area modulation, five seeds,
ten folds, supporting SI ablations, both coupled controls, the complete
density experiment, and the all-month leakage audits. Individual stages can
be selected, for example:

```bash
python code/run_all_v3.py --only supporting,rounds,pilrn,components
python code/run_all_v3.py --only audit --audit-device cpu
```

Manual execution follows the same order:

1. Download GRDC monthly series for the listed stations and the ERA5-Land
   subset. In a standalone clone, set `GRDC_DIR` to the directory containing
   the station CSV files and `NC_PATH` to the ERA5-Land NetCDF file. The
   released graph, fold assignment, and QC tables in `data/` are detected
   automatically, and generated files are written to `outputs/`. The original
   workspace layout remains available through `JOH_ROOT`; `JOH_OUTPUT_ROOT`
   can override the output location.
2. `python code/prepare_forcing.py --extract` — build the monthly forcing
   matrix.
3. Use the released `data/river_network_nodes_v3.csv` and
   `data/river_network_edges_v3.csv`. Run `python code/qc_stations.py` after
   configuring the GRDC directory to reproduce the record-availability
   screen. Rebuilding the graph itself is optional and additionally requires
   the source HydroRIVERS layer and legacy station mapping.
4. Stage 1 and matched controls:
   `python code/run_stability_10x5.py` (five seeds × ten folds).
5. Supporting SI experiments:
   `python code/run_ablation_physical_v4.py`,
   `python code/run_component_sweep_physical_v4.py`, and
   `python code/run_density_physical_v4.py`.
6. Depth and final PILRN-S:
   `python code/run_round_sweep_physical_v4.py`,
   `python code/run_final_pilrn_v5.py`, and
   `python code/run_final_components_v5.py`.
7. Coupled and deterministic controls:
   `python code/run_final_coupled_v5.py`,
   `python code/baseline_idw_v2.py`, and
   `python code/baseline_rf_v2.py`.
8. End-to-end density sensitivity: `run_final_density_v5.py`,
   `run_density_baselines_v6.py`, `run_density_coupled_warm_v6.py`, and
   `run_density_coupled_scratch_v7.py`.
9. Leakage audit: `python code/run_all_v3.py --only audit` — every reported
   model cell is replayed over all 297 months and must pass partition,
   visible-only normalization/training, canary-invariance, and
   archive-reproduction checks. The density-baseline runner performs its own
   140-cell canary audit. The five released batches sum to 1,560 audited model
   cells; their counts are listed in `results/leakage_audit_batches.csv`.
10. Evaluation: `python code/evaluate_final_manuscript_v5.py` reproduces every
   table entry; per-panel figure source data are provided in
   `results/figure_source_data/`.

Temporal protocol: training April 1995–June 2012; validation/model selection
July 2012–March 2016; outer testing April 2016–December 2019. Headline values
pool the ten folds within each seed and report the median of the five
seed-specific station medians.

## Parameter accounting

The parameter-efficiency claim concerns the optimization performed after the
regional prior is frozen:

| System | Frozen prior | Actively optimized for fusion | Complete stored system |
|---|---:|---:|---:|
| PILRN-S | 86,657 | 1,074 | 87,731 |
| Coupled from scratch / warm-start | 0 | 87,763 | 87,763 |
| Embedding-level GAT | 86,657 | 53,761 | 140,418 |

Thus, PILRN-S does not have only 1,074 parameters in total; it reaches the
accuracy of full joint optimization while optimizing only a 1,074-parameter
fusion correction after Stage 1 has been trained and frozen.

## Requirements

Python ≥ 3.10 with `torch`, `numpy`, `pandas`, `matplotlib`, `xarray`,
`netCDF4`, `scikit-learn` (see `requirements.txt`). Experiments were run on a
single consumer GPU; each PILRN-S fold–seed cell trains in under two minutes.

## License and citation

Code is released under the MIT License (see `LICENSE`). Please cite the paper
above. A versioned archive can be deposited with Zenodo when the manuscript
record is finalized.
