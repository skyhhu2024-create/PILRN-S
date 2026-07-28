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

## Headline evaluation (manuscript Table 4)

All methods use the same 164 scored gauges and ten fixed leave-station-out
folds. Values below reproduce manuscript Table 4; stochastic methods report
the median of five seed-specific station medians. For PILRN-S, “active
parameters” refers to the Stage-2 fusion correction optimized after the
86,657-parameter Regional-LSTM prior has been frozen.

| Method | NSE | KGE | PBIAS (%) | FHV (%) | FLV (%) | VR | aNSE | NSE < 0 (%) | NSE > 0.5 (%) | Active parameters | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PILRN-S** | **0.656** | **0.697** | **−1.9** | **−12.0** | **14.9** | **0.882** | **0.435** | **15.9** | **64.6** | **1,074** | 5 |
| Regional LSTM | 0.556 | 0.652 | −2.0 | −13.1 | 15.2 | 0.891 | 0.274 | 20.1 | 56.1 | 86,657 | 5 |
| Nearest+DAR | 0.630 | 0.667 | 2.3 | 0.0 | 10.4 | 1.020 | 0.359 | 32.3 | 55.5 | — | 5 |
| IDW+DAR | 0.268 | 0.428 | 17.5 | −13.5 | 108.6 | 0.815 | −0.321 | 39.0 | 31.7 | — | 5 |
| Random forest | 0.604 | 0.655 | −8.7 | −24.8 | 17.5 | 0.780 | 0.353 | 13.4 | 62.2 | — | 5 |
| Coupled from scratch | 0.660 | 0.687 | −3.2 | −14.1 | 19.1 | 0.863 | 0.438 | 20.1 | 64.0 | 87,763 | 5 |
| Coupled warm-start | 0.642 | 0.695 | −2.8 | −12.1 | 11.5 | 0.915 | 0.407 | 18.9 | 62.8 | 87,763 | 5 |
| Embedding-level GAT | 0.627 | 0.672 | −2.3 | −13.4 | 13.9 | 0.905 | 0.331 | 14.6 | 61.6 | 53,761 | 5 |

### What the evidence supports

- **Concurrent observations add useful local information to the regional
  prior.** PILRN-S raises median NSE from 0.556 to 0.656 and KGE from 0.652 to
  0.697 relative to the Regional LSTM. The paired station-median NSE gain is
  +0.050, and 77.4% of gauges improve.
- **Signal-level routing repairs severe donor incompatibility.** Of the 53
  gauges where nearest+DAR has negative NSE, PILRN-S restores positive NSE at
  35, while reducing the overall negative-NSE fraction from 32.3% to 15.9%.
- **Freezing is an effective optimization constraint, not an accuracy
  guarantee.** PILRN-S, coupled-from-scratch, and coupled-warm-start have
  statistically indistinguishable paired central accuracy, while PILRN-S
  optimizes only the 1,074-parameter fusion correction rather than jointly
  updating all 87,763 parameters. It also has fewer negative-NSE gauges than
  either coupled control and lower same-station seed sensitivity.
- **Explicit river descriptors provide strong parameter efficiency.** The
  1,074-parameter PILRN-S fusion module reaches the same overall performance
  range as the 53,761-parameter embedding-level GAT graph module, a 50.1-fold
  difference in actively optimized Stage-2 parameters.
- **The supporting experiments define the boundary of the claim.** Removing
  descriptor-conditioned edge weights reduces NSE/KGE to 0.641/0.684, and
  PILRN-S remains the strongest non-coupled estimator in the 50% and 25%
  training-gauge-density experiments. These results support sufficiency and
  robustness in the evaluated partially gauged setting, not universal
  superiority over every model or deployment regime.

The corresponding station-level, paired, stability, ablation, and density
evidence is released under [`results/`](results/).

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
            - evaluate_final_manuscript_v5.py  unified post-processing (manuscript Table 4)
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
            - method_summary.csv, seed_summary.csv, section_5_*.csv
                                              (Table 4 and Sections 5.1-5.3)
            - paired_significance_paperconv_v6.csv  (SI Table S13: CIs, Wilcoxon)
            - stability_evidence.csv, paired_*_evidence.csv (Tables S9-S10)
            - leakage_audit_batches.csv   released all-month audit batch summary
            - figure_source_data/              per-panel source CSVs
            - README.md                   scope and redistribution notes
```

## Manuscript-to-release mapping

Some source-data filenames retain figure numbers from an earlier manuscript
draft for compatibility with the evaluation scripts. The current manuscript
mapping is:

| Current manuscript item | Released source data |
|---|---|
| Table 4 — unified evaluation | `results/method_summary.csv`, `results/station_seed_metrics.csv`, `results/section_5_1_common_metrics.csv` |
| Figure 4 — non-graph baselines | `results/figure_source_data/fig04_non_graph_common_metrics.csv` |
| Figure 5 — station-level gains | `results/figure_source_data/fig05_paired_summary.csv`, `fig05_station_median_metrics.csv` |
| Figure 7 — frozen versus coupled | `results/figure_source_data/fig07_*.csv` |
| Figure 8 — PILRN-S versus GAT | `results/figure_source_data/fig08_*.csv` |
| Figure 9 — router interpretability | `results/figure_source_data/Fig09_panel_*.csv` |
| **Figure 10 — propagation depth and components** | `results/figure_source_data/fig09_round_seed_metrics.csv`, `fig09_component_summary.csv` |
| **Figure 11 — training-gauge density** | `results/figure_source_data/fig10_density_*.csv` |
| Figure S1 — closure diagnostic | `results/figure_source_data/FigS1_closure_errors.csv` |
| Figure S2 — screening thresholds | `results/figure_source_data/FigS2_source.csv` |

Thus, the legacy `fig09_round_*` prefix corresponds to current Figure 10, and
the legacy `fig10_density_*` prefix corresponds to current Figure 11.

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
