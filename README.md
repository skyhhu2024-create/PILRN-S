# PILRN-S — Compact signal-level river-network routing for monthly streamflow estimation in partially gauged networks

Code, audited configuration data, and results accompanying:

> Chen, M., Yong, B., Li, H., Liao, W., Wang, H. — *Compact signal-level
> river-network routing suffices to fuse frozen regional priors with concurrent
> observations for monthly streamflow estimation in partially gauged networks*


PILRN-S estimates monthly streamflow at a held-out gauge. A frozen Regional
LSTM supplies a runoff-depth prior. The second stage learns how much to adjust
that prior using contemporaneous observations on the river network and at
hydrologically similar gauges. Its training priors are cross-fitted within
each outer training fold. The model uses three routing rounds, eight explicit
edge descriptors, a river/similarity edge indicator and a learned fusion gate.

## Run the revised Rhine model

Install the packages in `requirements.txt`. Set `GRDC_DIR` to the directory of
station CSVs and `NC_PATH` to the Rhine ERA5-Land monthly NetCDF. The graph
and the ten random-gauge folds are included in `data/`.

```bash
python code/prepare_forcing.py --extract
python code/run_pilrn_s.py --folds 1 --seeds 42
python code/evaluate_downstream_masked.py --folds 1 --seeds 42
```

Run `python code/run_pilrn_s.py` for all ten folds and five seeds. For each
outer fold the runner fits a regional prior, divides its training gauges into
five inner groups and obtains an out-of-fold prior for each training gauge.
Each test gauge receives the outer model's estimate. The router sees the
cross-fitted priors during training. Random-gauge evaluation retains these
assembled priors at the non-test gauges and hides test-gauge discharge.

For each target, the model ranks up to eight other candidate gauges by
catchment attributes and training-period climate, then skips candidates
already connected by a river edge. The 299-node active Rhine graph has 596
directed river edges and 2,277 directed similarity edges. The full 307-row
arrays store another 64 similarity edges ending at eight excluded gauges.
Similarity edges transmit directly observed runoff depth only. River edges
can relay an updated signal in a later round. A target without donor support
retains its prior. The model uses no separate DAR branch.

The standard random-gauge experiment allows observations at eligible
downstream gauges in the month being estimated. The optional downstream
replay applies a separate mask for each target to the fitted model over
April 2016–December 2019. It checks the unrestricted replay against the
saved predictions and writes paired R, NSE and KGE (2012) to
`downstream_replay/` under the training output directory. Pass the same
`--output-root` to both commands when using a custom directory. Both
experiments estimate discharge in the observation month.

To test the routing and observation rules without licensed input data, run
`python -m unittest discover -s code -p "test_*.py"` from the repository root.

### Rhine random-gauge reference values

Results below are from revised manuscript Table 4 with downstream observations
available. The 164 scored gauges are pooled across folds within each seed.
Metrics are summarized as the median of the five seed-specific summaries.

| Evaluation period | Method | NSE | KGE | R | Gauges with NSE < 0 |
|---|---|---:|---:|---:|---:|
| April 1995–December 2019 | PILRN-S | 0.729 | 0.733 | 0.936 | 13.4% |
| | Regional LSTM | 0.556 | 0.652 | 0.880 | 20.1% |
| | Nearest+DAR | 0.630 | 0.667 | 0.932 | 32.3% |
| April 2016–December 2019 | PILRN-S | 0.744 | 0.714 | 0.947 | 14.6% |
| | Regional LSTM | 0.583 | 0.643 | 0.897 | 21.3% |
| | Nearest+DAR | 0.612 | 0.607 | 0.945 | 30.5% |

## Repository layout

```
code/       revised PILRN-S and original Rhine experiment pipeline
            - run_pilrn_s.py              cross-fitted priors and revised signal router
            - pilrn_similarity.py         similarity graph and observed-donor propagation
            - evaluate_downstream_masked.py   target-specific downstream replay
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
results/    archived evaluations of the original river-only release
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

## Reproducing the original river-only experiments

The original workflow is available as one sequential,
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

## Parameter accounting for the revised model

The parameter-efficiency claim concerns the optimization performed after the
regional prior is frozen:

| System | Frozen prior | Actively optimized for fusion | Complete stored system |
|---|---:|---:|---:|
| PILRN-S | 86,657 | 1,106 | 87,763 |
| Coupled from scratch / warm-start | 0 | 87,763 | 87,763 |
| Embedding-level GAT | 86,657 | 53,761 | 140,418 |

The edge-type indicator adds 32 trainable weights to the original 1,074-parameter
river-only correction. Cross-fitting requires additional regional fits during
training. The stored-system count covers the outer regional model and the
router. The auxiliary inner fits generate the cached out-of-fold priors.

## Requirements

Python ≥ 3.10 with `torch`, `numpy`, `pandas`, `matplotlib`, `xarray`,
`netCDF4`, `scikit-learn` (see `requirements.txt`).

## License and citation

Code is released under the MIT License (see `LICENSE`). Please cite the paper
above. A versioned archive can be deposited with Zenodo when the manuscript
record is finalized.
