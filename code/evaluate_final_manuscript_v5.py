# -*- coding: utf-8 -*-
"""Unified post-processing for manuscript Sections 5.1-5.3.

Every method is re-evaluated from archived monthly predictions against the
same observed reference.  The common metric set is NSE, KGE, PBIAS, FHV, FLV,
VR, aNSE, fraction with NSE < 0, and fraction with NSE > 0.5.

Aggregation follows the author decision:
  1. within each seed, take the station median (or station fraction);
  2. take the median across the five seed-specific summaries.
Deterministic baselines are repeated across seed labels only to support paired
comparison with stochastic PILRN-S; their predictions are not retrained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
from run_stability_10x5 import DECLARED_SEEDS


FOLDS = list(range(1, 11))
OUT = C.SP_V3 / "Final_Manuscript_Evaluation_v5"
METRICS = ["NSE", "KGE", "PBIAS", "FHV", "FLV", "VR", "aNSE"]
METHODS = [
    "PILRN-S",
    "Regional LSTM",
    "Nearest+DAR",
    "IDW+DAR",
    "Random forest",
    "Coupled from scratch",
    "Coupled warm-start",
    "GAT",
]
DETERMINISTIC = {"Nearest+DAR", "IDW+DAR", "Random forest"}
SECTIONS = {
    "5.1": [
        "PILRN-S",
        "Regional LSTM",
        "Nearest+DAR",
        "IDW+DAR",
        "Random forest",
    ],
    "5.2": [
        "PILRN-S",
        "Coupled from scratch",
        "Coupled warm-start",
    ],
    "5.3": ["PILRN-S", "GAT"],
}
PARAMETERS = {
    "PILRN-S": 1074,
    "GAT": 53761,
    "Coupled from scratch": 87763,
    "Coupled warm-start": 87763,
}


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def directory(method: str, seed: int, fold: int) -> Path:
    if method == "PILRN-S":
        root = C.SP_V3 / "PILRN_sig_final_noarea_r3_v5"
        return root / run_tag(seed, fold)
    if method == "Regional LSTM":
        return C.REGIONAL_OUT / run_tag(seed, fold)
    if method == "Nearest+DAR":
        return C.SP_V3 / "Baseline_IDW_v2" / "nearest" / f"fold_{fold:02d}"
    if method == "IDW+DAR":
        return C.SP_V3 / "Baseline_IDW_v2" / "idw" / f"fold_{fold:02d}"
    if method == "Random forest":
        return C.SP_V3 / "Baseline_RF_v2" / f"fold_{fold:02d}"
    if method == "Coupled from scratch":
        root = C.SP_V3 / "Coupled_scratch_final_noarea_r3_v5"
        return root / run_tag(seed, fold)
    if method == "Coupled warm-start":
        root = C.SP_V3 / "Coupled_warm_final_noarea_r3_v5"
        return root / run_tag(seed, fold)
    if method == "GAT":
        root = C.SP_V3 / "GAT_physical_comparator_v4"
        return root / run_tag(seed, fold)
    raise ValueError(method)


def full_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    climatology: dict[int, float],
    months: np.ndarray,
) -> dict[str, float] | None:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 10:
        return None
    observed = observed[valid]
    predicted = predicted[valid]
    months = months[valid]
    variance = max(float(((observed - observed.mean()) ** 2).sum()), 1e-12)
    nse = 1.0 - float(((observed - predicted) ** 2).sum()) / variance
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    bias_ratio = float(predicted.mean()) / max(float(observed.mean()), 1e-12)
    variability_ratio_cv = (
        float(predicted.std()) / max(float(predicted.mean()), 1e-12)
    ) / max(
        float(observed.std()) / max(float(observed.mean()), 1e-12),
        1e-12,
    )
    kge = 1.0 - float(
        np.sqrt(
            (correlation - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
            + (variability_ratio_cv - 1.0) ** 2
        )
    )
    pbias = (
        100.0
        * float((predicted - observed).sum())
        / max(float(observed.sum()), 1e-12)
    )
    q90 = float(np.quantile(observed, 0.9))
    q30 = float(np.quantile(observed, 0.3))
    high = observed >= q90
    low = observed <= q30
    fhv = (
        100.0
        * float((predicted[high] - observed[high]).sum())
        / max(float(observed[high].sum()), 1e-12)
        if high.sum() > 2
        else np.nan
    )
    flv = (
        100.0
        * float((predicted[low] - observed[low]).sum())
        / max(float(observed[low].sum()), 1e-12)
        if low.sum() > 2
        else np.nan
    )
    vr = float(predicted.std()) / max(float(observed.std()), 1e-12)
    seasonal = np.asarray(
        [climatology.get(int(month), np.nan) for month in months],
        dtype=float,
    )
    seasonal_valid = np.isfinite(seasonal)
    anse = np.nan
    if seasonal_valid.sum() > 10:
        observed_anomaly = observed[seasonal_valid] - seasonal[seasonal_valid]
        predicted_anomaly = predicted[seasonal_valid] - seasonal[seasonal_valid]
        anomaly_variance = max(
            float(
                (
                    (observed_anomaly - observed_anomaly.mean())
                    ** 2
                ).sum()
            ),
            1e-12,
        )
        anse = 1.0 - float(
            ((observed_anomaly - predicted_anomaly) ** 2).sum()
        ) / anomaly_variance
    return {
        "NSE": nse,
        "KGE": kge,
        "PBIAS": pbias,
        "FHV": fhv,
        "FLV": flv,
        "VR": vr,
        "aNSE": anse,
        "r": correlation,
        "n": int(valid.sum()),
    }


def evaluate_prediction(path: Path) -> dict[str, float] | None:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"predicted_flow_m3s", "observed_flow_m3s"}
    if not required.issubset(frame.columns):
        return None
    periods = pd.PeriodIndex(frame["date"], freq="M")
    months = periods.month.to_numpy()
    observed = pd.to_numeric(
        frame["observed_flow_m3s"], errors="coerce"
    ).to_numpy(dtype=float)
    predicted = pd.to_numeric(
        frame["predicted_flow_m3s"], errors="coerce"
    ).to_numpy(dtype=float)
    training = periods <= pd.Period(C.TRAIN_END, "M")
    climatology = (
        pd.DataFrame({"month": months[training], "observed": observed[training]})
        .groupby("month")["observed"]
        .mean()
        .to_dict()
    )
    test = periods > pd.Period(C.VAL_END, "M")
    full = full_metrics(observed, predicted, climatology, months)
    test_metrics = full_metrics(
        observed[test], predicted[test], climatology, months[test]
    )
    if full is None or test_metrics is None:
        return None
    output: dict[str, float] = {}
    output.update({f"{key}_full": value for key, value in full.items()})
    output.update({f"{key}_test": value for key, value in test_metrics.items()})
    return output


def collect() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for seed in DECLARED_SEEDS:
            source_seed = C.SEED if method in DETERMINISTIC else seed
            for fold in FOLDS:
                canonical = pd.read_csv(
                    directory("PILRN-S", seed, fold)
                    / "evaluation_holdout.csv",
                    dtype={"station_id": str},
                )
                target_stations = canonical.station_id.astype(str).tolist()
                prediction_dir = (
                    directory(method, source_seed, fold)
                    / "predictions_by_station"
                )
                if not prediction_dir.exists():
                    raise FileNotFoundError(
                        f"Missing prediction directory: {prediction_dir}"
                    )
                for station_id in target_stations:
                    path = prediction_dir / f"{station_id}.csv"
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Missing station prediction: {path}"
                        )
                    metrics = evaluate_prediction(path)
                    if metrics is None:
                        continue
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "source_seed": source_seed,
                            "fold": fold,
                            "station_id": path.stem,
                            **metrics,
                        }
                    )
    frame = pd.DataFrame(rows)
    expected_rows = len(METHODS) * len(DECLARED_SEEDS) * 164
    if len(frame) != expected_rows:
        counts = (
            frame.groupby(["method", "seed"])
            .size()
            .rename("n")
            .reset_index()
        )
        raise RuntimeError(
            f"Expected {expected_rows} station-seed rows, got {len(frame)}.\n"
            f"{counts.to_string(index=False)}"
        )
    duplicates = frame.duplicated(["method", "seed", "station_id"])
    if duplicates.any():
        raise RuntimeError("Duplicate method-seed-station keys detected.")
    return frame


def summarize_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, seed), group in frame.groupby(["method", "seed"], sort=False):
        row: dict[str, object] = {
            "method": method,
            "seed": int(seed),
            "n_stations": int(len(group)),
            "folds": int(group.fold.nunique()),
            "NSE_below_0_percent": 100.0 * float((group.NSE_full < 0).mean()),
            "NSE_above_0_5_percent": 100.0
            * float((group.NSE_full > 0.5).mean()),
        }
        for metric in METRICS:
            row[f"{metric}_median"] = float(group[f"{metric}_full"].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "seed"])


def summarize_methods(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in seed_summary.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "n_seeds": int(group.seed.nunique()),
            "NSE_below_0_percent": float(
                group.NSE_below_0_percent.median()
            ),
            "NSE_above_0_5_percent": float(
                group.NSE_above_0_5_percent.median()
            ),
            "trainable_parameters": PARAMETERS.get(method, np.nan),
        }
        for metric in METRICS:
            row[metric] = float(group[f"{metric}_median"].median())
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").loc[METHODS].reset_index()


def paired_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    contrasts = {
        "5.1": [
            ("PILRN-S", "Regional LSTM"),
            ("PILRN-S", "Nearest+DAR"),
            ("PILRN-S", "IDW+DAR"),
            ("PILRN-S", "Random forest"),
        ],
        "5.2": [
            ("PILRN-S", "Coupled from scratch"),
            ("PILRN-S", "Coupled warm-start"),
        ],
        "5.3": [("PILRN-S", "GAT")],
    }
    seed_rows: list[dict[str, object]] = []
    keys = ["seed", "station_id", "fold"]
    for section, pairs in contrasts.items():
        for reference, comparator in pairs:
            left = frame[frame.method == reference][
                keys + [f"{metric}_full" for metric in METRICS]
            ]
            right = frame[frame.method == comparator][
                keys + [f"{metric}_full" for metric in METRICS]
            ]
            joined = left.merge(
                right,
                on=keys,
                suffixes=("_reference", "_comparator"),
                validate="one_to_one",
            )
            for seed, group in joined.groupby("seed", sort=False):
                row: dict[str, object] = {
                    "section": section,
                    "reference": reference,
                    "comparator": comparator,
                    "seed": int(seed),
                    "n": int(len(group)),
                    "NSE_win_percent": 100.0
                    * float(
                        (
                            group.NSE_full_reference
                            > group.NSE_full_comparator
                        ).mean()
                    ),
                    "comparator_negative_NSE_count": int(
                        (group.NSE_full_comparator < 0).sum()
                    ),
                    "negative_to_positive_recovered_count": int(
                        (
                            (group.NSE_full_comparator < 0)
                            & (group.NSE_full_reference > 0)
                        ).sum()
                    ),
                }
                for metric in METRICS:
                    delta = (
                        group[f"{metric}_full_reference"]
                        - group[f"{metric}_full_comparator"]
                    )
                    row[f"paired_median_delta_{metric}"] = float(delta.median())
                seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    aggregate_rows: list[dict[str, object]] = []
    numeric = [
        column
        for column in seeds.columns
        if column not in {"section", "reference", "comparator", "seed"}
    ]
    for keys_value, group in seeds.groupby(
        ["section", "reference", "comparator"], sort=False
    ):
        section, reference, comparator = keys_value
        row: dict[str, object] = {
            "section": section,
            "reference": reference,
            "comparator": comparator,
            "n_seeds": int(group.seed.nunique()),
        }
        for column in numeric:
            row[column] = float(group[column].median())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    seeds.to_csv(OUT / "paired_seed_evidence.csv", index=False)
    return aggregate


def stability_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regional = frame[frame.method == "Regional LSTM"][
        ["seed", "station_id", "fold", "NSE_full"]
    ].rename(columns={"NSE_full": "NSE_regional"})
    for method in [
        "PILRN-S",
        "Coupled from scratch",
        "Coupled warm-start",
        "GAT",
    ]:
        group = frame[frame.method == method]
        station_sd = group.groupby("station_id").NSE_full.std(ddof=1)
        joined = group[
            ["seed", "station_id", "fold", "NSE_full"]
        ].merge(
            regional,
            on=["seed", "station_id", "fold"],
            validate="one_to_one",
        )
        seed_gain = (
            joined.assign(delta=joined.NSE_full - joined.NSE_regional)
            .groupby("seed")
            .agg(
                paired_median_gain=("delta", "median"),
                improved_fraction=("delta", lambda value: (value > 0).mean()),
            )
        )
        rows.append(
            {
                "method": method,
                "median_same_station_NSE_SD": float(station_sd.median()),
                "same_station_NSE_SD_q75": float(station_sd.quantile(0.75)),
                "median_paired_gain_vs_regional": float(
                    seed_gain.paired_median_gain.median()
                ),
                "median_improved_percent_vs_regional": 100.0
                * float(seed_gain.improved_fraction.median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    definitions = {
        "aggregation": (
            "station median within each seed, followed by the median of the "
            "five seed-specific summaries"
        ),
        "period": "full study period for headline metrics",
        "reference": "observed discharge",
        "NSE": "Nash-Sutcliffe efficiency",
        "KGE": "Kling-Gupta efficiency",
        "PBIAS": "overall volume bias (%)",
        "FHV": "volume bias for months with observed flow >= station Q90 (%)",
        "FLV": "volume bias for months with observed flow <= station Q30 (%)",
        "VR": "standard deviation(prediction) / standard deviation(observation)",
        "aNSE": (
            "NSE after subtracting the station's training-period observed "
            "monthly climatology from observations and predictions"
        ),
        "NSE_below_0_percent": "percentage of scored stations with NSE < 0",
        "NSE_above_0_5_percent": "percentage of scored stations with NSE > 0.5",
        "deterministic_baselines": (
            "nearest+DAR, IDW+DAR and RF predictions are repeated across seed "
            "labels solely for paired aggregation"
        ),
    }
    (OUT / "metric_definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stations = collect()
    stations.to_csv(OUT / "station_seed_metrics.csv", index=False)
    seeds = summarize_seeds(stations)
    seeds.to_csv(OUT / "seed_summary.csv", index=False)
    methods = summarize_methods(seeds)
    methods.to_csv(OUT / "method_summary.csv", index=False)
    for section, section_methods in SECTIONS.items():
        section_frame = (
            methods.set_index("method").loc[section_methods].reset_index()
        )
        section_frame.to_csv(
            OUT / f"section_{section.replace('.', '_')}_common_metrics.csv",
            index=False,
        )
    paired = paired_evidence(stations)
    paired.to_csv(OUT / "paired_aggregate_evidence.csv", index=False)
    stability = stability_evidence(stations)
    stability.to_csv(OUT / "stability_evidence.csv", index=False)

    result = {
        "station_seed_rows": int(len(stations)),
        "methods": int(stations.method.nunique()),
        "seeds": int(stations.seed.nunique()),
        "stations_per_method_seed": 164,
        "common_metrics": METRICS
        + ["NSE_below_0_percent", "NSE_above_0_5_percent"],
        "section_tables": {
            section: section_methods for section, section_methods in SECTIONS.items()
        },
    }
    (OUT / "evaluation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(methods.round(3).to_string(index=False), flush=True)
    print(paired.round(3).to_string(index=False), flush=True)
    print(stability.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
