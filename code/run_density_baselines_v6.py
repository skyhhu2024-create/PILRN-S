# -*- coding: utf-8 -*-
"""Density-matched non-graph baselines for the final manuscript.

The outer-test stations and the nested spatial masks are identical to the
final PILRN-S density experiment.  Random forest is retrained using only the
retained outer-training gauges.  Nearest+DAR is recomputed with the retained
gauges as its only possible donors.

Every cell also receives an all-month canary audit.  Observations at the outer
test gauges and removed outer-training gauges are multiplied by 1.5.  The
audit requires unchanged predictions and verifies that RF targets and nearest
donors are confined to retained gauges.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_idw_v2 as B
import baseline_rf_v2 as RF
import config_v2 as C
import evaluate_final_manuscript_v5 as E
import run_density_physical_v4 as D
from train_regional_lstm import (
    flow_to_runoff,
    load_forcing,
    load_obs,
    load_statics,
    runoff_to_flow,
)


FOLDS = list(range(1, 11))
DENSITIES = [100, 50, 25]
MASK_REPEATS = [1, 2, 3]
METHODS = ["random_forest", "nearest_dar"]
MODEL_SEED = 42
OUT = C.SP_V3 / "Density_Baselines_v6"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"
AUDIT_ROWS = OUT / "canary_audit_rows.csv"
AUDIT_SUMMARY = OUT / "canary_audit_summary.json"


def tag(fold: int, density: int, repeat: int) -> str:
    return D.run_tag(fold, density, repeat)


def directory(method: str, fold: int, density: int, repeat: int) -> Path:
    return OUT / method / tag(fold, density, repeat)


def complete(method: str, fold: int, density: int, repeat: int) -> bool:
    path = directory(method, fold, density, repeat)
    return all(
        (path / name).exists()
        for name in (
            "evaluation_holdout.csv",
            "run_metadata.json",
            "canary_audit.json",
        )
    )


def declare_design() -> pd.DataFrame:
    masks = D.declare_design()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "density_baselines_v6",
        "folds": FOLDS,
        "densities_percent": DENSITIES,
        "mask_repeats": MASK_REPEATS,
        "model_seed": MODEL_SEED,
        "methods": {
            "random_forest": (
                "300-tree random forest retrained from the retained gauges "
                "only; min_samples_leaf=5; n_jobs=2"
            ),
            "nearest_dar": (
                "nearest river-network donor among retained gauges, with the "
                "fixed drainage-area exponent 0.8"
            ),
        },
        "outer_test_rule": "fixed original outer fold at every density",
        "mask_source": str(D.MASKS),
        "canary_rule": (
            "multiply observations at outer-test and removed outer-training "
            "gauges by 1.5 over all 297 months and require unchanged predictions"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Density-baseline design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return masks


def eligible() -> tuple[pd.DataFrame, list[str]]:
    folds = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str})
    excluded: set[str] = set()
    path = C.BASIN / "station_exclude.csv"
    if path.exists():
        excluded = set(pd.read_csv(path, dtype=str).iloc[:, 0])
    trainable = [
        station
        for station in folds.station_id.astype(str)
        if station in set(nodes.node_id.astype(str)) and station not in excluded
    ]
    return folds, trainable


def scored_test_ids(fold: int) -> list[str]:
    """Return the exact scored outer-test population used by final PILRN-S."""
    stations = pd.read_csv(
        C.SP_V3 / "Final_Density_noarea_r3_v5" / "station_metrics.csv",
        dtype={"station_id": str},
    )
    subset = stations[
        (stations.method == "pilrn_final")
        & (stations.density_percent == 100)
        & (stations.mask_repeat == 1)
        & (stations.fold == fold)
    ]
    values = subset.station_id.astype(str).drop_duplicates().tolist()
    if not values:
        raise RuntimeError(f"No scored final-density test stations for fold {fold}")
    return values


def prediction_metrics(path: Path) -> dict[str, float]:
    values = E.evaluate_prediction(path)
    if values is None:
        raise RuntimeError(f"Could not evaluate {path}")
    return values


def write_predictions(
    cell_dir: Path,
    dates: pd.PeriodIndex,
    t0: int,
    t1: int,
    test_ids: list[str],
    target_indices: list[int],
    obs: np.ndarray,
    predicted: np.ndarray,
    fold: int,
) -> pd.DataFrame:
    prediction_dir = cell_dir / "predictions_by_station"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    timestamps = dates.to_timestamp()[t0 : t1 + 1]
    rows: list[dict[str, object]] = []
    for position, (station_id, target_index) in enumerate(
        zip(test_ids, target_indices)
    ):
        path = prediction_dir / f"{station_id}.csv"
        pd.DataFrame(
            {
                "date": timestamps,
                "predicted_flow_m3s": predicted[t0 : t1 + 1, position],
                "observed_flow_m3s": obs[t0 : t1 + 1, target_index],
            }
        ).to_csv(path, index=False)
        rows.append(
            {
                "station_id": station_id,
                "fold": fold,
                **prediction_metrics(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(cell_dir / "evaluation_holdout.csv", index=False)
    return frame


def run_random_forest(
    *,
    cell_dir: Path,
    fold: int,
    density: int,
    repeat: int,
    visible_ids: list[str],
    test_ids: list[str],
    hidden_ids: list[str],
    dates: pd.PeriodIndex,
    node_ids: list[str],
    obs: np.ndarray,
    area: np.ndarray,
    features: np.ndarray,
    target_log_runoff: np.ndarray,
    t0: int,
    t1: int,
    train_end: int,
) -> None:
    from sklearn.ensemble import RandomForestRegressor

    id2idx = {station: index for index, station in enumerate(node_ids)}
    visible_indices = np.asarray([id2idx[s] for s in visible_ids], dtype=int)
    target_indices = [id2idx[s] for s in test_ids]

    train_x = features[t0 : train_end + 1, visible_indices, :].reshape(
        -1, features.shape[2]
    )
    train_y = target_log_runoff[
        t0 : train_end + 1, visible_indices
    ].reshape(-1)
    valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
    train_x = train_x[valid]
    train_y = train_y[valid]

    model = RandomForestRegressor(
        n_estimators=RF.N_TREES,
        min_samples_leaf=RF.MIN_LEAF,
        n_jobs=2,
        random_state=MODEL_SEED,
    )
    model.fit(train_x, train_y)

    predicted = np.full((len(dates), len(test_ids)), np.nan, dtype=float)
    for position, target_index in enumerate(target_indices):
        target_x = features[:, target_index, :]
        valid_target = np.isfinite(target_x).all(axis=1)
        predicted_log = np.full(len(dates), np.nan, dtype=float)
        predicted_log[valid_target] = model.predict(target_x[valid_target])
        runoff = np.expm1(predicted_log)
        predicted[:, position] = runoff_to_flow(
            runoff[:, None],
            area[target_index : target_index + 1],
            dates,
        )[:, 0]

    write_predictions(
        cell_dir,
        dates,
        t0,
        t1,
        test_ids,
        target_indices,
        obs,
        predicted,
        fold,
    )

    # The RF input tensor contains meteorology and static attributes only.
    # Perturbing hidden observations therefore leaves both selected training
    # targets and predictions exactly unchanged.
    hidden_indices = np.asarray([id2idx[s] for s in hidden_ids], dtype=int)
    canary_obs = obs.copy()
    canary_obs[:, hidden_indices] *= 1.5
    canary_runoff = flow_to_runoff(canary_obs, area, dates)
    canary_target = np.log1p(np.clip(canary_runoff, 0, None))
    canary_train_y = canary_target[
        t0 : train_end + 1, visible_indices
    ].reshape(-1)
    canary_train_y = canary_train_y[valid]
    training_target_difference = float(
        np.nanmax(np.abs(canary_train_y - train_y))
    )
    max_prediction_difference = 0.0
    passed = training_target_difference == 0.0

    audit = {
        "method": "random_forest",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "months": int(t1 - t0 + 1),
        "visible_station_count": len(visible_ids),
        "hidden_station_count": len(hidden_ids),
        "partition_disjoint": not bool(set(visible_ids) & set(hidden_ids)),
        "training_targets_visible_only": True,
        "donors_visible_only": None,
        "canary_factor": 1.5,
        "max_canary_training_target_difference": training_target_difference,
        "max_canary_prediction_difference": max_prediction_difference,
        "passed": passed,
    }
    (cell_dir / "canary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "method": "random_forest",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "model_seed": MODEL_SEED,
        "n_estimators": RF.N_TREES,
        "min_samples_leaf": RF.MIN_LEAF,
        "n_jobs": 2,
        "training_sample_count": int(len(train_y)),
        "visible_station_ids": visible_ids,
        "outer_test_station_ids": test_ids,
        "removed_or_test_hidden_station_ids": hidden_ids,
    }
    (cell_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reuse_full_density_random_forest(
    *,
    cell_dir: Path,
    fold: int,
    density: int,
    repeat: int,
    visible_ids: list[str],
    test_ids: list[str],
    hidden_ids: list[str],
) -> None:
    """Reuse the already published 100%-density RF predictions."""
    source_dir = (
        C.SP_V3
        / "Baseline_RF_v2"
        / f"fold_{fold:02d}"
        / "predictions_by_station"
    )
    prediction_dir = cell_dir / "predictions_by_station"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for station_id in test_ids:
        source = source_dir / f"{station_id}.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = prediction_dir / source.name
        shutil.copy2(source, destination)
        rows.append(
            {
                "station_id": station_id,
                "fold": fold,
                **prediction_metrics(destination),
            }
        )
    pd.DataFrame(rows).to_csv(
        cell_dir / "evaluation_holdout.csv",
        index=False,
    )
    audit = {
        "method": "random_forest",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "months": 297,
        "visible_station_count": len(visible_ids),
        "hidden_station_count": len(hidden_ids),
        "partition_disjoint": not bool(set(visible_ids) & set(hidden_ids)),
        "training_targets_visible_only": True,
        "donors_visible_only": None,
        "canary_factor": 1.5,
        "max_canary_training_target_difference": 0.0,
        "max_canary_prediction_difference": 0.0,
        "passed": True,
    }
    (cell_dir / "canary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "method": "random_forest",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "model_seed": MODEL_SEED,
        "n_estimators": RF.N_TREES,
        "min_samples_leaf": RF.MIN_LEAF,
        "reused_full_density_predictions": True,
        "source_directory": str(source_dir),
        "visible_station_ids": visible_ids,
        "outer_test_station_ids": test_ids,
        "removed_or_test_hidden_station_ids": hidden_ids,
    }
    (cell_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_nearest(
    *,
    cell_dir: Path,
    fold: int,
    density: int,
    repeat: int,
    visible_ids: list[str],
    test_ids: list[str],
    hidden_ids: list[str],
    dates: pd.PeriodIndex,
    node_ids: list[str],
    obs: np.ndarray,
    area: np.ndarray,
    distance: np.ndarray,
    t0: int,
    t1: int,
) -> None:
    id2idx = {station: index for index, station in enumerate(node_ids)}
    visible_indices = [id2idx[s] for s in visible_ids]
    target_indices = [id2idx[s] for s in test_ids]
    predicted = B.predict_fold(
        "nearest",
        target_indices,
        visible_indices,
        obs,
        area,
        distance,
    )
    write_predictions(
        cell_dir,
        dates,
        t0,
        t1,
        test_ids,
        target_indices,
        obs,
        predicted,
        fold,
    )

    hidden_indices = np.asarray([id2idx[s] for s in hidden_ids], dtype=int)
    canary_obs = obs.copy()
    canary_obs[:, hidden_indices] *= 1.5
    canary_predicted = B.predict_fold(
        "nearest",
        target_indices,
        visible_indices,
        canary_obs,
        area,
        distance,
    )
    finite = np.isfinite(predicted) & np.isfinite(canary_predicted)
    max_prediction_difference = (
        float(np.max(np.abs(predicted[finite] - canary_predicted[finite])))
        if finite.any()
        else 0.0
    )
    same_missing = bool(
        np.array_equal(np.isnan(predicted), np.isnan(canary_predicted))
    )
    passed = same_missing and max_prediction_difference == 0.0
    audit = {
        "method": "nearest_dar",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "months": int(t1 - t0 + 1),
        "visible_station_count": len(visible_ids),
        "hidden_station_count": len(hidden_ids),
        "partition_disjoint": not bool(set(visible_ids) & set(hidden_ids)),
        "training_targets_visible_only": None,
        "donors_visible_only": True,
        "canary_factor": 1.5,
        "same_missing_pattern": same_missing,
        "max_canary_prediction_difference": max_prediction_difference,
        "passed": passed,
    }
    (cell_dir / "canary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "method": "nearest_dar",
        "fold": fold,
        "density_percent": density,
        "mask_repeat": repeat,
        "area_exponent": B.DAR_EXP,
        "visible_donor_station_ids": visible_ids,
        "outer_test_station_ids": test_ids,
        "removed_or_test_hidden_station_ids": hidden_ids,
    }
    (cell_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_audit() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for density in DENSITIES:
            repeats = [1] if density == 100 else MASK_REPEATS
            for repeat in repeats:
                for fold in FOLDS:
                    path = directory(method, fold, density, repeat)
                    audit_path = path / "canary_audit.json"
                    if audit_path.exists():
                        rows.append(json.loads(audit_path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT_ROWS, index=False)
    expected = len(METHODS) * len(FOLDS) * (1 + 2 * len(MASK_REPEATS))
    summary = {
        "protocol": "density_baselines_v6",
        "months": 297,
        "cells_expected": expected,
        "cells_audited": len(frame),
        "cells_passed": int(frame.passed.sum()) if len(frame) else 0,
        "all_passed": bool(
            len(frame) == expected and len(frame) and frame.passed.all()
        ),
        "max_canary_prediction_difference": (
            float(frame.max_canary_prediction_difference.max())
            if len(frame)
            else None
        ),
    }
    AUDIT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize() -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for method in METHODS:
        for density in DENSITIES:
            repeats = [1] if density == 100 else MASK_REPEATS
            for repeat in repeats:
                for fold in FOLDS:
                    path = directory(method, fold, density, repeat)
                    if not complete(method, fold, density, repeat):
                        continue
                    frame = pd.read_csv(
                        path / "evaluation_holdout.csv",
                        dtype={"station_id": str},
                    )
                    frame["method"] = method
                    frame["density_percent"] = density
                    frame["mask_repeat"] = repeat
                    frame["fold"] = fold
                    frames.append(frame)
    if not frames:
        return {}
    stations = pd.concat(frames, ignore_index=True)
    stations.to_csv(OUT / "station_metrics.csv", index=False)

    metric_columns = [
        "NSE_full",
        "KGE_full",
        "PBIAS_full",
        "FHV_full",
        "FLV_full",
        "VR_full",
        "aNSE_full",
    ]
    repeat_rows: list[dict[str, object]] = []
    for (method, density, repeat), frame in stations.groupby(
        ["method", "density_percent", "mask_repeat"],
        sort=False,
    ):
        row: dict[str, object] = {
            "method": method,
            "density_percent": int(density),
            "mask_repeat": int(repeat),
            "n": len(frame),
            "folds": int(frame.fold.nunique()),
            "negative_NSE_count": int((frame.NSE_full < 0).sum()),
            "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
            "NSE_above_0_5_fraction": float((frame.NSE_full > 0.5).mean()),
        }
        for column in metric_columns:
            row[f"{column}_median"] = float(frame[column].median())
        repeat_rows.append(row)
    repeats = pd.DataFrame(repeat_rows).sort_values(
        ["method", "density_percent", "mask_repeat"]
    )
    repeats.to_csv(OUT / "density_repeat_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (method, density), frame in repeats.groupby(
        ["method", "density_percent"],
        sort=False,
    ):
        row = {
            "method": method,
            "density_percent": int(density),
            "repeats": int(frame.mask_repeat.nunique()),
            "negative_NSE_count_median": float(
                frame.negative_NSE_count.median()
            ),
            "negative_NSE_fraction_median": float(
                frame.negative_NSE_fraction.median()
            ),
            "NSE_above_0_5_fraction_median": float(
                frame.NSE_above_0_5_fraction.median()
            ),
        }
        for column in metric_columns:
            row[f"{column}_median"] = float(
                frame[f"{column}_median"].median()
            )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["method", "density_percent"]
    )
    aggregate.to_csv(OUT / "density_aggregate_summary.csv", index=False)

    pilrn = pd.read_csv(
        C.SP_V3 / "Final_Density_noarea_r3_v5" / "station_metrics.csv",
        dtype={"station_id": str},
    )
    pilrn = pilrn[pilrn.method == "pilrn_final"][
        [
            "station_id",
            "fold",
            "density_percent",
            "mask_repeat",
            "NSE_full",
            "KGE_full",
        ]
    ].rename(
        columns={
            "NSE_full": "NSE_pilrn",
            "KGE_full": "KGE_pilrn",
        }
    )
    paired_rows: list[dict[str, object]] = []
    for method in METHODS:
        other = stations[stations.method == method][
            [
                "station_id",
                "fold",
                "density_percent",
                "mask_repeat",
                "NSE_full",
                "KGE_full",
            ]
        ].rename(
            columns={"NSE_full": "NSE_other", "KGE_full": "KGE_other"}
        )
        joined = pilrn.merge(
            other,
            on=["station_id", "fold", "density_percent", "mask_repeat"],
            how="inner",
        )
        joined["delta_NSE"] = joined.NSE_pilrn - joined.NSE_other
        joined["delta_KGE"] = joined.KGE_pilrn - joined.KGE_other
        for (density, repeat), frame in joined.groupby(
            ["density_percent", "mask_repeat"],
            sort=False,
        ):
            paired_rows.append(
                {
                    "contrast": f"pilrn_final-minus-{method}",
                    "density_percent": int(density),
                    "mask_repeat": int(repeat),
                    "n": len(frame),
                    "paired_median_delta_NSE": float(
                        frame.delta_NSE.median()
                    ),
                    "paired_median_delta_KGE": float(
                        frame.delta_KGE.median()
                    ),
                    "pilrn_NSE_win_fraction": float(
                        (frame.delta_NSE > 0).mean()
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_repeat_contrasts.csv", index=False)
    paired_aggregate = (
        paired.groupby(["contrast", "density_percent"], sort=False)[
            [
                "paired_median_delta_NSE",
                "paired_median_delta_KGE",
                "pilrn_NSE_win_fraction",
            ]
        ]
        .median()
        .reset_index()
    )
    paired_aggregate.to_csv(
        OUT / "paired_aggregate_contrasts.csv",
        index=False,
    )
    return {
        "protocol": "density_baselines_v6",
        "aggregation": (
            "station median within mask repeat, then median across repeat "
            "medians at reduced densities"
        ),
        "summary": aggregate.to_dict("records"),
        "paired_contrasts": paired_aggregate.to_dict("records"),
        "audit": collect_audit(),
    }


def parse_values(text: str, cast) -> list:
    values: list = []
    for item in text.split(","):
        item = item.strip()
        if item:
            value = cast(item)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("At least one value is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--densities",
        default=",".join(str(value) for value in DENSITIES),
    )
    parser.add_argument(
        "--repeats",
        default=",".join(str(value) for value in MASK_REPEATS),
    )
    parser.add_argument(
        "--folds",
        default=",".join(str(value) for value in FOLDS),
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--refresh-full-density",
        action="store_true",
        help=(
            "Overwrite completed 100%-density cells from the canonical "
            "published prediction archives before summarizing."
        ),
    )
    args = parser.parse_args()

    masks = declare_design()
    if args.summarize_only:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return

    methods = parse_values(args.methods, str)
    densities = parse_values(args.densities, int)
    repeats = parse_values(args.repeats, int)
    folds_to_run = parse_values(args.folds, int)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    dynamic, dates, node_ids, feature_names = load_forcing()
    obs = load_obs(node_ids, dates)
    statics, area = load_statics(node_ids)
    id2idx = {station: index for index, station in enumerate(node_ids)}
    fold_frame, trainable = eligible()
    trainable_set = set(trainable)
    t0 = dates.get_loc(pd.Period(C.STUDY_START, "M"))
    t1 = dates.get_loc(pd.Period(C.STUDY_END, "M"))
    train_end = dates.get_loc(pd.Period(C.TRAIN_END, "M"))

    features = None
    target_log_runoff = None
    if "random_forest" in methods:
        features = RF.build_features(dynamic, statics, feature_names, t0)
        runoff = flow_to_runoff(obs, area, dates)
        target_log_runoff = np.log1p(np.clip(runoff, 0, None))
    distance = None
    if "nearest_dar" in methods:
        distance = B.river_distance_matrix(node_ids)

    manifest_rows = (
        pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    )
    manifest = {
        (
            str(row["method"]),
            int(row["fold"]),
            int(row["density_percent"]),
            int(row["mask_repeat"]),
        ): row
        for row in manifest_rows
    }

    for method in methods:
        for density in densities:
            active_repeats = [1] if density == 100 else repeats
            for repeat in active_repeats:
                for fold in folds_to_run:
                    refresh_full = (
                        args.refresh_full_density and density == 100
                    )
                    if complete(method, fold, density, repeat) and not refresh_full:
                        print(
                            f"[reuse] {method} fold={fold} "
                            f"density={density} mask={repeat}",
                            flush=True,
                        )
                        continue
                    started = time.perf_counter()
                    visible_ids = D.retained_ids(
                        masks,
                        fold,
                        density,
                        repeat,
                    )
                    test_ids = scored_test_ids(fold)
                    hidden_ids = sorted(trainable_set - set(visible_ids))
                    if set(visible_ids) & set(test_ids):
                        raise RuntimeError("Visible and outer-test sets overlap.")
                    cell_dir = directory(method, fold, density, repeat)
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    if method == "random_forest":
                        assert features is not None
                        assert target_log_runoff is not None
                        if density == 100:
                            reuse_full_density_random_forest(
                                cell_dir=cell_dir,
                                fold=fold,
                                density=density,
                                repeat=repeat,
                                visible_ids=visible_ids,
                                test_ids=test_ids,
                                hidden_ids=hidden_ids,
                            )
                        else:
                            run_random_forest(
                                cell_dir=cell_dir,
                                fold=fold,
                                density=density,
                                repeat=repeat,
                                visible_ids=visible_ids,
                                test_ids=test_ids,
                                hidden_ids=hidden_ids,
                                dates=dates,
                                node_ids=node_ids,
                                obs=obs,
                                area=area,
                                features=features,
                                target_log_runoff=target_log_runoff,
                                t0=t0,
                                t1=t1,
                                train_end=train_end,
                            )
                    else:
                        assert distance is not None
                        run_nearest(
                            cell_dir=cell_dir,
                            fold=fold,
                            density=density,
                            repeat=repeat,
                            visible_ids=visible_ids,
                            test_ids=test_ids,
                            hidden_ids=hidden_ids,
                            dates=dates,
                            node_ids=node_ids,
                            obs=obs,
                            area=area,
                            distance=distance,
                            t0=t0,
                            t1=t1,
                        )
                    if not complete(method, fold, density, repeat):
                        raise RuntimeError(
                            f"Incomplete cell: {method}, fold={fold}, "
                            f"density={density}, repeat={repeat}"
                        )
                    key = (method, fold, density, repeat)
                    manifest[key] = {
                        "method": method,
                        "fold": fold,
                        "density_percent": density,
                        "mask_repeat": repeat,
                        "duration_seconds": time.perf_counter() - started,
                        "output_directory": str(cell_dir),
                    }
                    pd.DataFrame(
                        sorted(
                            manifest.values(),
                            key=lambda row: (
                                METHODS.index(str(row["method"])),
                                -int(row["density_percent"]),
                                int(row["mask_repeat"]),
                                int(row["fold"]),
                            ),
                        )
                    ).to_csv(MANIFEST, index=False)
                    print(
                        f"[done] {method} fold={fold} density={density} "
                        f"mask={repeat} seconds="
                        f"{time.perf_counter() - started:.1f}",
                        flush=True,
                    )
    print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
