# -*- coding: utf-8 -*-
"""Section 5.5 end-to-end sensitivity to training-gauge density.

Outer-fold test stations remain fixed.  Only outer-training gauges are thinned.
The 25%, 50%, and 100% masks are nested.  At each non-full density the Regional
LSTM is retrained and Stage 2 receives observations/losses only from retained
gauges.  Removed gauges remain graph nodes with forcing, attributes, topology,
and a density-specific regional prior.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
import train_pilrn_v2 as P
import train_regional_lstm as R


FOLDS = list(range(1, 11))
DENSITIES = [100, 50, 25]
MASK_REPEATS = [1, 2, 3]
MODEL_SEED = 42
METHODS = ["regional", "pilrn_sig", "pilrn_sig_topology"]

OUT = C.SP_V3 / "Density_physical_v4"
REGIONAL_ROOT = C.SP_V3 / "Regional_LSTM_density_physical_v4"
PILRN_ROOT = C.SP_V3 / "PILRN_sig_density_physical_v4"
TOPOLOGY_ROOT = C.SP_V3 / "PILRN_topology_density_physical_v4"
DESIGN = OUT / "experiment_design.json"
MASKS = OUT / "nested_spatial_masks.csv"
MANIFEST = OUT / "run_manifest.csv"


def run_tag(fold: int, density: int, repeat: int) -> str:
    return f"fold_{fold:02d}_d{density:03d}_mask{repeat:02d}"


def method_root(method: str) -> Path:
    return {
        "regional": REGIONAL_ROOT,
        "pilrn_sig": PILRN_ROOT,
        "pilrn_sig_topology": TOPOLOGY_ROOT,
    }[method]


def full_density_directory(method: str, fold: int) -> Path:
    tag = f"fold_{fold:02d}"
    return {
        "regional": C.REGIONAL_OUT / tag,
        "pilrn_sig": P.VARIANT_OUT["pilrn_sig"] / tag,
        "pilrn_sig_topology": P.VARIANT_OUT["pilrn_sig_topology"] / tag,
    }[method]


def directory(method: str, fold: int, density: int, repeat: int) -> Path:
    if density == 100:
        return full_density_directory(method, fold)
    return method_root(method) / run_tag(fold, density, repeat)


def complete(method: str, fold: int, density: int, repeat: int) -> bool:
    path = directory(method, fold, density, repeat)
    return all(
        (path / name).exists()
        for name in ("checkpoint.pth", "norm_params.json", "evaluation_holdout.csv")
    )


def eligible_stations() -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str})
    excluded = set()
    path = C.BASIN / "station_exclude.csv"
    if path.exists():
        excluded = set(pd.read_csv(path, dtype=str).iloc[:, 0])
    folds = folds[
        folds.station_id.isin(set(nodes.node_id)) & ~folds.station_id.isin(excluded)
    ].copy()
    coordinates = nodes.set_index("node_id")[["lat", "lon"]]
    return folds, coordinates


def spatially_balanced_order(
    station_ids: list[str],
    coordinates: pd.DataFrame,
    random_seed: int,
) -> list[str]:
    """Interleave randomized 4x4 geographic strata to obtain a balanced prefix."""
    frame = coordinates.reindex(station_ids).copy()
    frame["station_id"] = frame.index.astype(str)
    frame["lat_bin"] = pd.qcut(
        frame["lat"].rank(method="first"), q=4, labels=False, duplicates="drop"
    )
    frame["lon_bin"] = pd.qcut(
        frame["lon"].rank(method="first"), q=4, labels=False, duplicates="drop"
    )
    rng = np.random.default_rng(random_seed)
    groups: dict[tuple[int, int], list[str]] = {}
    for key, group in frame.groupby(["lat_bin", "lon_bin"], sort=True):
        values = group.station_id.to_numpy(dtype=str)
        groups[(int(key[0]), int(key[1]))] = list(rng.permutation(values))

    keys = list(groups)
    result: list[str] = []
    level = 0
    while len(result) < len(station_ids):
        active = [key for key in keys if level < len(groups[key])]
        for position in rng.permutation(len(active)):
            key = active[int(position)]
            result.append(groups[key][level])
        level += 1
    if len(result) != len(set(result)) or set(result) != set(station_ids):
        raise RuntimeError("Spatial ordering is not a one-to-one station permutation.")
    return result


def build_masks() -> pd.DataFrame:
    folds, coordinates = eligible_stations()
    rows = []
    for fold in FOLDS:
        outer_training = folds.loc[folds.fold != fold, "station_id"].tolist()
        for repeat in MASK_REPEATS:
            order = spatially_balanced_order(
                outer_training,
                coordinates,
                random_seed=55000 + fold * 100 + repeat,
            )
            sizes = {
                100: len(order),
                50: max(5, int(round(0.50 * len(order)))),
                25: max(5, int(round(0.25 * len(order)))),
            }
            for density in DENSITIES:
                selected = set(order[: sizes[density]])
                for rank, station_id in enumerate(order, start=1):
                    rows.append(
                        {
                            "fold": fold,
                            "mask_repeat": repeat,
                            "density_percent": density,
                            "station_id": station_id,
                            "selected": station_id in selected,
                            "spatial_order_rank": rank,
                        }
                    )
    return pd.DataFrame(rows)


def declare_design() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "density_physical_v4",
        "folds": FOLDS,
        "densities_percent": DENSITIES,
        "mask_repeats": MASK_REPEATS,
        "model_seed": MODEL_SEED,
        "methods": METHODS,
        "outer_test_rule": "fixed original outer fold at every density",
        "thinning_rule": (
            "nested prefixes of a repeated, randomized, 4x4 geographic-stratum "
            "interleaving of outer-training gauges"
        ),
        "removed_gauge_role": (
            "graph node with forcing/static/topology and regional prediction; "
            "no observation, donor role, loss, validation, or early stopping"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Declared density design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    generated = build_masks()
    if MASKS.exists():
        existing_masks = pd.read_csv(MASKS, dtype={"station_id": str})
        pd.testing.assert_frame_equal(existing_masks, generated, check_dtype=False)
    else:
        generated.to_csv(MASKS, index=False)
    return generated


def retained_ids(
    masks: pd.DataFrame, fold: int, density: int, repeat: int
) -> list[str]:
    subset = masks[
        (masks.fold == fold)
        & (masks.mask_repeat == repeat)
        & (masks.density_percent == density)
        & masks.selected
    ]
    return subset.station_id.astype(str).tolist()


def execute(
    method: str,
    fold: int,
    density: int,
    repeat: int,
    visible: list[str],
) -> None:
    tag = run_tag(fold, density, repeat)
    if method == "regional":
        R.run(
            fold=fold,
            seed=MODEL_SEED,
            tag=tag,
            visible_ids=visible,
            output_root=REGIONAL_ROOT,
        )
    elif method in {"pilrn_sig", "pilrn_sig_topology"}:
        P.run(
            fold=fold,
            seed=MODEL_SEED,
            stage1_tag=tag,
            tag=tag,
            variant=method,
            visible_ids=visible,
            stage1_root=REGIONAL_ROOT,
            output_root=method_root(method),
        )
    else:
        raise ValueError(method)


def summarize() -> None:
    frames = []
    beta_rows = []
    for method in METHODS:
        for density in DENSITIES:
            repeats = [1] if density == 100 else MASK_REPEATS
            for repeat in repeats:
                for fold in FOLDS:
                    path = directory(method, fold, density, repeat)
                    evaluation = path / "evaluation_holdout.csv"
                    if not evaluation.exists():
                        continue
                    frame = pd.read_csv(evaluation, dtype={"station_id": str})
                    frame["method"] = method
                    frame["density_percent"] = density
                    frame["mask_repeat"] = repeat
                    frame["fold"] = fold
                    frames.append(frame)
                    norm = path / "norm_params.json"
                    if norm.exists():
                        payload = json.loads(norm.read_text(encoding="utf-8"))
                        if payload.get("learned_beta") is not None:
                            beta_rows.append(
                                {
                                    "method": method,
                                    "density_percent": density,
                                    "mask_repeat": repeat,
                                    "fold": fold,
                                    "learned_beta": payload["learned_beta"],
                                }
                            )
    if not frames:
        return
    stations = pd.concat(frames, ignore_index=True)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    if beta_rows:
        pd.DataFrame(beta_rows).to_csv(OUT / "learned_beta.csv", index=False)

    metrics = ["NSE_full", "NSE_test", "KGE_full", "PBIAS_full", "FHV_full"]
    summary_rows = []
    for (method, density, repeat), frame in stations.groupby(
        ["method", "density_percent", "mask_repeat"], sort=True
    ):
        row = {
            "method": method,
            "density_percent": int(density),
            "mask_repeat": int(repeat),
            "n": int(len(frame)),
            "folds": int(frame.fold.nunique()),
            "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
        }
        for metric in metrics:
            row[f"{metric}_median"] = float(frame[metric].median())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "density_repeat_summary.csv", index=False)

    contrast_rows = []
    full = stations[stations.method == "pilrn_sig"]
    for comparator in ["regional", "pilrn_sig_topology"]:
        other = stations[stations.method == comparator]
        for density in DENSITIES:
            repeats = [1] if density == 100 else MASK_REPEATS
            for repeat in repeats:
                keys = ["station_id", "fold", "density_percent", "mask_repeat"]
                left = full[
                    (full.density_percent == density) & (full.mask_repeat == repeat)
                ][keys + ["NSE_full"]].rename(columns={"NSE_full": "NSE_full_model"})
                right = other[
                    (other.density_percent == density) & (other.mask_repeat == repeat)
                ][keys + ["NSE_full"]].rename(columns={"NSE_full": "NSE_comparator"})
                joined = left.merge(right, on=keys, how="inner")
                delta = joined.NSE_full_model - joined.NSE_comparator
                contrast_rows.append(
                    {
                        "contrast": f"pilrn_sig-minus-{comparator}",
                        "density_percent": density,
                        "mask_repeat": repeat,
                        "n": int(len(joined)),
                        "paired_median_delta_NSE_full": (
                            float(delta.median()) if len(delta) else np.nan
                        ),
                        "full_model_win_fraction": (
                            float((delta > 0).mean()) if len(delta) else np.nan
                        ),
                    }
                )
    pd.DataFrame(contrast_rows).to_csv(OUT / "paired_density_contrasts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--densities", default=",".join(str(value) for value in DENSITIES)
    )
    parser.add_argument(
        "--repeats", default=",".join(str(value) for value in MASK_REPEATS)
    )
    parser.add_argument(
        "--folds", default=",".join(str(value) for value in FOLDS)
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    densities = [int(value) for value in args.densities.split(",") if value]
    repeats = [int(value) for value in args.repeats.split(",") if value]
    folds = [int(value) for value in args.folds.split(",") if value]
    masks = declare_design()
    if args.summarize_only:
        summarize()
        return

    rows = pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    by_key = {
        (
            str(row["method"]),
            int(row["fold"]),
            int(row["density_percent"]),
            int(row["mask_repeat"]),
        ): row
        for row in rows
    }
    for density in densities:
        active_repeats = [1] if density == 100 else repeats
        for repeat in active_repeats:
            for fold in folds:
                visible = retained_ids(masks, fold, density, repeat)
                for method in methods:
                    started = time.perf_counter()
                    status = "reused"
                    if args.rerun or not complete(method, fold, density, repeat):
                        if density == 100:
                            raise RuntimeError(
                                "Run the full-density stability/ablation experiments "
                                f"before density analysis: {method}, fold={fold}"
                            )
                        execute(method, fold, density, repeat, visible)
                        status = "completed"
                    if not complete(method, fold, density, repeat):
                        raise RuntimeError(
                            f"Incomplete {method}, density={density}, "
                            f"repeat={repeat}, fold={fold}"
                        )
                    key = (method, fold, density, repeat)
                    by_key[key] = {
                        "method": method,
                        "fold": fold,
                        "density_percent": density,
                        "mask_repeat": repeat,
                        "n_training_gauges": len(visible),
                        "status": status,
                        "duration_seconds": time.perf_counter() - started,
                        "output_directory": str(
                            directory(method, fold, density, repeat)
                        ),
                    }
                    pd.DataFrame(
                        sorted(
                            by_key.values(),
                            key=lambda row: (
                                -int(row["density_percent"]),
                                int(row["mask_repeat"]),
                                int(row["fold"]),
                                METHODS.index(str(row["method"])),
                            ),
                        )
                    ).to_csv(MANIFEST, index=False)
            summarize()
    summarize()


if __name__ == "__main__":
    main()
