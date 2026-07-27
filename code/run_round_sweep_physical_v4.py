# -*- coding: utf-8 -*-
"""Leakage-safe routing-depth selection for the physical-v4 signal router.

The round count is selected from the inner validation loss. Outer-fold metrics
are archived only as sensitivity evidence and are not used by the selection
rule. The smallest round count within one standard error of the best mean
validation loss is preferred.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
import train_pilrn_v2 as P
from run_stability_10x5 import DECLARED_SEEDS


FOLDS = list(range(1, 11))
INITIAL_ROUNDS = list(range(1, 9))
OUT = C.SP_V3 / "Round_Sweep_physical_v4"
MODEL_ROOT = C.SP_V3 / "PILRN_sig_round_sweep_physical_v4"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"

# Original range check over all 174 fold-assigned, non-excluded candidates for
# seed 42. Ten of these candidates have insufficient observations and do not
# enter the 164-station scored result. The distinction is retained here so the
# pre-run design record remains auditable.
HOP_COUNTS = {1: 113, 2: 41, 3: 12, 4: 6, 5: 1, 6: 1}
SCORED_HOP_COUNTS_SEED42 = {1: 107, 2: 38, 3: 11, 4: 6, 5: 1, 6: 1}
SCORED_HOP_COUNTS_ALL_SEEDS = {1: 528, 2: 200, 3: 59, 4: 23, 5: 5, 6: 5}
MAX_REQUIRED_HOPS = 6


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def parse_int_set(text: str) -> list[int]:
    values: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            values.update(range(int(left), int(right) + 1))
        else:
            values.add(int(chunk))
    result = sorted(values)
    if not result or result[0] < 1:
        raise ValueError(f"Rounds must be positive integers: {text}")
    return result


def generic_round_root(rounds: int) -> Path:
    return MODEL_ROOT / f"round_{rounds:02d}"


def directory(rounds: int, seed: int, fold: int) -> Path:
    tag = run_tag(seed, fold)
    if rounds == 2:
        return P.VARIANT_OUT["pilrn_sig"] / tag
    if seed == C.SEED and rounds == 1:
        return P.VARIANT_OUT["pilrn_sig_r1"] / tag
    if seed == C.SEED and rounds == 3:
        return P.VARIANT_OUT["pilrn_sig_r3"] / tag
    return generic_round_root(rounds) / tag


def complete(rounds: int, seed: int, fold: int) -> bool:
    path = directory(rounds, seed, fold)
    return all(
        (path / name).exists()
        for name in (
            "checkpoint.pth",
            "norm_params.json",
            "train_log.csv",
            "evaluation_holdout.csv",
        )
    )


def declare_design() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "physical_v4_round_sweep",
        "folds": FOLDS,
        "seeds": DECLARED_SEEDS,
        "initial_rounds": INITIAL_ROUNDS,
        "nearest_donor_hop_counts": {
            str(key): value for key, value in HOP_COUNTS.items()
        },
        "nearest_donor_hop_counts_population": (
            "seed-42 range check over 174 fold-assigned non-excluded "
            "candidates; includes 10 stations later omitted from scoring for "
            "insufficient records"
        ),
        "scored_station_hop_counts_seed42": {
            str(key): value for key, value in SCORED_HOP_COUNTS_SEED42.items()
        },
        "scored_station_hop_counts_all_five_seeds": {
            str(key): value
            for key, value in SCORED_HOP_COUNTS_ALL_SEEDS.items()
        },
        "scored_station_hop_counts_population": (
            "820 target-seed cells = 164 scored stations x 5 seeds; validation "
            "donors are seed-specific"
        ),
        "distance_count_correction": (
            "The original range check correctly established a six-hop maximum "
            "but its 174 candidates must not be described as 164 scored "
            "stations. The corrected scored distributions were added after "
            "the completed scan; the round range and selection rules are "
            "unchanged."
        ),
        "maximum_nearest_donor_hops": MAX_REQUIRED_HOPS,
        "initial_scan_rationale": (
            "Rounds 1-6 span the complete observed nearest-donor distance; "
            "rounds 7-8 test repeated fusion after reachability saturates."
        ),
        "extension_rule": (
            "If rounds 7-8 do not both worsen mean inner-validation loss "
            "relative to the best earlier round, extend in two-round blocks "
            "to 10 and at most 12."
        ),
        "selection_metric": (
            "mean across all seed-fold cells of the minimum inner-validation "
            "standardized-log runoff MSE"
        ),
        "selection_rule": (
            "choose the smallest round count whose mean validation loss is "
            "within one standard error of the raw best round"
        ),
        "outer_test_rule": (
            "outer-fold NSE/KGE/PBIAS/FHV are descriptive sensitivity results "
            "and are not used to choose the round count"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Round-sweep design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def best_validation_loss(path: Path) -> float:
    params_path = path / "norm_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        value = params.get("best_validation_loss")
        if value is not None and np.isfinite(float(value)):
            return float(value)
    log = pd.read_csv(path / "train_log.csv")
    return float(pd.to_numeric(log["val"], errors="coerce").min())


def summarize(rounds_to_include: list[int] | None = None) -> dict[str, object]:
    if rounds_to_include is None:
        if not MANIFEST.exists():
            raise FileNotFoundError(MANIFEST)
        rounds_to_include = sorted(
            pd.read_csv(MANIFEST)["rounds"].astype(int).unique().tolist()
        )

    station_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    for rounds in rounds_to_include:
        for seed in DECLARED_SEEDS:
            for fold in FOLDS:
                path = directory(rounds, seed, fold)
                if not complete(rounds, seed, fold):
                    continue
                frame = pd.read_csv(
                    path / "evaluation_holdout.csv", dtype={"station_id": str}
                )
                frame["rounds"] = rounds
                frame["seed"] = seed
                frame["fold"] = fold
                station_frames.append(frame)
                cell_rows.append(
                    {
                        "rounds": rounds,
                        "seed": seed,
                        "fold": fold,
                        "n": len(frame),
                        "best_validation_loss": best_validation_loss(path),
                        "NSE_full_median": float(frame.NSE_full.median()),
                        "KGE_full_median": float(frame.KGE_full.median()),
                        "PBIAS_full_median": float(frame.PBIAS_full.median()),
                        "FHV_full_median": float(frame.FHV_full.median()),
                        "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
                        "output_directory": str(path),
                    }
                )
    if not station_frames:
        return {}

    stations = pd.concat(station_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    cells.to_csv(OUT / "cell_metrics.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    for (rounds, seed), frame in stations.groupby(["rounds", "seed"], sort=True):
        matching = cells[(cells.rounds == rounds) & (cells.seed == seed)]
        seed_rows.append(
            {
                "rounds": int(rounds),
                "seed": int(seed),
                "n": len(frame),
                "folds": int(frame.fold.nunique()),
                "validation_loss_mean": float(
                    matching.best_validation_loss.mean()
                ),
                "NSE_full_median": float(frame.NSE_full.median()),
                "KGE_full_median": float(frame.KGE_full.median()),
                "PBIAS_full_median": float(frame.PBIAS_full.median()),
                "FHV_full_median": float(frame.FHV_full.median()),
                "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
            }
        )
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "seed_summary.csv", index=False)

    round_rows: list[dict[str, object]] = []
    for rounds, frame in cells.groupby("rounds", sort=True):
        seed_frame = seeds[seeds.rounds == rounds]
        val_sd = float(frame.best_validation_loss.std(ddof=1))
        round_rows.append(
            {
                "rounds": int(rounds),
                "cells": len(frame),
                "seeds_complete": int(seed_frame.seed.nunique()),
                "min_folds_per_seed": int(seed_frame.folds.min()),
                "validation_loss_mean": float(
                    frame.best_validation_loss.mean()
                ),
                "validation_loss_se": (
                    val_sd / math.sqrt(len(frame)) if len(frame) > 1 else np.nan
                ),
                "validation_loss_median": float(
                    frame.best_validation_loss.median()
                ),
                "NSE_full_seed_median_mean": float(
                    seed_frame.NSE_full_median.mean()
                ),
                "NSE_full_seed_median_sd": float(
                    seed_frame.NSE_full_median.std(ddof=1)
                ),
                "KGE_full_seed_median_mean": float(
                    seed_frame.KGE_full_median.mean()
                ),
                "negative_NSE_fraction_mean": float(
                    seed_frame.negative_NSE_fraction.mean()
                ),
                "PBIAS_full_seed_median_mean": float(
                    seed_frame.PBIAS_full_median.mean()
                ),
                "FHV_full_seed_median_mean": float(
                    seed_frame.FHV_full_median.mean()
                ),
            }
        )
    summary = pd.DataFrame(round_rows).sort_values("rounds")
    summary.to_csv(OUT / "round_summary.csv", index=False)

    complete_summary = summary[
        (summary.cells == len(DECLARED_SEEDS) * len(FOLDS))
        & (summary.seeds_complete == len(DECLARED_SEEDS))
        & (summary.min_folds_per_seed == len(FOLDS))
    ].copy()
    if complete_summary.empty:
        return {}
    raw = complete_summary.loc[
        complete_summary.validation_loss_mean.idxmin()
    ]
    threshold = float(raw.validation_loss_mean + raw.validation_loss_se)
    eligible = complete_summary[
        complete_summary.validation_loss_mean <= threshold
    ]
    selected = int(eligible.rounds.min())
    selection = {
        "raw_best_round": int(raw.rounds),
        "raw_best_validation_loss_mean": float(raw.validation_loss_mean),
        "raw_best_validation_loss_se": float(raw.validation_loss_se),
        "one_standard_error_threshold": threshold,
        "selected_round": selected,
        "selection_rule": "smallest round within one SE of raw best",
        "rounds_compared": complete_summary.rounds.astype(int).tolist(),
    }
    (OUT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def execute(rounds: int, seed: int, fold: int) -> None:
    tag = run_tag(seed, fold)
    if rounds == 2:
        output_root = P.VARIANT_OUT["pilrn_sig"]
    elif seed == C.SEED and rounds == 1:
        output_root = P.VARIANT_OUT["pilrn_sig_r1"]
    elif seed == C.SEED and rounds == 3:
        output_root = P.VARIANT_OUT["pilrn_sig_r3"]
    else:
        output_root = generic_round_root(rounds)
    P.run(
        fold=fold,
        seed=seed,
        stage1_tag=tag,
        tag=tag,
        variant="pilrn_sig",
        output_root=output_root,
        rounds_override=rounds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="1-8")
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DECLARED_SEEDS)
    )
    parser.add_argument(
        "--folds", default=",".join(str(value) for value in FOLDS)
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--manifest-name", default=MANIFEST.name)
    parser.add_argument("--defer-summary", action="store_true")
    args = parser.parse_args()
    rounds_values = parse_int_set(args.rounds)
    seeds = parse_int_set(args.seeds)
    folds = parse_int_set(args.folds)
    declare_design()
    if args.summarize_only:
        selection = summarize()
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
        return

    manifest_name = Path(args.manifest_name).name
    if manifest_name != args.manifest_name or not manifest_name.endswith(".csv"):
        raise ValueError(f"Invalid manifest name: {args.manifest_name}")
    manifest_path = OUT / manifest_name
    rows = (
        pd.read_csv(manifest_path).to_dict("records")
        if manifest_path.exists()
        else []
    )
    by_key = {
        (int(row["rounds"]), int(row["seed"]), int(row["fold"])): row
        for row in rows
    }
    for rounds in rounds_values:
        for seed in seeds:
            for fold in folds:
                started = time.perf_counter()
                status = "reused"
                if args.rerun or not complete(rounds, seed, fold):
                    execute(rounds, seed, fold)
                    status = "completed"
                if not complete(rounds, seed, fold):
                    raise RuntimeError(
                        f"Incomplete rounds={rounds}, seed={seed}, fold={fold}"
                    )
                key = (rounds, seed, fold)
                by_key[key] = {
                    "rounds": rounds,
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "duration_seconds": time.perf_counter() - started,
                    "output_directory": str(directory(rounds, seed, fold)),
                }
                pd.DataFrame(
                    sorted(
                        by_key.values(),
                        key=lambda row: (
                            int(row["rounds"]),
                            DECLARED_SEEDS.index(int(row["seed"])),
                            int(row["fold"]),
                        ),
                    )
                ).to_csv(manifest_path, index=False)
        if not args.defer_summary:
            selection = summarize()
            print(
                f"[round={rounds}] selection={selection}",
                flush=True,
            )
    if not args.defer_summary:
        selection = summarize()
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
