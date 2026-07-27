# -*- coding: utf-8 -*-
"""Matched coupled-training comparison for the manuscript-final router.

Both coupled regimes use exactly the final Stage-2 operator: three signal
propagation rounds, learned descriptor-conditioned edge weights, a learned
gate, and no multiplicative drainage-area modulation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config_v2 as C
import train_coupled_v2 as CP
from run_stability_10x5 import DECLARED_SEEDS


FOLDS = list(range(1, 11))
REGIMES = ["scratch", "warm"]
ROUNDS = 3
OUT = C.SP_V3 / "Final_Coupled_noarea_r3_v5"
MODEL_ROOTS = {
    "scratch": C.SP_V3 / "Coupled_scratch_final_noarea_r3_v5",
    "warm": C.SP_V3 / "Coupled_warm_final_noarea_r3_v5",
}
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def directory(regime: str, seed: int, fold: int) -> Path:
    return MODEL_ROOTS[regime] / run_tag(seed, fold)


def complete(regime: str, seed: int, fold: int) -> bool:
    path = directory(regime, seed, fold)
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
        "protocol": "manuscript_final_coupled_v5",
        "folds": FOLDS,
        "seeds": DECLARED_SEEDS,
        "regimes": REGIMES,
        "routing_rounds": ROUNDS,
        "area_modulation": False,
        "learned_edge_weights": True,
        "learned_gate": True,
        "warm_start": (
            "initialize encoder/head from the seed-matched frozen Regional LSTM "
            "and reproduce the seed-matched final-router initialization"
        ),
        "scratch": "initialize the complete encoder-router system from scratch",
        "headline_aggregation": (
            "within each seed, median across the 164 scored stations; "
            "then median across the five seed-specific medians"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Final coupled-design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def execute(regime: str, seed: int, fold: int) -> None:
    tag = run_tag(seed, fold)
    CP.run(
        fold=fold,
        seed=seed,
        tag=tag,
        init_stage1_tag=tag if regime == "warm" else None,
        out_root=MODEL_ROOTS[regime],
        match_router_init=regime == "warm",
        router_rounds=ROUNDS,
        router_edge_mode="full",
        router_area_correction=False,
    )


def summarize() -> dict[str, object]:
    station_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    for regime in REGIMES:
        for seed in DECLARED_SEEDS:
            for fold in FOLDS:
                path = directory(regime, seed, fold)
                if not complete(regime, seed, fold):
                    continue
                frame = pd.read_csv(
                    path / "evaluation_holdout.csv", dtype={"station_id": str}
                )
                frame["regime"] = regime
                frame["seed"] = seed
                frame["fold"] = fold
                station_frames.append(frame)
                params = json.loads(
                    path.joinpath("norm_params.json").read_text(encoding="utf-8")
                )
                train_log = pd.read_csv(path / "train_log.csv")
                cell_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "fold": fold,
                        "n": len(frame),
                        "best_validation_loss": float(train_log.val.min()),
                        "n_param": int(params["n_param"]),
                        "NSE_full_median": float(frame.NSE_full.median()),
                        "KGE_full_median": float(frame.KGE_full.median()),
                        "PBIAS_full_median": float(frame.PBIAS_full.median()),
                        "FHV_full_median": float(frame.FHV_full.median()),
                        "negative_NSE_count": int((frame.NSE_full < 0).sum()),
                    }
                )
    if not station_frames:
        return {}

    stations = pd.concat(station_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    cells.to_csv(OUT / "cell_metrics.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    for (regime, seed), frame in stations.groupby(["regime", "seed"], sort=False):
        seed_cells = cells[(cells.regime == regime) & (cells.seed == seed)]
        seed_rows.append(
            {
                "regime": regime,
                "seed": int(seed),
                "n": len(frame),
                "folds": int(frame.fold.nunique()),
                "validation_loss_median": float(
                    seed_cells.best_validation_loss.median()
                ),
                "NSE_full_median": float(frame.NSE_full.median()),
                "NSE_test_median": float(frame.NSE_test.median()),
                "KGE_full_median": float(frame.KGE_full.median()),
                "PBIAS_full_median": float(frame.PBIAS_full.median()),
                "FHV_full_median": float(frame.FHV_full.median()),
                "negative_NSE_count": int((frame.NSE_full < 0).sum()),
                "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
                "NSE_above_0_5_fraction": float((frame.NSE_full > 0.5).mean()),
                "n_param": int(seed_cells.n_param.median()),
            }
        )
    seeds = pd.DataFrame(seed_rows).sort_values(["regime", "seed"])
    seeds.to_csv(OUT / "seed_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for regime, frame in seeds.groupby("regime", sort=False):
        aggregate_rows.append(
            {
                "regime": regime,
                "seeds_complete": int(frame.seed.nunique()),
                "min_folds_per_seed": int(frame.folds.min()),
                "scored_stations_per_seed": int(frame.n.min()),
                "validation_loss_median": float(
                    frame.validation_loss_median.median()
                ),
                "NSE_full_median": float(frame.NSE_full_median.median()),
                "NSE_test_median": float(frame.NSE_test_median.median()),
                "KGE_full_median": float(frame.KGE_full_median.median()),
                "PBIAS_full_median": float(frame.PBIAS_full_median.median()),
                "FHV_full_median": float(frame.FHV_full_median.median()),
                "negative_NSE_count_median": float(
                    frame.negative_NSE_count.median()
                ),
                "negative_NSE_fraction_median": float(
                    frame.negative_NSE_fraction.median()
                ),
                "NSE_above_0_5_fraction_median": float(
                    frame.NSE_above_0_5_fraction.median()
                ),
                "n_param": int(frame.n_param.median()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "aggregate_summary.csv", index=False)
    return {
        "protocol": "manuscript_final_coupled_v5",
        "cells_complete": int(len(cells)),
        "regimes_complete": int(aggregate.regime.nunique()),
        "aggregation": "median of five seed-specific station medians",
        "summary": aggregate.to_dict("records"),
    }


def parse_set(text: str, cast) -> list:
    values = []
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
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DECLARED_SEEDS)
    )
    parser.add_argument(
        "--folds", default=",".join(str(value) for value in FOLDS)
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    declare_design()
    if args.summarize_only:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)
        return

    regimes = parse_set(args.regimes, str)
    unknown = sorted(set(regimes) - set(REGIMES))
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}")
    seeds = parse_set(args.seeds, int)
    folds = parse_set(args.folds, int)
    rows = (
        pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    )
    by_key = {
        (str(row["regime"]), int(row["seed"]), int(row["fold"])): row
        for row in rows
    }
    for regime in regimes:
        for seed in seeds:
            for fold in folds:
                started = time.perf_counter()
                status = "reused"
                if args.rerun or not complete(regime, seed, fold):
                    execute(regime, seed, fold)
                    status = "completed"
                if not complete(regime, seed, fold):
                    raise RuntimeError(
                        f"Incomplete final coupled model: "
                        f"regime={regime}, seed={seed}, fold={fold}"
                    )
                by_key[(regime, seed, fold)] = {
                    "regime": regime,
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "duration_seconds": time.perf_counter() - started,
                    "output_directory": str(directory(regime, seed, fold)),
                }
                pd.DataFrame(
                    sorted(
                        by_key.values(),
                        key=lambda row: (
                            REGIMES.index(str(row["regime"])),
                            int(row["seed"]),
                            int(row["fold"]),
                        ),
                    )
                ).to_csv(MANIFEST, index=False)
    print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
