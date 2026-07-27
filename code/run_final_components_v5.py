# -*- coding: utf-8 -*-
"""Matched three-round component comparison for the final manuscript model.

The full model is reused from the completed round sweep, the no-area model is
reused from ``run_final_pilrn_v5.py``, and only the topology-only counterpart is
newly trained.  All three use the same five seeds and ten outer folds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config_v2 as C
import run_final_pilrn_v5 as F
import run_round_sweep_physical_v4 as R
import train_pilrn_v2 as P
from run_stability_10x5 import DECLARED_SEEDS


FOLDS = list(range(1, 11))
ROUNDS = 3
COMPONENTS = ["full", "no_area", "topology_only"]
OUT = C.SP_V3 / "Final_Components_r3_v5"
TOPOLOGY_ROOT = C.SP_V3 / "PILRN_topology_final_r3_v5"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def directory(component: str, seed: int, fold: int) -> Path:
    if component == "full":
        return R.directory(ROUNDS, seed, fold)
    if component == "no_area":
        return F.directory(seed, fold)
    return TOPOLOGY_ROOT / run_tag(seed, fold)


def complete(component: str, seed: int, fold: int) -> bool:
    path = directory(component, seed, fold)
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
        "protocol": "manuscript_final_components_v5",
        "folds": FOLDS,
        "seeds": DECLARED_SEEDS,
        "routing_rounds": ROUNDS,
        "components": {
            "full": "learned edge weights, learned area modulation, learned gate",
            "no_area": "learned edge weights and learned gate; no area modulation",
            "topology_only": "uniform edge weights and learned gate; no area modulation",
        },
        "headline_aggregation": (
            "within each seed, median across the 164 scored stations; "
            "then median across the five seed-specific medians"
        ),
        "purpose": (
            "verify the final removal of area modulation and quantify whether "
            "descriptor-conditioned edge weights add value beyond topology alone"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Final component-design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def execute(component: str, seed: int, fold: int) -> None:
    if component == "full":
        if not R.complete(ROUNDS, seed, fold):
            R.execute(ROUNDS, seed, fold)
        return
    if component == "no_area":
        if not F.complete(seed, fold):
            F.execute(seed, fold)
        return
    tag = run_tag(seed, fold)
    P.run(
        fold=fold,
        seed=seed,
        stage1_tag=tag,
        tag=tag,
        variant="pilrn_sig_topology",
        output_root=TOPOLOGY_ROOT,
        rounds_override=ROUNDS,
    )


def summarize() -> dict[str, object]:
    station_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        for seed in DECLARED_SEEDS:
            for fold in FOLDS:
                path = directory(component, seed, fold)
                if not complete(component, seed, fold):
                    continue
                frame = pd.read_csv(
                    path / "evaluation_holdout.csv", dtype={"station_id": str}
                )
                frame["component"] = component
                frame["seed"] = seed
                frame["fold"] = fold
                station_frames.append(frame)
                params = json.loads(
                    path.joinpath("norm_params.json").read_text(encoding="utf-8")
                )
                best_validation_loss = params.get("best_validation_loss")
                if best_validation_loss is None:
                    train_log = pd.read_csv(path / "train_log.csv")
                    best_validation_loss = float(
                        pd.to_numeric(train_log["val"], errors="coerce").min()
                    )
                cell_rows.append(
                    {
                        "component": component,
                        "seed": seed,
                        "fold": fold,
                        "n": len(frame),
                        "best_validation_loss": float(best_validation_loss),
                        "n_param": int(params["n_param"]),
                    }
                )
    if not station_frames:
        return {}

    stations = pd.concat(station_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    cells.to_csv(OUT / "cell_metrics.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    for (component, seed), frame in stations.groupby(
        ["component", "seed"], sort=False
    ):
        seed_cells = cells[
            (cells.component == component) & (cells.seed == seed)
        ]
        seed_rows.append(
            {
                "component": component,
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
    seeds = pd.DataFrame(seed_rows).sort_values(["component", "seed"])
    seeds.to_csv(OUT / "seed_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for component, frame in seeds.groupby("component", sort=False):
        aggregate_rows.append(
            {
                "component": component,
                "seeds_complete": int(frame.seed.nunique()),
                "min_folds_per_seed": int(frame.folds.min()),
                "NSE_full_median": float(frame.NSE_full_median.median()),
                "NSE_test_median": float(frame.NSE_test_median.median()),
                "KGE_full_median": float(frame.KGE_full_median.median()),
                "PBIAS_full_median": float(frame.PBIAS_full_median.median()),
                "FHV_full_median": float(frame.FHV_full_median.median()),
                "negative_NSE_count_median": float(
                    frame.negative_NSE_count.median()
                ),
                "NSE_above_0_5_fraction_median": float(
                    frame.NSE_above_0_5_fraction.median()
                ),
                "validation_loss_median": float(
                    frame.validation_loss_median.median()
                ),
                "n_param": int(frame.n_param.median()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "aggregate_summary.csv", index=False)

    reference = stations[stations.component == "no_area"][
        ["station_id", "seed", "fold", "NSE_full", "KGE_full"]
    ].rename(
        columns={
            "NSE_full": "NSE_reference",
            "KGE_full": "KGE_reference",
        }
    )
    paired_rows: list[dict[str, object]] = []
    for component in ["full", "topology_only"]:
        other = stations[stations.component == component][
            ["station_id", "seed", "fold", "NSE_full", "KGE_full"]
        ].rename(
            columns={"NSE_full": "NSE_other", "KGE_full": "KGE_other"}
        )
        joined = reference.merge(
            other, on=["station_id", "seed", "fold"], how="inner"
        )
        joined["delta_NSE"] = joined.NSE_reference - joined.NSE_other
        joined["delta_KGE"] = joined.KGE_reference - joined.KGE_other
        for seed, frame in joined.groupby("seed", sort=False):
            paired_rows.append(
                {
                    "contrast": f"no_area-minus-{component}",
                    "seed": int(seed),
                    "n": len(frame),
                    "paired_median_delta_NSE": float(frame.delta_NSE.median()),
                    "paired_median_delta_KGE": float(frame.delta_KGE.median()),
                    "reference_NSE_win_fraction": float(
                        (frame.delta_NSE > 0).mean()
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_seed_contrasts.csv", index=False)
    paired_aggregate = (
        paired.groupby("contrast", sort=False)[
            [
                "paired_median_delta_NSE",
                "paired_median_delta_KGE",
                "reference_NSE_win_fraction",
            ]
        ]
        .median()
        .reset_index()
    )
    paired_aggregate.to_csv(OUT / "paired_aggregate_contrasts.csv", index=False)
    return {
        "protocol": "manuscript_final_components_v5",
        "cells_complete": int(len(cells)),
        "components_complete": int(aggregate.component.nunique()),
        "aggregation": "median of five seed-specific station medians",
        "summary": aggregate.to_dict("records"),
        "paired_contrasts": paired_aggregate.to_dict("records"),
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
    parser.add_argument("--components", default=",".join(COMPONENTS))
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

    components = parse_set(args.components, str)
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown components: {unknown}")
    seeds = parse_set(args.seeds, int)
    folds = parse_set(args.folds, int)
    rows = (
        pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    )
    by_key = {
        (str(row["component"]), int(row["seed"]), int(row["fold"])): row
        for row in rows
    }
    for component in components:
        for seed in seeds:
            for fold in folds:
                started = time.perf_counter()
                status = "reused"
                if args.rerun or not complete(component, seed, fold):
                    execute(component, seed, fold)
                    status = "completed"
                if not complete(component, seed, fold):
                    raise RuntimeError(
                        f"Incomplete final component: "
                        f"component={component}, seed={seed}, fold={fold}"
                    )
                by_key[(component, seed, fold)] = {
                    "component": component,
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "duration_seconds": time.perf_counter() - started,
                    "output_directory": str(directory(component, seed, fold)),
                }
                pd.DataFrame(
                    sorted(
                        by_key.values(),
                        key=lambda row: (
                            COMPONENTS.index(str(row["component"])),
                            int(row["seed"]),
                            int(row["fold"]),
                        ),
                    )
                ).to_csv(MANIFEST, index=False)
    print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
