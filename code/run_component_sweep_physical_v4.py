# -*- coding: utf-8 -*-
"""Multiseed component ablation at the validation-selected routing depth.

The routing depth is read from the completed physical-v4 round sweep.  Model
components are then compared over the same 10 outer folds and five declared
seeds.  Inner-validation loss selects the deployable configuration; outer-test
metrics are archived only as paired descriptive evidence.
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
import run_round_sweep_physical_v4 as R
import train_pilrn_v2 as P
from run_stability_10x5 import DECLARED_SEEDS


FOLDS = list(range(1, 11))
COMPONENTS = ["full", "no_area", "topology_only"]
VARIANTS = {
    "full": "pilrn_sig",
    "no_area": "pilrn_sig_no_area",
    "topology_only": "pilrn_sig_topology",
}
# Lower is simpler and wins a one-standard-error tie.
COMPLEXITY_RANK = {"topology_only": 0, "no_area": 1, "full": 2}
OUT = C.SP_V3 / "Component_Sweep_physical_v4"
MODEL_ROOT = C.SP_V3 / "PILRN_component_sweep_physical_v4"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def selected_round() -> int:
    if not R.OUT.joinpath("selection.json").exists():
        raise FileNotFoundError(
            f"Complete and summarize the routing-depth sweep first: {R.OUT}"
        )
    payload = json.loads(
        R.OUT.joinpath("selection.json").read_text(encoding="utf-8")
    )
    compared = [int(value) for value in payload.get("rounds_compared", [])]
    if not all(value in compared for value in R.INITIAL_ROUNDS):
        raise RuntimeError(
            "Routing-depth selection is provisional; rounds 1-8 are not all complete."
        )
    return int(payload["selected_round"])


def output_root(component: str, rounds: int) -> Path:
    return MODEL_ROOT / component / f"round_{rounds:02d}"


def directory(component: str, rounds: int, seed: int, fold: int) -> Path:
    if component == "full":
        return R.directory(rounds, seed, fold)
    return output_root(component, rounds) / run_tag(seed, fold)


def complete(component: str, rounds: int, seed: int, fold: int) -> bool:
    path = directory(component, rounds, seed, fold)
    return all(
        (path / name).exists()
        for name in (
            "checkpoint.pth",
            "norm_params.json",
            "train_log.csv",
            "evaluation_holdout.csv",
        )
    )


def declare_design(rounds: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    round_selection = json.loads(
        R.OUT.joinpath("selection.json").read_text(encoding="utf-8")
    )
    payload = {
        "protocol": "physical_v4_component_sweep",
        "folds": FOLDS,
        "seeds": DECLARED_SEEDS,
        "routing_rounds": rounds,
        "routing_round_selection": round_selection,
        "components": {
            "full": (
                "learned edge weights, learned global drainage-area exponent, "
                "and learned signal gate"
            ),
            "no_area": (
                "learned edge weights and learned signal gate; drainage-area "
                "exponent fixed at one"
            ),
            "topology_only": (
                "uniform edge weights, drainage-area exponent fixed at one, "
                "and learned signal gate"
            ),
        },
        "selection_metric": (
            "mean across all seed-fold cells of the minimum inner-validation "
            "standardized-log runoff MSE"
        ),
        "selection_rule": (
            "find the raw-best mean inner-validation loss; retain components "
            "within one standard error of that raw best; choose the simplest "
            "retained component in the predeclared order topology_only, "
            "no_area, full"
        ),
        "outer_test_rule": (
            "outer-fold NSE/KGE/PBIAS/FHV and paired station differences are "
            "descriptive evidence and are not used to choose the component set"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Component-sweep design mismatch: {DESIGN}")
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


def summarize(rounds: int) -> dict[str, object]:
    station_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        for seed in DECLARED_SEEDS:
            for fold in FOLDS:
                path = directory(component, rounds, seed, fold)
                if not complete(component, rounds, seed, fold):
                    continue
                frame = pd.read_csv(
                    path / "evaluation_holdout.csv", dtype={"station_id": str}
                )
                frame["component"] = component
                frame["rounds"] = rounds
                frame["seed"] = seed
                frame["fold"] = fold
                station_frames.append(frame)
                cell_rows.append(
                    {
                        "component": component,
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
                params = json.loads(
                    path.joinpath("norm_params.json").read_text(encoding="utf-8")
                )
                parameter_rows.append(
                    {
                        "component": component,
                        "seed": seed,
                        "fold": fold,
                        "n_param": int(params["n_param"]),
                        "learned_beta": params.get("learned_beta"),
                    }
                )
    if not station_frames:
        return {}

    stations = pd.concat(station_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows)
    parameters = pd.DataFrame(parameter_rows)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    cells.to_csv(OUT / "cell_metrics.csv", index=False)
    parameters.to_csv(OUT / "parameter_metrics.csv", index=False)

    seed_rows: list[dict[str, object]] = []
    for (component, seed), frame in stations.groupby(
        ["component", "seed"], sort=False
    ):
        matching = cells[
            (cells.component == component) & (cells.seed == seed)
        ]
        seed_rows.append(
            {
                "component": component,
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

    summary_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        frame = cells[cells.component == component]
        seed_frame = seeds[seeds.component == component]
        if frame.empty:
            continue
        val_sd = float(frame.best_validation_loss.std(ddof=1))
        component_params = parameters[parameters.component == component]
        beta = pd.to_numeric(component_params.learned_beta, errors="coerce")
        summary_rows.append(
            {
                "component": component,
                "complexity_rank": COMPLEXITY_RANK[component],
                "cells": len(frame),
                "seeds_complete": int(seed_frame.seed.nunique()),
                "min_folds_per_seed": int(seed_frame.folds.min()),
                "validation_loss_mean": float(
                    frame.best_validation_loss.mean()
                ),
                "validation_loss_se": (
                    val_sd / math.sqrt(len(frame)) if len(frame) > 1 else np.nan
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
                "n_param_median": int(component_params.n_param.median()),
                "learned_beta_median": (
                    float(beta.median()) if beta.notna().any() else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "component_summary.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    reference = stations[stations.component == "full"][
        ["station_id", "seed", "fold", "NSE_full", "KGE_full"]
    ].rename(
        columns={"NSE_full": "NSE_full_ref", "KGE_full": "KGE_full_ref"}
    )
    for component in ["no_area", "topology_only"]:
        other = stations[stations.component == component][
            ["station_id", "seed", "fold", "NSE_full", "KGE_full"]
        ].rename(
            columns={
                "NSE_full": "NSE_full_other",
                "KGE_full": "KGE_full_other",
            }
        )
        joined = reference.merge(
            other, on=["station_id", "seed", "fold"], how="inner"
        )
        delta_nse = joined.NSE_full_ref - joined.NSE_full_other
        delta_kge = joined.KGE_full_ref - joined.KGE_full_other
        paired_rows.append(
            {
                "contrast": f"full-minus-{component}",
                "n": len(joined),
                "paired_median_delta_NSE_full": float(delta_nse.median()),
                "paired_mean_delta_NSE_full": float(delta_nse.mean()),
                "full_NSE_win_fraction": float((delta_nse > 0).mean()),
                "paired_median_delta_KGE_full": float(delta_kge.median()),
                "full_KGE_win_fraction": float((delta_kge > 0).mean()),
            }
        )
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_contrasts.csv", index=False)

    required = len(DECLARED_SEEDS) * len(FOLDS)
    complete_summary = summary[
        (summary.cells == required)
        & (summary.seeds_complete == len(DECLARED_SEEDS))
        & (summary.min_folds_per_seed == len(FOLDS))
    ].copy()
    if len(complete_summary) != len(COMPONENTS):
        return {}
    raw = complete_summary.loc[
        complete_summary.validation_loss_mean.idxmin()
    ]
    threshold = float(raw.validation_loss_mean + raw.validation_loss_se)
    eligible = complete_summary[
        complete_summary.validation_loss_mean <= threshold
    ].sort_values("complexity_rank")
    selected = str(eligible.iloc[0].component)
    selection = {
        "rounds": rounds,
        "raw_best_component": str(raw.component),
        "raw_best_validation_loss_mean": float(raw.validation_loss_mean),
        "raw_best_validation_loss_se": float(raw.validation_loss_se),
        "one_standard_error_threshold": threshold,
        "eligible_components": eligible.component.tolist(),
        "selected_component": selected,
        "selection_rule": (
            "simplest component within one SE of raw-best validation loss"
        ),
    }
    OUT.joinpath("selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def execute(component: str, rounds: int, seed: int, fold: int) -> None:
    tag = run_tag(seed, fold)
    if component == "full":
        if not R.complete(rounds, seed, fold):
            R.execute(rounds, seed, fold)
        return
    P.run(
        fold=fold,
        seed=seed,
        stage1_tag=tag,
        tag=tag,
        variant=VARIANTS[component],
        output_root=output_root(component, rounds),
        rounds_override=rounds,
    )


def parse_int_set(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise ValueError("At least one integer is required.")
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
    parser.add_argument("--manifest-name", default=MANIFEST.name)
    parser.add_argument("--defer-summary", action="store_true")
    args = parser.parse_args()

    rounds = selected_round()
    components = [
        value.strip() for value in args.components.split(",") if value.strip()
    ]
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown components: {unknown}")
    seeds = parse_int_set(args.seeds)
    folds = parse_int_set(args.folds)
    declare_design(rounds)
    if args.summarize_only:
        print(
            json.dumps(summarize(rounds), ensure_ascii=False, indent=2),
            flush=True,
        )
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
        (str(row["component"]), int(row["seed"]), int(row["fold"])): row
        for row in rows
    }
    for component in components:
        for seed in seeds:
            for fold in folds:
                started = time.perf_counter()
                status = "reused"
                if args.rerun or not complete(component, rounds, seed, fold):
                    execute(component, rounds, seed, fold)
                    status = "completed"
                if not complete(component, rounds, seed, fold):
                    raise RuntimeError(
                        f"Incomplete component={component}, seed={seed}, fold={fold}"
                    )
                key = (component, seed, fold)
                by_key[key] = {
                    "component": component,
                    "rounds": rounds,
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "duration_seconds": time.perf_counter() - started,
                    "output_directory": str(
                        directory(component, rounds, seed, fold)
                    ),
                }
                pd.DataFrame(
                    sorted(
                        by_key.values(),
                        key=lambda row: (
                            COMPONENTS.index(str(row["component"])),
                            DECLARED_SEEDS.index(int(row["seed"])),
                            int(row["fold"]),
                        ),
                    )
                ).to_csv(manifest_path, index=False)
        if not args.defer_summary:
            print(
                json.dumps(summarize(rounds), ensure_ascii=False, indent=2),
                flush=True,
            )
    if not args.defer_summary:
        print(
            json.dumps(summarize(rounds), ensure_ascii=False, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
