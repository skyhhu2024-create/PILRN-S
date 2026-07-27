# -*- coding: utf-8 -*-
"""Complete 10-fold x 5-seed stability experiment.

The design is declared before execution and compares four training regimes:

1. regional: standalone Stage-1 regional LSTM;
2. pilrn_sig: pretrained Stage-1 frozen, signal-level router trained separately;
3. coupled_scratch: encoder and router trained jointly from scratch;
4. coupled_finetune: encoder/head initialized from the identical seed/fold
   Regional LSTM checkpoint and then jointly fine-tuned. Its router uses the
   same seed-specific initial state as PILRN-S.

Outputs are resume-safe. Existing completed runs are reused unless --rerun is
specified. Summary files are refreshed after every completed model run.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C


DECLARED_SEEDS = [42, 123, 2026, 7, 2024]
DECLARED_FOLDS = list(range(1, 11))
DECLARED_MODELS = [
    "regional",
    "pilrn_sig",
    "coupled_scratch",
    "coupled_finetune",
    "gat",
]

OUT = C.SP_V3 / "Stability_10x5_physical_v4"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def model_directory(model: str, seed: int, fold: int) -> Path:
    import train_coupled_v2 as CP
    import train_pilrn_v2 as P

    tag = run_tag(seed, fold)
    roots = {
        "regional": C.REGIONAL_OUT,
        "pilrn_sig": P.VARIANT_OUT["pilrn_sig"],
        "coupled_scratch": CP.OUT,
        "coupled_finetune": CP.FT_OUT,
        "gat": P.VARIANT_OUT["gat"],
    }
    return roots[model] / tag


def is_complete(model: str, seed: int, fold: int) -> bool:
    directory = model_directory(model, seed, fold)
    return (
        (directory / "evaluation_holdout.csv").exists()
        and (directory / "checkpoint.pth").exists()
        and (directory / "norm_params.json").exists()
    )


def declare_design(seeds: list[int], folds: list[int], models: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": seeds,
        "folds": folds,
        "models": models,
        "primary_metric": "NSE_full",
        "secondary_metrics": [
            "NSE_test",
            "KGE_full",
            "PBIAS_full",
            "FHV_full",
        ],
        "comparison_principle": (
            "All methods use identical fold and seed labels. PILRN-S uses the "
            "seed-matched frozen Regional LSTM. Coupled fine-tuning initializes "
            "its encoder/head from that same checkpoint and matches the initial "
            "router state; coupled-from-scratch is retained as a separate regime."
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Existing declared design differs from requested design: {DESIGN}"
            )
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return pd.read_csv(MANIFEST).to_dict("records")


def write_manifest(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)


def execute_model(model: str, seed: int, fold: int) -> None:
    import train_coupled_v2 as CP
    import train_pilrn_v2 as P
    import train_regional_lstm as R

    tag = run_tag(seed, fold)
    if model == "regional":
        R.run(fold=fold, seed=seed, tag=tag)
    elif model == "pilrn_sig":
        P.run(
            fold=fold,
            seed=seed,
            stage1_tag=tag,
            tag=tag,
            variant="pilrn_sig",
        )
    elif model == "coupled_scratch":
        CP.run(fold=fold, seed=seed, tag=tag, out_root=CP.OUT)
    elif model == "coupled_finetune":
        CP.run(
            fold=fold,
            seed=seed,
            tag=tag,
            init_stage1_tag=tag,
            out_root=CP.FT_OUT,
            match_router_init=True,
        )
    elif model == "gat":
        P.run(
            fold=fold,
            seed=seed,
            stage1_tag=tag,
            tag=tag,
            variant="gat",
        )
    else:
        raise ValueError(model)


def numeric_medians(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in {"fold", "seed"}:
            continue
        result[f"{column}_median"] = float(frame[column].median())
    return result


def refresh_summaries(
    seeds: list[int], folds: list[int], models: list[str]
) -> None:
    station_frames: list[pd.DataFrame] = []
    cell_rows: list[dict] = []
    for model in models:
        for seed in seeds:
            for fold in folds:
                evaluation = model_directory(model, seed, fold) / "evaluation_holdout.csv"
                if not evaluation.exists():
                    continue
                frame = pd.read_csv(evaluation, dtype={"station_id": str})
                if not len(frame):
                    continue
                frame["method"] = model
                frame["seed"] = seed
                frame["fold"] = fold
                station_frames.append(frame)
                row = {
                    "method": model,
                    "seed": seed,
                    "fold": fold,
                    "n": int(len(frame)),
                    "NSE_full_gt_0.5_fraction": float((frame.NSE_full > 0.5).mean()),
                    "NSE_full_lt_0_fraction": float((frame.NSE_full < 0).mean()),
                }
                row.update(numeric_medians(frame))
                cell_rows.append(row)

    if not station_frames:
        return

    stations = pd.concat(station_frames, ignore_index=True)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    cells = pd.DataFrame(cell_rows).sort_values(["method", "seed", "fold"])
    cells.to_csv(OUT / "cell_metrics.csv", index=False)

    seed_rows: list[dict] = []
    for (method, seed), frame in stations.groupby(["method", "seed"], sort=True):
        row = {
            "method": method,
            "seed": int(seed),
            "n_station_rows": int(len(frame)),
            "n_unique_stations": int(frame.station_id.nunique()),
            "folds_complete": int(frame.fold.nunique()),
            "NSE_full_gt_0.5_fraction": float((frame.NSE_full > 0.5).mean()),
            "NSE_full_lt_0_fraction": float((frame.NSE_full < 0).mean()),
        }
        row.update(numeric_medians(frame))
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows).sort_values(["method", "seed"])
    seed_summary.to_csv(OUT / "seed_summary.csv", index=False)

    method_rows: list[dict] = []
    for method, frame in seed_summary.groupby("method", sort=True):
        row: dict[str, float | int | str] = {
            "method": method,
            "seeds_complete": int(frame.seed.nunique()),
            "min_folds_complete": int(frame.folds_complete.min()),
        }
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column in {
                "seed",
                "n_station_rows",
                "n_unique_stations",
                "folds_complete",
            }:
                continue
            row[f"{column}_across_seed_mean"] = float(frame[column].mean())
            row[f"{column}_across_seed_sd"] = float(frame[column].std(ddof=1))
            row[f"{column}_across_seed_min"] = float(frame[column].min())
            row[f"{column}_across_seed_max"] = float(frame[column].max())
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(OUT / "method_summary.csv", index=False)

    paired_rows: list[dict] = []
    reference = stations[stations.method == "pilrn_sig"]
    for method in [m for m in models if m != "pilrn_sig"]:
        comparison = stations[stations.method == method]
        for seed in seeds:
            left = reference[reference.seed == seed][
                ["station_id", "fold", "NSE_full"]
            ].rename(columns={"NSE_full": "NSE_pilrn"})
            right = comparison[comparison.seed == seed][
                ["station_id", "fold", "NSE_full"]
            ].rename(columns={"NSE_full": "NSE_other"})
            joined = left.merge(right, on=["station_id", "fold"], how="inner")
            if not len(joined):
                continue
            difference = joined.NSE_pilrn - joined.NSE_other
            paired_rows.append(
                {
                    "comparison": f"pilrn_sig-minus-{method}",
                    "seed": seed,
                    "n": int(len(joined)),
                    "paired_median_delta_NSE_full": float(difference.median()),
                    "pilrn_win_fraction": float((difference > 0).mean()),
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        OUT / "paired_seed_differences.csv", index=False
    )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_model_list(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(DECLARED_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DECLARED_SEEDS)
    )
    parser.add_argument(
        "--folds", default=",".join(str(fold) for fold in DECLARED_FOLDS)
    )
    parser.add_argument("--models", default=",".join(DECLARED_MODELS))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    folds = parse_int_list(args.folds)
    models = parse_model_list(args.models)
    declare_design(seeds, folds, models)
    manifest_by_key = {
        (str(row["method"]), int(row["seed"]), int(row["fold"])): row
        for row in read_manifest()
    }

    if args.summarize_only:
        refresh_summaries(seeds, folds, models)
        return

    for seed in seeds:
        for fold in folds:
            for model in models:
                complete = is_complete(model, seed, fold)
                start = time.perf_counter()
                status = "reused" if complete and not args.rerun else "running"
                print(
                    f"\n=== {model} seed={seed} fold={fold:02d} "
                    f"status={status} ===",
                    flush=True,
                )
                if not complete or args.rerun:
                    execute_model(model, seed, fold)
                    if not is_complete(model, seed, fold):
                        raise RuntimeError(
                            f"Run finished without complete artifacts: "
                            f"{model}, seed={seed}, fold={fold}"
                        )
                    status = "completed"
                duration = time.perf_counter() - start
                manifest_by_key[(model, seed, fold)] = {
                    "method": model,
                    "seed": seed,
                    "fold": fold,
                    "status": status,
                    "duration_seconds": duration,
                    "output_directory": str(model_directory(model, seed, fold)),
                }
                write_manifest(
                    sorted(
                        manifest_by_key.values(),
                        key=lambda row: (
                            seeds.index(int(row["seed"])),
                            int(row["fold"]),
                            models.index(str(row["method"])),
                        ),
                    )
                )
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        # Refresh the growing analytical tables once per completed seed,
        # rather than re-reading every result after every individual model.
        refresh_summaries(seeds, folds, models)

    refresh_summaries(seeds, folds, models)
    print(f"\nComplete 10x5 outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
