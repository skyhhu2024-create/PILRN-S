# -*- coding: utf-8 -*-
"""Density-matched warm-start coupled control for Section 5.5.

The control starts from exactly the density-specific Regional-LSTM checkpoint
used by PILRN-S and reproduces the matched router initialization, but then
jointly fine-tunes the encoder, head, and three-round no-area router.  Thus the
reduced-density comparison isolates freezing versus unfreezing more directly
than a coupled-from-scratch model would.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config_v2 as C
import evaluate_final_manuscript_v5 as E
import run_density_physical_v4 as D
import train_coupled_v2 as CP


FOLDS = list(range(1, 11))
DENSITIES = [100, 50, 25]
MASK_REPEATS = [1, 2, 3]
MODEL_SEED = 42
ROUNDS = 3
OUT = C.SP_V3 / "Final_Density_Coupled_warm_v6"
MODEL_ROOT = C.SP_V3 / "Coupled_warm_density_final_noarea_r3_v6"
FULL_ROOT = C.SP_V3 / "Coupled_warm_final_noarea_r3_v5"
FINAL_DENSITY = C.SP_V3 / "Final_Density_noarea_r3_v5"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(fold: int, density: int, repeat: int) -> str:
    return D.run_tag(fold, density, repeat)


def directory(fold: int, density: int, repeat: int) -> Path:
    if density == 100:
        return FULL_ROOT / f"fold_{fold:02d}"
    return MODEL_ROOT / run_tag(fold, density, repeat)


def complete(fold: int, density: int, repeat: int) -> bool:
    path = directory(fold, density, repeat)
    return all(
        (path / name).exists()
        for name in (
            "checkpoint.pth",
            "norm_params.json",
            "train_log.csv",
            "evaluation_holdout.csv",
        )
    )


def declare_design() -> pd.DataFrame:
    masks = D.declare_design()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "density_coupled_warm_v6",
        "folds": FOLDS,
        "densities_percent": DENSITIES,
        "mask_repeats": MASK_REPEATS,
        "model_seed": MODEL_SEED,
        "routing_rounds": ROUNDS,
        "area_modulation": False,
        "outer_test_rule": "fixed original outer fold at every density",
        "initialization": (
            "density-specific Regional LSTM plus router initialization matched "
            "to density-specific PILRN-S"
        ),
        "optimization": (
            "jointly fine-tune encoder, head, and router; same auxiliary local "
            "loss and training budget as the manuscript coupled controls"
        ),
        "comparison_role": (
            "isolate freezing versus unfreezing after matched density-specific "
            "pretraining"
        ),
        "mask_source": str(D.MASKS),
        "aggregation": (
            "station median within each mask repeat; median across repeat "
            "medians at 50% and 25%"
        ),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Density-coupled design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return masks


def execute(
    fold: int,
    density: int,
    repeat: int,
    visible_ids: list[str],
) -> None:
    if density == 100:
        if not complete(fold, density, repeat):
            raise FileNotFoundError(
                f"Missing full-density warm-start coupled model for fold {fold}"
            )
        return
    tag = run_tag(fold, density, repeat)
    CP.run(
        fold=fold,
        seed=MODEL_SEED,
        tag=tag,
        init_stage1_tag=tag,
        stage1_root=D.REGIONAL_ROOT,
        visible_ids=visible_ids,
        out_root=MODEL_ROOT,
        match_router_init=True,
        router_rounds=ROUNDS,
        router_edge_mode="full",
        router_area_correction=False,
    )


def scored_test_ids(fold: int) -> list[str]:
    frame = pd.read_csv(
        FINAL_DENSITY / "station_metrics.csv",
        dtype={"station_id": str},
    )
    subset = frame[
        (frame.method == "pilrn_final")
        & (frame.density_percent == 100)
        & (frame.mask_repeat == 1)
        & (frame.fold == fold)
    ]
    values = subset.station_id.astype(str).drop_duplicates().tolist()
    if not values:
        raise RuntimeError(f"No scored outer-test stations for fold {fold}")
    return values


def collect_cell(
    fold: int,
    density: int,
    repeat: int,
) -> pd.DataFrame:
    pdir = directory(fold, density, repeat) / "predictions_by_station"
    rows: list[dict[str, object]] = []
    for station_id in scored_test_ids(fold):
        path = pdir / f"{station_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        values = E.evaluate_prediction(path)
        if values is None:
            raise RuntimeError(f"Could not evaluate {path}")
        rows.append(
            {
                "method": "coupled_warm",
                "station_id": station_id,
                "fold": fold,
                "density_percent": density,
                "mask_repeat": repeat,
                **values,
            }
        )
    return pd.DataFrame(rows)


def summarize() -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for density in DENSITIES:
        repeats = [1] if density == 100 else MASK_REPEATS
        for repeat in repeats:
            for fold in FOLDS:
                if complete(fold, density, repeat):
                    frames.append(collect_cell(fold, density, repeat))
    if not frames:
        return {}
    stations = pd.concat(frames, ignore_index=True)
    stations.to_csv(OUT / "station_metrics.csv", index=False)

    repeat_rows: list[dict[str, object]] = []
    for (density, repeat), frame in stations.groupby(
        ["density_percent", "mask_repeat"], sort=False
    ):
        repeat_rows.append(
            {
                "method": "coupled_warm",
                "density_percent": int(density),
                "mask_repeat": int(repeat),
                "n": int(len(frame)),
                "folds": int(frame.fold.nunique()),
                "NSE_full_median": float(frame.NSE_full.median()),
                "KGE_full_median": float(frame.KGE_full.median()),
                "PBIAS_full_median": float(frame.PBIAS_full.median()),
                "FHV_full_median": float(frame.FHV_full.median()),
                "FLV_full_median": float(frame.FLV_full.median()),
                "VR_full_median": float(frame.VR_full.median()),
                "aNSE_full_median": float(frame.aNSE_full.median()),
                "negative_NSE_count": int((frame.NSE_full < 0).sum()),
                "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
                "NSE_above_0_5_fraction": float(
                    (frame.NSE_full > 0.5).mean()
                ),
            }
        )
    repeats = pd.DataFrame(repeat_rows).sort_values(
        ["density_percent", "mask_repeat"]
    )
    repeats.to_csv(OUT / "density_repeat_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for density, frame in repeats.groupby("density_percent", sort=False):
        aggregate_rows.append(
            {
                "method": "coupled_warm",
                "density_percent": int(density),
                "repeats": int(frame.mask_repeat.nunique()),
                "NSE_full_median": float(frame.NSE_full_median.median()),
                "KGE_full_median": float(frame.KGE_full_median.median()),
                "PBIAS_full_median": float(frame.PBIAS_full_median.median()),
                "FHV_full_median": float(frame.FHV_full_median.median()),
                "FLV_full_median": float(frame.FLV_full_median.median()),
                "VR_full_median": float(frame.VR_full_median.median()),
                "aNSE_full_median": float(frame.aNSE_full_median.median()),
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
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("density_percent")
    aggregate.to_csv(OUT / "density_aggregate_summary.csv", index=False)

    pilrn = pd.read_csv(
        FINAL_DENSITY / "station_metrics.csv",
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
    coupled = stations[
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
            "NSE_full": "NSE_coupled",
            "KGE_full": "KGE_coupled",
        }
    )
    joined = pilrn.merge(
        coupled,
        on=["station_id", "fold", "density_percent", "mask_repeat"],
        how="inner",
    )
    joined["delta_NSE"] = joined.NSE_pilrn - joined.NSE_coupled
    joined["delta_KGE"] = joined.KGE_pilrn - joined.KGE_coupled
    paired_rows: list[dict[str, object]] = []
    for (density, repeat), frame in joined.groupby(
        ["density_percent", "mask_repeat"], sort=False
    ):
        paired_rows.append(
            {
                "contrast": "pilrn_final-minus-coupled_warm",
                "density_percent": int(density),
                "mask_repeat": int(repeat),
                "n": int(len(frame)),
                "paired_median_delta_NSE": float(frame.delta_NSE.median()),
                "paired_median_delta_KGE": float(frame.delta_KGE.median()),
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
        "protocol": "density_coupled_warm_v6",
        "aggregation": (
            "station median within repeat, then median across repeat medians"
        ),
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
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    masks = declare_design()
    if args.summarize_only:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return

    densities = parse_set(args.densities, int)
    repeats = parse_set(args.repeats, int)
    folds = parse_set(args.folds, int)
    rows = (
        pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    )
    by_key = {
        (
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
                visible = D.retained_ids(masks, fold, density, repeat)
                started = time.perf_counter()
                status = "reused"
                if args.rerun or not complete(fold, density, repeat):
                    execute(fold, density, repeat, visible)
                    status = "completed"
                if not complete(fold, density, repeat):
                    raise RuntimeError(
                        "Incomplete density-coupled cell: "
                        f"fold={fold}, density={density}, repeat={repeat}"
                    )
                key = (fold, density, repeat)
                by_key[key] = {
                    "method": "coupled_warm",
                    "fold": fold,
                    "density_percent": density,
                    "mask_repeat": repeat,
                    "status": status,
                    "duration_seconds": time.perf_counter() - started,
                    "output_directory": str(directory(fold, density, repeat)),
                }
                pd.DataFrame(
                    sorted(
                        by_key.values(),
                        key=lambda row: (
                            -int(row["density_percent"]),
                            int(row["mask_repeat"]),
                            int(row["fold"]),
                        ),
                    )
                ).to_csv(MANIFEST, index=False)
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
