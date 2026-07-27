# -*- coding: utf-8 -*-
"""End-to-end training-gauge-density sensitivity for the final router.

The fixed outer test stations remain the evaluation population.  Within every
outer fold, 50% and 25% nested subsets of the outer-training gauges are
retained.  Regional LSTM training, Stage-2 targets/donors, and normalization
all use only retained gauges.  Removed gauges remain unobserved graph nodes
with forcing, attributes, topology, and the density-specific regional prior.

Three repeat masks are used at 50% and 25%.  The final descriptor-conditioned
router is compared with the density-specific Regional prior and a matched
topology-only signal router.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config_v2 as C
import run_density_physical_v4 as D
import run_final_pilrn_v5 as F
import train_pilrn_v2 as P


FOLDS = list(range(1, 11))
DENSITIES = [100, 50, 25]
MASK_REPEATS = [1, 2, 3]
MODEL_SEED = 42
ROUNDS = 3
METHODS = ["regional", "pilrn_final", "topology_only"]
OUT = C.SP_V3 / "Final_Density_noarea_r3_v5"
PILRN_ROOT = C.SP_V3 / "PILRN_sig_density_final_noarea_r3_v5"
TOPOLOGY_ROOT = C.SP_V3 / "PILRN_topology_density_final_r3_v5"
FULL_TOPOLOGY_ROOT = C.SP_V3 / "PILRN_topology_final_r3_v5"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def run_tag(fold: int, density: int, repeat: int) -> str:
    return D.run_tag(fold, density, repeat)


def directory(method: str, fold: int, density: int, repeat: int) -> Path:
    if density == 100:
        tag = f"fold_{fold:02d}"
        if method == "regional":
            return C.REGIONAL_OUT / tag
        if method == "pilrn_final":
            return F.directory(MODEL_SEED, fold)
        return FULL_TOPOLOGY_ROOT / tag
    tag = run_tag(fold, density, repeat)
    if method == "regional":
        return D.REGIONAL_ROOT / tag
    if method == "pilrn_final":
        return PILRN_ROOT / tag
    return TOPOLOGY_ROOT / tag


def complete(method: str, fold: int, density: int, repeat: int) -> bool:
    path = directory(method, fold, density, repeat)
    required = ["checkpoint.pth", "norm_params.json", "evaluation_holdout.csv"]
    if method != "regional":
        required.append("train_log.csv")
    return all((path / name).exists() for name in required)


def declare_design() -> pd.DataFrame:
    masks = D.declare_design()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "manuscript_final_density_v5",
        "folds": FOLDS,
        "model_seed": MODEL_SEED,
        "densities_percent": DENSITIES,
        "mask_repeats": MASK_REPEATS,
        "routing_rounds": ROUNDS,
        "area_modulation": False,
        "methods": {
            "regional": "density-specific Regional LSTM prior",
            "pilrn_final": (
                "three-round signal router with descriptor-conditioned edge "
                "weights and no area modulation"
            ),
            "topology_only": (
                "matched three-round signal router with uniform edge weights "
                "and no area modulation"
            ),
        },
        "outer_test_rule": "fixed original outer fold at every density",
        "removed_training_gauge_role": (
            "unobserved graph node with forcing/static attributes/topology and "
            "a density-specific regional prior; never a Stage-1 or Stage-2 "
            "training target or donor"
        ),
        "aggregation": (
            "station median within each mask repeat; median across the three "
            "repeat-specific medians at 50% and 25%"
        ),
        "mask_source": str(D.MASKS),
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Final density-design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return masks


def execute(
    method: str,
    fold: int,
    density: int,
    repeat: int,
    visible: list[str],
) -> None:
    if method == "regional":
        if density == 100:
            if not complete(method, fold, density, repeat):
                raise FileNotFoundError(
                    f"Missing full-density Regional model for fold {fold}"
                )
            return
        if not D.complete("regional", fold, density, repeat):
            D.execute("regional", fold, density, repeat, visible)
        return
    if density == 100:
        if method == "pilrn_final":
            if not F.complete(MODEL_SEED, fold):
                F.execute(MODEL_SEED, fold)
            return
        tag = f"fold_{fold:02d}"
        P.run(
            fold=fold,
            seed=MODEL_SEED,
            stage1_tag=tag,
            tag=tag,
            variant="pilrn_sig_topology",
            output_root=FULL_TOPOLOGY_ROOT,
            rounds_override=ROUNDS,
        )
        return

    tag = run_tag(fold, density, repeat)
    P.run(
        fold=fold,
        seed=MODEL_SEED,
        stage1_tag=tag,
        tag=tag,
        variant=(
            "pilrn_sig_no_area"
            if method == "pilrn_final"
            else "pilrn_sig_topology"
        ),
        visible_ids=visible,
        stage1_root=D.REGIONAL_ROOT,
        output_root=PILRN_ROOT if method == "pilrn_final" else TOPOLOGY_ROOT,
        rounds_override=ROUNDS,
    )


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

    repeat_rows: list[dict[str, object]] = []
    for (method, density, repeat), frame in stations.groupby(
        ["method", "density_percent", "mask_repeat"], sort=False
    ):
        repeat_rows.append(
            {
                "method": method,
                "density_percent": int(density),
                "mask_repeat": int(repeat),
                "n": len(frame),
                "folds": int(frame.fold.nunique()),
                "NSE_full_median": float(frame.NSE_full.median()),
                "NSE_test_median": float(frame.NSE_test.median()),
                "KGE_full_median": float(frame.KGE_full.median()),
                "PBIAS_full_median": float(frame.PBIAS_full.median()),
                "FHV_full_median": float(frame.FHV_full.median()),
                "negative_NSE_count": int((frame.NSE_full < 0).sum()),
                "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
                "NSE_above_0_5_fraction": float((frame.NSE_full > 0.5).mean()),
            }
        )
    repeats = pd.DataFrame(repeat_rows).sort_values(
        ["method", "density_percent", "mask_repeat"]
    )
    repeats.to_csv(OUT / "density_repeat_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (method, density), frame in repeats.groupby(
        ["method", "density_percent"], sort=False
    ):
        aggregate_rows.append(
            {
                "method": method,
                "density_percent": int(density),
                "repeats": int(frame.mask_repeat.nunique()),
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
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["method", "density_percent"]
    )
    aggregate.to_csv(OUT / "density_aggregate_summary.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    reference = stations[stations.method == "pilrn_final"][
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
            "NSE_full": "NSE_reference",
            "KGE_full": "KGE_reference",
        }
    )
    for comparator in ["regional", "topology_only"]:
        other = stations[stations.method == comparator][
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
        joined = reference.merge(
            other,
            on=["station_id", "fold", "density_percent", "mask_repeat"],
            how="inner",
        )
        joined["delta_NSE"] = joined.NSE_reference - joined.NSE_other
        joined["delta_KGE"] = joined.KGE_reference - joined.KGE_other
        for (density, repeat), frame in joined.groupby(
            ["density_percent", "mask_repeat"], sort=False
        ):
            paired_rows.append(
                {
                    "contrast": f"pilrn_final-minus-{comparator}",
                    "density_percent": int(density),
                    "mask_repeat": int(repeat),
                    "n": len(frame),
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
    paired_aggregate.to_csv(OUT / "paired_aggregate_contrasts.csv", index=False)
    return {
        "protocol": "manuscript_final_density_v5",
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

    masks = declare_design()
    if args.summarize_only:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)
        return

    methods = parse_set(args.methods, str)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    densities = parse_set(args.densities, int)
    repeats = parse_set(args.repeats, int)
    folds = parse_set(args.folds, int)
    rows = (
        pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    )
    by_key = {
        (
            str(row["method"]),
            int(row["fold"]),
            int(row["density_percent"]),
            int(row["mask_repeat"]),
        ): row
        for row in rows
    }
    for method in methods:
        for density in densities:
            active_repeats = [1] if density == 100 else repeats
            for repeat in active_repeats:
                for fold in folds:
                    visible = D.retained_ids(masks, fold, density, repeat)
                    started = time.perf_counter()
                    status = "reused"
                    if not complete(method, fold, density, repeat):
                        execute(method, fold, density, repeat, visible)
                        status = "completed"
                    if not complete(method, fold, density, repeat):
                        raise RuntimeError(
                            f"Incomplete final density cell: method={method}, "
                            f"density={density}, repeat={repeat}, fold={fold}"
                        )
                    key = (method, fold, density, repeat)
                    by_key[key] = {
                        "method": method,
                        "fold": fold,
                        "density_percent": density,
                        "mask_repeat": repeat,
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
                                METHODS.index(str(row["method"])),
                                -int(row["density_percent"]),
                                int(row["mask_repeat"]),
                                int(row["fold"]),
                            ),
                        )
                    ).to_csv(MANIFEST, index=False)
    print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
