# -*- coding: utf-8 -*-
"""Run the predeclared Section 5.4 physical-routing ablations.

The outer ten folds and seed 42 are fixed.  The full model is shared with the
10x5 stability experiment; every other variant changes one declared component.
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


SEED = 42
FOLDS = list(range(1, 11))
VARIANTS = [
    "pilrn_sig",
    "pilrn_sig_no_area",
    "pilrn_sig_topology",
    "pilrn_sig_shuffled",
    "pilrn",
    "pilrn_sig_r1",
    "pilrn_sig_r3",
]
OUT = C.SP_V3 / "Ablation_physical_v4"
DESIGN = OUT / "experiment_design.json"
MANIFEST = OUT / "run_manifest.csv"


def tag(fold: int) -> str:
    return f"fold_{fold:02d}"


def directory(variant: str, fold: int) -> Path:
    return P.VARIANT_OUT[variant] / tag(fold)


def complete(variant: str, fold: int) -> bool:
    path = directory(variant, fold)
    return all(
        (path / name).exists()
        for name in ("checkpoint.pth", "norm_params.json", "evaluation_holdout.csv")
    )


def write_design() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "physical_v4",
        "seed": SEED,
        "folds": FOLDS,
        "variants": VARIANTS,
        "contrasts": {
            "no_area": "beta fixed at one; learned edge weights retained",
            "topology": "uniform edge weights and beta fixed at one",
            "shuffled": (
                "graph and true transfer area ratio retained; edge descriptors "
                "supplied to the edge-weight MLP are deterministically permuted"
            ),
            "embedding": (
                "matched embedding-level graph model with the same edge-weight "
                "network and no physically interpreted latent-space area multiplier"
            ),
            "rounds": "one, predeclared two, and three routing rounds",
        },
    }
    if DESIGN.exists():
        existing = json.loads(DESIGN.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Declared design mismatch: {DESIGN}")
    else:
        DESIGN.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def summarize() -> None:
    frames = []
    for variant in VARIANTS:
        for fold in FOLDS:
            path = directory(variant, fold) / "evaluation_holdout.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, dtype={"station_id": str})
            frame["variant"] = variant
            frame["fold"] = fold
            frames.append(frame)
    if not frames:
        return
    stations = pd.concat(frames, ignore_index=True)
    stations.to_csv(OUT / "station_metrics.csv", index=False)

    rows = []
    numeric = [
        "NSE_full",
        "NSE_test",
        "KGE_full",
        "PBIAS_full",
        "FHV_full",
    ]
    for variant, frame in stations.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "n": int(len(frame)),
            "folds": int(frame.fold.nunique()),
            "negative_NSE_fraction": float((frame.NSE_full < 0).mean()),
        }
        for name in numeric:
            row[f"{name}_median"] = float(frame[name].median())
            row[f"{name}_iqr"] = float(
                frame[name].quantile(0.75) - frame[name].quantile(0.25)
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "variant_summary.csv", index=False)

    reference = stations[stations.variant == "pilrn_sig"][
        ["station_id", "fold", "NSE_full"]
    ].rename(columns={"NSE_full": "NSE_reference"})
    paired = []
    for variant in [value for value in VARIANTS if value != "pilrn_sig"]:
        other = stations[stations.variant == variant][
            ["station_id", "fold", "NSE_full"]
        ].rename(columns={"NSE_full": "NSE_variant"})
        joined = reference.merge(other, on=["station_id", "fold"], how="inner")
        delta = joined.NSE_reference - joined.NSE_variant
        paired.append(
            {
                "contrast": f"pilrn_sig-minus-{variant}",
                "n": int(len(joined)),
                "paired_median_delta_NSE_full": float(delta.median()),
                "full_model_win_fraction": float((delta > 0).mean()),
                "mean_delta_NSE_full": float(delta.mean()),
            }
        )
    pd.DataFrame(paired).to_csv(OUT / "paired_contrasts.csv", index=False)

    params = []
    for variant in VARIANTS:
        values = []
        betas = []
        for fold in FOLDS:
            path = directory(variant, fold) / "norm_params.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.append(int(payload["n_param"]))
            if payload.get("learned_beta") is not None:
                betas.append(float(payload["learned_beta"]))
        if values:
            params.append(
                {
                    "variant": variant,
                    "n_param": int(np.median(values)),
                    "learned_beta_median": (
                        float(np.median(betas)) if betas else np.nan
                    ),
                    "learned_beta_min": float(np.min(betas)) if betas else np.nan,
                    "learned_beta_max": float(np.max(betas)) if betas else np.nan,
                }
            )
    pd.DataFrame(params).to_csv(OUT / "parameter_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help="Comma-separated subset."
    )
    parser.add_argument(
        "--folds", default=",".join(str(value) for value in FOLDS)
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    write_design()
    if args.summarize_only:
        summarize()
        return

    rows = []
    if MANIFEST.exists():
        rows = pd.read_csv(MANIFEST).to_dict("records")
    by_key = {(str(row["variant"]), int(row["fold"])): row for row in rows}
    for variant in variants:
        for fold in folds:
            started = time.perf_counter()
            status = "reused"
            if args.rerun or not complete(variant, fold):
                P.run(
                    fold=fold,
                    seed=SEED,
                    stage1_tag=tag(fold),
                    tag=tag(fold),
                    variant=variant,
                )
                status = "completed"
            if not complete(variant, fold):
                raise RuntimeError(f"Incomplete {variant} fold {fold}")
            by_key[(variant, fold)] = {
                "variant": variant,
                "fold": fold,
                "seed": SEED,
                "status": status,
                "duration_seconds": time.perf_counter() - started,
                "output_directory": str(directory(variant, fold)),
            }
            pd.DataFrame(
                sorted(
                    by_key.values(),
                    key=lambda row: (
                        VARIANTS.index(str(row["variant"])),
                        int(row["fold"]),
                    ),
                )
            ).to_csv(MANIFEST, index=False)
        summarize()
    summarize()


if __name__ == "__main__":
    main()
