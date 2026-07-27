# -*- coding: utf-8 -*-
"""Density-matched coupled-from-scratch diagnostic.

This companion to ``run_density_coupled_warm_v6.py`` uses the same retained
gauges, folds, three-round no-area router, loss, and training budget, but does
not load a Regional-LSTM checkpoint or match the PILRN-S router state.  It is
intended to distinguish benefits of joint adaptation from benefits that depend
on density-specific pretraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config_v2 as C
import run_density_coupled_warm_v6 as W
import run_density_physical_v4 as D
import train_coupled_v2 as CP


OUT = C.SP_V3 / "Final_Density_Coupled_scratch_v7"
MODEL_ROOT = C.SP_V3 / "Coupled_scratch_density_final_noarea_r3_v7"
FULL_ROOT = C.SP_V3 / "Coupled_scratch_final_noarea_r3_v5"
MANIFEST = OUT / "run_manifest.csv"
DESIGN = OUT / "experiment_design.json"


def declare_design() -> pd.DataFrame:
    masks = D.declare_design()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "density_coupled_scratch_v7",
        "folds": W.FOLDS,
        "densities_percent": W.DENSITIES,
        "mask_repeats": W.MASK_REPEATS,
        "model_seed": W.MODEL_SEED,
        "routing_rounds": W.ROUNDS,
        "area_modulation": False,
        "outer_test_rule": "fixed original outer fold at every density",
        "initialization": "random encoder, head, and router initialization",
        "optimization": (
            "jointly train encoder, head, and router from scratch; same "
            "auxiliary local loss and training budget as the warm control"
        ),
        "comparison_role": (
            "distinguish generic joint adaptation from density-specific "
            "pretraining followed by joint fine-tuning"
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
            raise RuntimeError(f"Density-scratch design mismatch: {DESIGN}")
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
        if not W.complete(fold, density, repeat):
            raise FileNotFoundError(
                f"Missing full-density coupled-from-scratch model for fold {fold}"
            )
        return
    tag = W.run_tag(fold, density, repeat)
    CP.run(
        fold=fold,
        seed=W.MODEL_SEED,
        tag=tag,
        init_stage1_tag=None,
        stage1_root=None,
        visible_ids=visible_ids,
        out_root=MODEL_ROOT,
        match_router_init=False,
        router_rounds=W.ROUNDS,
        router_edge_mode="full",
        router_area_correction=False,
    )


def relabel_outputs() -> None:
    for name in (
        "station_metrics.csv",
        "density_repeat_summary.csv",
        "density_aggregate_summary.csv",
    ):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"station_id": str})
        if "method" in frame.columns:
            frame["method"] = "coupled_scratch"
        frame.to_csv(path, index=False)
    for name in (
        "paired_repeat_contrasts.csv",
        "paired_aggregate_contrasts.csv",
    ):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["contrast"] = "pilrn_final-minus-coupled_scratch"
        frame.to_csv(path, index=False)
    if MANIFEST.exists():
        frame = pd.read_csv(MANIFEST)
        frame["method"] = "coupled_scratch"
        frame.to_csv(MANIFEST, index=False)


def main() -> None:
    W.OUT = OUT
    W.MODEL_ROOT = MODEL_ROOT
    W.FULL_ROOT = FULL_ROOT
    W.MANIFEST = MANIFEST
    W.DESIGN = DESIGN
    W.declare_design = declare_design
    W.execute = execute
    W.main()
    relabel_outputs()


if __name__ == "__main__":
    main()
