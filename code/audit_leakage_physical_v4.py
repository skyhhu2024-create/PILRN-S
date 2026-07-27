# -*- coding: utf-8 -*-
"""Exhaustive canary leakage audit for every physical-v4 experiment cell.

For each archived checkpoint this script verifies partition metadata,
visible-only normalization/training membership, archived-prediction replay, and
prediction invariance after multiplying every externally hidden observation
(outer test, density-thinned, and excluded stations) by 1.5.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import config_v2 as C
import graph_v2
import run_component_sweep_physical_v4 as K
import run_density_physical_v4 as D
import run_round_sweep_physical_v4 as R
import train_coupled_v2 as CP
import train_pilrn_v2 as P
from run_stability_10x5 import DECLARED_SEEDS
from train_regional_lstm import (
    RegionalLSTM,
    flow_to_runoff,
    load_forcing,
    load_obs,
    load_statics,
    runoff_to_flow,
)


OUT = C.SP_V3 / "Leakage_Audit_physical_v4"


@dataclass(frozen=True)
class Cell:
    scope: str
    method: str
    kind: str
    variant: str | None
    seed: int
    fold: int
    density: int
    repeat: int
    directory: Path

    @property
    def key(self) -> str:
        return (
            f"{self.scope}:{self.method}:seed{self.seed}:fold{self.fold:02d}:"
            f"d{self.density}:m{self.repeat}"
        )


class BaseData:
    def __init__(self):
        dyn, dates, node_ids, feats = load_forcing()
        self.dyn = dyn
        self.dates = dates
        self.node_ids = [str(value) for value in node_ids]
        self.feats = [str(value) for value in feats]
        self.id2idx = {value: i for i, value in enumerate(self.node_ids)}
        self.obs = load_obs(self.node_ids, dates)
        self.sta_raw, self.area = load_statics(self.node_ids)
        self.folds = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
        excluded_path = C.BASIN / "station_exclude.csv"
        self.excluded = (
            set(pd.read_csv(excluded_path, dtype=str).iloc[:, 0].astype(str))
            if excluded_path.exists()
            else set()
        )
        self.trainable = [
            value
            for value in self.folds.station_id.astype(str)
            if value in self.id2idx and value not in self.excluded
        ]
        self.t0 = dates.get_loc(pd.Period(C.STUDY_START, "M"))
        self.t1 = dates.get_loc(pd.Period(C.STUDY_END, "M"))
        self.t_tr = dates.get_loc(pd.Period(C.TRAIN_END, "M"))
        self.t_va = dates.get_loc(pd.Period(C.VAL_END, "M"))

    def outer_test(self, fold: int) -> list[str]:
        allowed = set(self.trainable)
        return [
            value
            for value in self.folds.loc[
                self.folds.fold == fold, "station_id"
            ].astype(str)
            if value in allowed
        ]


def tag(seed: int, fold: int) -> str:
    return f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"


def discover_cells(scopes: list[str]) -> list[Cell]:
    cells: list[Cell] = []
    if "main" in scopes:
        for seed in DECLARED_SEEDS:
            for fold in range(1, 11):
                run_tag = tag(seed, fold)
                definitions = [
                    ("regional", "regional", None, C.REGIONAL_OUT / run_tag),
                    (
                        "pilrn_sig",
                        "graph",
                        "pilrn_sig",
                        P.VARIANT_OUT["pilrn_sig"] / run_tag,
                    ),
                    (
                        "coupled_scratch",
                        "coupled",
                        None,
                        CP.OUT / run_tag,
                    ),
                    (
                        "coupled_finetune",
                        "coupled",
                        None,
                        CP.FT_OUT / run_tag,
                    ),
                    (
                        "gat",
                        "graph",
                        "gat",
                        P.VARIANT_OUT["gat"] / run_tag,
                    ),
                ]
                for method, kind, variant, directory in definitions:
                    cells.append(
                        Cell(
                            "main",
                            method,
                            kind,
                            variant,
                            seed,
                            fold,
                            100,
                            1,
                            directory,
                        )
                    )
    if "ablation" in scopes:
        variants = [
            "pilrn_sig_no_area",
            "pilrn_sig_topology",
            "pilrn_sig_shuffled",
            "pilrn",
            "pilrn_sig_r1",
            "pilrn_sig_r3",
        ]
        for fold in range(1, 11):
            for variant in variants:
                cells.append(
                    Cell(
                        "ablation",
                        variant,
                        "graph",
                        variant,
                        42,
                        fold,
                        100,
                        1,
                        P.VARIANT_OUT[variant] / f"fold_{fold:02d}",
                    )
                )
    if "density" in scopes:
        for density in [50, 25]:
            for repeat in D.MASK_REPEATS:
                for fold in D.FOLDS:
                    for method in D.METHODS:
                        cells.append(
                            Cell(
                                "density",
                                method,
                                "regional" if method == "regional" else "graph",
                                None if method == "regional" else method,
                                D.MODEL_SEED,
                                fold,
                                density,
                                repeat,
                                D.directory(method, fold, density, repeat),
                            )
                        )
    if "round_sweep" in scopes:
        if not R.OUT.joinpath("selection.json").exists():
            raise FileNotFoundError(
                f"Complete and summarize the routing-depth sweep first: {R.OUT}"
            )
        selection = json.loads(
            R.OUT.joinpath("selection.json").read_text(encoding="utf-8")
        )
        rounds_values = [int(value) for value in selection["rounds_compared"]]
        for rounds in rounds_values:
            for seed in DECLARED_SEEDS:
                for fold in R.FOLDS:
                    cells.append(
                        Cell(
                            "round_sweep",
                            f"pilrn_sig_r{rounds}",
                            "graph",
                            "pilrn_sig",
                            seed,
                            fold,
                            100,
                            1,
                            R.directory(rounds, seed, fold),
                        )
                    )
    if "components" in scopes:
        rounds = K.selected_round()
        for component in K.COMPONENTS:
            for seed in DECLARED_SEEDS:
                for fold in K.FOLDS:
                    cells.append(
                        Cell(
                            "components",
                            component,
                            "graph",
                            K.VARIANTS[component],
                            seed,
                            fold,
                            100,
                            1,
                            K.directory(component, rounds, seed, fold),
                        )
                    )
    if "final_pilrn" in scopes:
        root = C.SP_V3 / "PILRN_sig_final_noarea_r3_v5"
        for seed in DECLARED_SEEDS:
            for fold in range(1, 11):
                cells.append(
                    Cell(
                        "final_pilrn",
                        "pilrn_final",
                        "graph",
                        "pilrn_sig_no_area",
                        seed,
                        fold,
                        100,
                        1,
                        root / tag(seed, fold),
                    )
                )
    if "final_topology" in scopes:
        root = C.SP_V3 / "PILRN_topology_final_r3_v5"
        for seed in DECLARED_SEEDS:
            for fold in range(1, 11):
                cells.append(
                    Cell(
                        "final_topology",
                        "topology_only",
                        "graph",
                        "pilrn_sig_topology",
                        seed,
                        fold,
                        100,
                        1,
                        root / tag(seed, fold),
                    )
                )
    if "final_coupled" in scopes:
        roots = {
            "coupled_scratch": C.SP_V3
            / "Coupled_scratch_final_noarea_r3_v5",
            "coupled_warm": C.SP_V3
            / "Coupled_warm_final_noarea_r3_v5",
        }
        for method, root in roots.items():
            for seed in DECLARED_SEEDS:
                for fold in range(1, 11):
                    cells.append(
                        Cell(
                            "final_coupled",
                            method,
                            "coupled",
                            None,
                            seed,
                            fold,
                            100,
                            1,
                            root / tag(seed, fold),
                        )
                    )
    if "final_density" in scopes:
        root = C.SP_V3 / "PILRN_sig_density_final_noarea_r3_v5"
        for density in [50, 25]:
            for repeat in [1, 2, 3]:
                for fold in range(1, 11):
                    cells.append(
                        Cell(
                            "final_density",
                            "pilrn_final",
                            "graph",
                            "pilrn_sig_no_area",
                            42,
                            fold,
                            density,
                            repeat,
                            root
                            / f"fold_{fold:02d}_d{density:03d}_mask{repeat:02d}",
                        )
                    )
    if "final_density_coupled" in scopes:
        root = C.SP_V3 / "Coupled_warm_density_final_noarea_r3_v6"
        for density in [50, 25]:
            for repeat in [1, 2, 3]:
                for fold in range(1, 11):
                    cells.append(
                        Cell(
                            "final_density_coupled",
                            "coupled_warm_density",
                            "coupled",
                            None,
                            42,
                            fold,
                            density,
                            repeat,
                            root
                            / f"fold_{fold:02d}_d{density:03d}_mask{repeat:02d}",
                        )
                    )
    if "final_density_coupled_scratch" in scopes:
        root = C.SP_V3 / "Coupled_scratch_density_final_noarea_r3_v7"
        for density in [50, 25]:
            for repeat in [1, 2, 3]:
                for fold in range(1, 11):
                    cells.append(
                        Cell(
                            "final_density_coupled_scratch",
                            "coupled_scratch_density",
                            "coupled",
                            None,
                            42,
                            fold,
                            density,
                            repeat,
                            root
                            / f"fold_{fold:02d}_d{density:03d}_mask{repeat:02d}",
                        )
                    )
    return cells


def read_params(cell: Cell) -> dict[str, Any]:
    path = cell.directory / "norm_params.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def visible_and_validation(
    data: BaseData, cell: Cell, params: dict[str, Any]
) -> tuple[list[str], list[str]]:
    test = set(data.outer_test(cell.fold))
    default_visible = [value for value in data.trainable if value not in test]
    visible = [
        str(value)
        for value in params.get(
            "visible_stations", params.get("training_visible", default_visible)
        )
    ]
    validation = [str(value) for value in params.get("validation_stations", [])]
    return visible, validation


def time_indices(data: BaseData, months: str) -> np.ndarray:
    if months == "all":
        return np.arange(data.t0, data.t1 + 1, dtype=np.int64)
    return np.unique(np.linspace(data.t0, data.t1, 12).astype(np.int64))


def load_archive(
    data: BaseData, cell: Cell, indices: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    stations = data.outer_test(cell.fold)
    result = np.full((len(indices), len(stations)), np.nan, dtype=np.float64)
    errors: list[str] = []
    periods = data.dates[indices]
    for column, station in enumerate(stations):
        path = cell.directory / "predictions_by_station" / f"{station}.csv"
        if not path.exists():
            errors.append(f"missing:{path}")
            continue
        frame = pd.read_csv(path)
        source = pd.Series(
            pd.to_numeric(frame.predicted_flow_m3s, errors="coerce").values,
            index=pd.PeriodIndex(pd.to_datetime(frame.date), freq="M"),
        )
        result[:, column] = source.reindex(periods).values
    return result, errors


def compare_archive(
    replay: np.ndarray, archived: np.ndarray, errors: list[str]
) -> tuple[bool, float, str]:
    if errors:
        return False, float("inf"), errors[0]
    if replay.shape != archived.shape:
        return False, float("inf"), "shape mismatch"
    if not np.array_equal(np.isnan(replay), np.isnan(archived)):
        return False, float("inf"), "NaN layout mismatch"
    mask = np.isfinite(replay) & np.isfinite(archived)
    difference = (
        float(np.max(np.abs(replay[mask] - archived[mask]))) if mask.any() else 0.0
    )
    scale = float(np.max(np.abs(archived[mask]))) if mask.any() else 1.0
    tolerance = 1e-4 + 1e-5 * max(scale, 1.0)
    return difference <= tolerance, difference, f"tolerance={tolerance:.6g}"


def exact_prediction_pair(
    original: np.ndarray, perturbed: np.ndarray
) -> tuple[bool, float]:
    if original.shape != perturbed.shape:
        return False, float("inf")
    if not np.array_equal(np.isnan(original), np.isnan(perturbed)):
        return False, float("inf")
    mask = np.isfinite(original) & np.isfinite(perturbed)
    difference = (
        float(np.max(np.abs(original[mask] - perturbed[mask]))) if mask.any() else 0.0
    )
    return np.array_equal(original, perturbed, equal_nan=True), difference


def partition_and_normalization(
    data: BaseData,
    cell: Cell,
    params: dict[str, Any],
    visible: list[str],
    validation: list[str],
) -> tuple[bool, bool, bool, np.ndarray, np.ndarray, float, float]:
    test = set(data.outer_test(cell.fold))
    visible_set = set(visible)
    validation_set = set(validation)
    trainable_set = set(data.trainable)
    partition_ok = all(
        [
            bool(test),
            visible_set <= trainable_set,
            not (visible_set & test),
            not (visible_set & data.excluded),
            validation_set <= visible_set,
            test <= (trainable_set - visible_set),
        ]
    )

    visible_idx = np.asarray([data.id2idx[value] for value in visible], dtype=np.int64)
    runoff = flow_to_runoff(data.obs, data.area, data.dates)
    y_log = np.log1p(np.clip(runoff, 0, None))
    values = y_log[data.t0 : data.t_tr + 1][:, visible_idx]
    y_mu = float(np.nanmean(values))
    y_sd = float(np.nanstd(values)) + 1e-8
    norm_ok = (
        abs(float(params["y_mu"]) - y_mu) <= 1e-12
        and abs(float(params["y_sd"]) - y_sd) <= 1e-12
    )

    external_hidden = (
        (trainable_set - visible_set) | data.excluded
    )
    hidden_idx = np.asarray(
        [data.id2idx[value] for value in sorted(external_hidden) if value in data.id2idx],
        dtype=np.int64,
    )
    obs_perturbed = data.obs.copy()
    values_hidden = obs_perturbed[:, hidden_idx]
    finite = np.isfinite(values_hidden)
    values_hidden[finite] *= 1.5
    obs_perturbed[:, hidden_idx] = values_hidden
    perturbed_runoff = flow_to_runoff(obs_perturbed, data.area, data.dates)
    y_log_perturbed = np.log1p(np.clip(perturbed_runoff, 0, None))
    perturbed_values = y_log_perturbed[data.t0 : data.t_tr + 1][:, visible_idx]
    perturb_mu = float(np.nanmean(perturbed_values))
    perturb_sd = float(np.nanstd(perturbed_values)) + 1e-8
    training_index_ok = y_mu == perturb_mu and y_sd == perturb_sd
    return (
        partition_ok,
        norm_ok,
        training_index_ok,
        hidden_idx,
        obs_perturbed,
        y_mu,
        y_sd,
    )


def regional_replay(
    data: BaseData,
    cell: Cell,
    params: dict[str, Any],
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    d_mu = np.asarray(params["d_mu"], dtype=np.float32)
    d_sd = np.asarray(params["d_sd"], dtype=np.float32)
    s_mu = np.asarray(params["s_mu"], dtype=np.float32)
    s_sd = np.asarray(params["s_sd"], dtype=np.float32)
    dyn_n = (data.dyn - d_mu) / d_sd
    sta_n = (data.sta_raw - s_mu) / s_sd
    clim = np.nanmean(dyn_n[data.t0 : data.t_tr + 1], axis=0)
    tp_i, t2_i = data.feats.index("tp"), data.feats.index("t2m")
    sta_n = np.concatenate(
        [sta_n, clim[:, [tp_i, t2_i]]], axis=1
    ).astype(np.float32)
    dyn_ok = ~np.isnan(data.dyn).any(axis=2)
    model = RegionalLSTM(data.dyn.shape[2] + sta_n.shape[1]).to(device)
    model.load_state_dict(
        torch.load(
            cell.directory / "checkpoint.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    test_idx = [data.id2idx[value] for value in data.outer_test(cell.fold)]
    pred_n = np.full((len(indices), len(test_idx)), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for row, time_index in enumerate(indices):
            valid_columns = [
                column
                for column, node in enumerate(test_idx)
                if dyn_ok[
                    time_index - C.SEQ_LENGTH + 1 : time_index + 1, node
                ].all()
            ]
            if not valid_columns:
                continue
            nodes = [test_idx[column] for column in valid_columns]
            sequences = np.stack(
                [
                    np.concatenate(
                        [
                            dyn_n[
                                time_index - C.SEQ_LENGTH + 1 : time_index + 1,
                                node,
                            ],
                            np.repeat(
                                sta_n[node][None, :], C.SEQ_LENGTH, axis=0
                            ),
                        ],
                        axis=1,
                    )
                    for node in nodes
                ]
            ).astype(np.float32)
            pred_n[row, valid_columns] = (
                model(torch.from_numpy(sequences).to(device)).cpu().numpy()
            )
    pred_runoff = np.expm1(pred_n * float(params["y_sd"]) + float(params["y_mu"]))
    return runoff_to_flow(
        pred_runoff,
        data.area[np.asarray(test_idx)],
        data.dates[indices],
    )


class GraphCommonCache:
    def __init__(self, data: BaseData, maximum: int = 3):
        self.data = data
        self.maximum = maximum
        self.values: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

    def get(
        self,
        cell: Cell,
        params: dict[str, Any],
        visible: list[str],
    ) -> dict[str, Any]:
        key = (
            tuple(visible),
            str(params.get("stage1_root", C.REGIONAL_OUT)),
            str(params.get("stage1_tag", tag(cell.seed, cell.fold))),
        )
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        value = self.build(cell, params, visible)
        self.values[key] = value
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return value

    def build(
        self,
        cell: Cell,
        params: dict[str, Any],
        visible: list[str],
    ) -> dict[str, Any]:
        data = self.data
        visible_idx = np.asarray([data.id2idx[value] for value in visible], dtype=np.int64)
        y_mu, y_sd = float(params["y_mu"]), float(params["y_sd"])
        runoff = flow_to_runoff(data.obs, data.area, data.dates)
        y_log = np.log1p(np.clip(runoff, 0, None))
        y_norm = ((y_log - y_mu) / y_sd).astype(np.float32)

        d_slice = data.dyn[data.t0 : data.t_tr + 1][:, visible_idx, :]
        d_mu = np.nanmean(d_slice, axis=(0, 1))
        d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
        dyn_n = np.nan_to_num((data.dyn - d_mu) / d_sd, nan=0.0).astype(np.float32)
        s_mu = data.sta_raw[visible_idx].mean(axis=0)
        s_sd = data.sta_raw[visible_idx].std(axis=0) + 1e-8
        sta_n = ((data.sta_raw - s_mu) / s_sd).astype(np.float32)

        tp_i, t2_i = data.feats.index("tp"), data.feats.index("t2m")
        clim_tp = np.nan_to_num(
            np.nanmean(data.dyn[data.t0 : data.t_tr + 1, :, tp_i], axis=0)
        )
        clim_t2 = np.nan_to_num(
            np.nanmean(data.dyn[data.t0 : data.t_tr + 1, :, t2_i], axis=0)
        )
        edge_index, edge_feat, area_ratio, _, _, _ = graph_v2.build_graph(
            data.node_ids, clim_tp, clim_t2
        )

        stage1_root = Path(params.get("stage1_root", C.REGIONAL_OUT))
        stage1_tag = str(params.get("stage1_tag", tag(cell.seed, cell.fold)))
        reg_flow = np.full(
            (len(data.dates), len(data.node_ids)), np.nan, dtype=np.float64
        )
        prediction_directory = (
            stage1_root / stage1_tag / "predictions_by_station"
        )
        for path in prediction_directory.glob("*.csv"):
            node = data.id2idx.get(path.stem)
            if node is None:
                continue
            frame = pd.read_csv(path)
            periods = pd.PeriodIndex(pd.to_datetime(frame.date), freq="M")
            positions = data.dates.get_indexer(periods)
            valid = positions >= 0
            reg_flow[positions[valid], node] = pd.to_numeric(
                frame.loc[valid, "predicted_flow_m3s"], errors="coerce"
            ).values
        reg_runoff = flow_to_runoff(reg_flow, data.area, data.dates)
        reg_norm = np.nan_to_num(
            (np.log1p(np.clip(reg_runoff, 0, None)) - y_mu) / y_sd,
            nan=0.0,
        ).astype(np.float32)
        return {
            "y_norm": y_norm,
            "y_ok": np.isfinite(y_norm),
            "dyn_n": dyn_n,
            "sta_n": sta_n,
            "tp_i": tp_i,
            "t2_i": t2_i,
            "edge_index": edge_index,
            "edge_feat": edge_feat,
            "area_ratio": area_ratio,
            "reg_norm": reg_norm,
            "reg_flag": np.isfinite(reg_runoff).astype(np.float32),
        }


def signal_configuration(variant: str) -> tuple[int, str, bool]:
    rounds = {"pilrn_sig_r1": 1, "pilrn_sig_r3": 3}.get(variant, 2)
    edge_mode = "uniform" if variant == "pilrn_sig_topology" else "full"
    area = variant not in {"pilrn_sig_no_area", "pilrn_sig_topology"}
    return rounds, edge_mode, area


def graph_replay_pair(
    data: BaseData,
    cell: Cell,
    params: dict[str, Any],
    visible: list[str],
    validation: list[str],
    obs_perturbed: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    cache: GraphCommonCache,
) -> tuple[np.ndarray, np.ndarray, bool]:
    common = cache.get(cell, params, visible)
    n_nodes = len(data.node_ids)
    src = torch.from_numpy(common["edge_index"][0]).to(device)
    dst = torch.from_numpy(common["edge_index"][1]).to(device)
    edge_features = torch.from_numpy(common["edge_feat"]).to(device)
    if cell.variant == "pilrn_sig_shuffled":
        rng = np.random.default_rng(47000 + cell.fold)
        permutation = torch.from_numpy(
            rng.permutation(edge_features.shape[0])
        ).to(device)
        edge_features = edge_features[permutation]
    log_ratio = torch.from_numpy(np.log(common["area_ratio"])).to(device)
    static = torch.from_numpy(common["sta_n"]).to(device)
    dyn_t2 = torch.from_numpy(
        common["dyn_n"][:, :, common["t2_i"]]
    ).to(device)
    regional = torch.from_numpy(common["reg_norm"]).to(device)
    regional_flag = torch.from_numpy(common["reg_flag"]).to(device)
    s_dim = common["sta_n"].shape[1]

    if cell.variant and cell.variant.startswith("pilrn_sig"):
        declared_rounds, edge_mode, area = signal_configuration(cell.variant)
        rounds = int(params.get("rounds", declared_rounds))
        model: Any = P.PILRNSignal(
            s_dim,
            y_mu=float(params["y_mu"]),
            y_sd=float(params["y_sd"]),
            rounds=rounds,
            edge_mode=edge_mode,
            area_correction=area,
        ).to(device)
    elif cell.variant == "gat":
        model = P.GATNet(5 + 1 + s_dim).to(device)
    elif cell.variant == "pilrn":
        model = P.PILRN(5 + 1 + s_dim).to(device)
    else:
        raise ValueError(cell.variant)
    model.load_state_dict(
        torch.load(
            cell.directory / "checkpoint.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    visible_idx = [data.id2idx[value] for value in visible]
    validation_idx = [data.id2idx[value] for value in validation]
    donor_mask = np.zeros(n_nodes, dtype=bool)
    donor_mask[visible_idx] = True
    donor_mask[validation_idx] = False
    test_idx = np.asarray(
        [data.id2idx[value] for value in data.outer_test(cell.fold)], dtype=np.int64
    )
    y_mu, y_sd = float(params["y_mu"]), float(params["y_sd"])
    runoff_perturbed = flow_to_runoff(obs_perturbed, data.area, data.dates)
    y_norm_perturbed = (
        (np.log1p(np.clip(runoff_perturbed, 0, None)) - y_mu) / y_sd
    ).astype(np.float32)
    y_ok_perturbed = np.isfinite(y_norm_perturbed)

    original = np.full((len(indices), len(test_idx)), np.nan, dtype=np.float32)
    perturbed = np.full_like(original, np.nan)
    all_inputs_equal = True

    def observed_input(
        y_norm: np.ndarray, y_ok: np.ndarray, time_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = y_ok[time_index] & donor_mask
        obs = np.where(valid, y_norm[time_index], 0.0).astype(np.float32)
        flag = valid.astype(np.float32)
        obs[test_idx] = 0.0
        flag[test_idx] = 0.0
        return obs, flag

    def embedding_input(
        y_norm: np.ndarray, y_ok: np.ndarray, time_index: int
    ) -> np.ndarray:
        obs, flag = observed_input(y_norm, y_ok, time_index)
        x = np.zeros((n_nodes, 5), dtype=np.float32)
        x[:, 0] = common["reg_norm"][time_index]
        x[:, 1] = common["reg_flag"][time_index]
        x[:, 2] = obs
        x[:, 3] = flag
        x[:, 4] = common["dyn_n"][time_index, :, common["tp_i"]]
        return x

    with torch.inference_mode():
        for row, time_index in enumerate(indices):
            if cell.variant and cell.variant.startswith("pilrn_sig"):
                obs, flag = observed_input(
                    common["y_norm"], common["y_ok"], time_index
                )
                obs_p, flag_p = observed_input(
                    y_norm_perturbed, y_ok_perturbed, time_index
                )
                all_inputs_equal = all_inputs_equal and np.array_equal(
                    obs, obs_p, equal_nan=True
                ) and np.array_equal(flag, flag_p, equal_nan=True)
                pred, _, _ = model(
                    torch.from_numpy(obs).to(device),
                    torch.from_numpy(flag).to(device),
                    regional[time_index],
                    regional_flag[time_index],
                    static,
                    src,
                    dst,
                    edge_features,
                    log_ratio,
                    n_nodes,
                )
                pred_p, _, _ = model(
                    torch.from_numpy(obs_p).to(device),
                    torch.from_numpy(flag_p).to(device),
                    regional[time_index],
                    regional_flag[time_index],
                    static,
                    src,
                    dst,
                    edge_features,
                    log_ratio,
                    n_nodes,
                )
            else:
                x = embedding_input(common["y_norm"], common["y_ok"], time_index)
                xp = embedding_input(
                    y_norm_perturbed, y_ok_perturbed, time_index
                )
                all_inputs_equal = all_inputs_equal and np.array_equal(
                    x, xp, equal_nan=True
                )
                xt = torch.cat(
                    [
                        torch.from_numpy(x).to(device),
                        dyn_t2[time_index].unsqueeze(-1),
                        static,
                    ],
                    dim=-1,
                )
                xpt = torch.cat(
                    [
                        torch.from_numpy(xp).to(device),
                        dyn_t2[time_index].unsqueeze(-1),
                        static,
                    ],
                    dim=-1,
                )
                pred, _, _ = model(
                    xt, src, dst, edge_features, log_ratio, n_nodes
                )
                pred_p, _, _ = model(
                    xpt, src, dst, edge_features, log_ratio, n_nodes
                )
            original[row] = pred[test_idx].cpu().numpy()
            perturbed[row] = pred_p[test_idx].cpu().numpy()
    original_flow = runoff_to_flow(
        np.expm1(original * y_sd + y_mu),
        data.area[test_idx],
        data.dates[indices],
    )
    perturbed_flow = runoff_to_flow(
        np.expm1(perturbed * y_sd + y_mu),
        data.area[test_idx],
        data.dates[indices],
    )
    return original_flow, perturbed_flow, all_inputs_equal


def coupled_replay_pair(
    data: BaseData,
    cell: Cell,
    params: dict[str, Any],
    visible: list[str],
    validation: list[str],
    obs_perturbed: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, bool]:
    visible_idx = np.asarray([data.id2idx[value] for value in visible], dtype=np.int64)
    y_mu, y_sd = float(params["y_mu"]), float(params["y_sd"])
    runoff = flow_to_runoff(data.obs, data.area, data.dates)
    y_norm = (
        (np.log1p(np.clip(runoff, 0, None)) - y_mu) / y_sd
    ).astype(np.float32)
    runoff_p = flow_to_runoff(obs_perturbed, data.area, data.dates)
    y_norm_p = (
        (np.log1p(np.clip(runoff_p, 0, None)) - y_mu) / y_sd
    ).astype(np.float32)
    y_ok, y_ok_p = np.isfinite(y_norm), np.isfinite(y_norm_p)

    d_slice = data.dyn[data.t0 : data.t_tr + 1][:, visible_idx, :]
    d_mu = np.nanmean(d_slice, axis=(0, 1))
    d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
    dyn_n = np.nan_to_num((data.dyn - d_mu) / d_sd, nan=0.0).astype(np.float32)
    s_mu = data.sta_raw[visible_idx].mean(axis=0)
    s_sd = data.sta_raw[visible_idx].std(axis=0) + 1e-8
    sta_n = (data.sta_raw - s_mu) / s_sd
    tp_i, t2_i = data.feats.index("tp"), data.feats.index("t2m")
    clim = np.nanmean(dyn_n[data.t0 : data.t_tr + 1], axis=0)
    sta_n = np.concatenate(
        [sta_n, clim[:, [tp_i, t2_i]]], axis=1
    ).astype(np.float32)

    clim_tp = np.nan_to_num(
        np.nanmean(data.dyn[data.t0 : data.t_tr + 1, :, tp_i], axis=0)
    )
    clim_t2 = np.nan_to_num(
        np.nanmean(data.dyn[data.t0 : data.t_tr + 1, :, t2_i], axis=0)
    )
    edge_index, edge_feat, area_ratio, _, _, _ = graph_v2.build_graph(
        data.node_ids, clim_tp, clim_t2
    )
    src = torch.from_numpy(edge_index[0]).to(device)
    dst = torch.from_numpy(edge_index[1]).to(device)
    ef = torch.from_numpy(edge_feat).to(device)
    lar = torch.from_numpy(np.log(area_ratio)).to(device)
    sta_t = torch.from_numpy(sta_n).to(device)
    sta_rep = np.repeat(sta_n[None, :, :], CP.L, axis=0).transpose(1, 0, 2)
    n_nodes = len(data.node_ids)

    model = CP.CoupledNet(
        dyn_n.shape[2] + sta_n.shape[1],
        sta_n.shape[1],
        y_mu=y_mu,
        y_sd=y_sd,
        router_rounds=int(params.get("rounds", 2)),
        router_edge_mode=str(params.get("router_edge_mode", "full")),
        router_area_correction=bool(
            params.get("router_area_correction", True)
        ),
    ).to(device)
    model.load_state_dict(
        torch.load(
            cell.directory / "checkpoint.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    donor_mask = np.zeros(n_nodes, dtype=bool)
    donor_mask[visible_idx] = True
    donor_mask[[data.id2idx[value] for value in validation]] = False
    test_idx = np.asarray(
        [data.id2idx[value] for value in data.outer_test(cell.fold)], dtype=np.int64
    )
    original = np.full((len(indices), len(test_idx)), np.nan, dtype=np.float32)
    perturbed = np.full_like(original, np.nan)
    all_inputs_equal = True

    def observed_input(
        target: np.ndarray, valid_target: np.ndarray, time_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = valid_target[time_index] & donor_mask
        obs = np.where(valid, target[time_index], 0.0).astype(np.float32)
        flag = valid.astype(np.float32)
        obs[test_idx] = 0.0
        flag[test_idx] = 0.0
        return obs, flag

    with torch.inference_mode():
        for row, time_index in enumerate(indices):
            sequence = dyn_n[
                time_index - CP.L + 1 : time_index + 1
            ].transpose(1, 0, 2)
            sequence_t = torch.from_numpy(
                np.concatenate([sequence, sta_rep], axis=2)
            ).to(device)
            obs, flag = observed_input(y_norm, y_ok, time_index)
            obs_p, flag_p = observed_input(y_norm_p, y_ok_p, time_index)
            all_inputs_equal = all_inputs_equal and np.array_equal(
                obs, obs_p, equal_nan=True
            ) and np.array_equal(flag, flag_p, equal_nan=True)
            pred, _, _, _ = model(
                sequence_t,
                torch.from_numpy(obs).to(device),
                torch.from_numpy(flag).to(device),
                sta_t,
                src,
                dst,
                ef,
                lar,
                n_nodes,
            )
            pred_p, _, _, _ = model(
                sequence_t,
                torch.from_numpy(obs_p).to(device),
                torch.from_numpy(flag_p).to(device),
                sta_t,
                src,
                dst,
                ef,
                lar,
                n_nodes,
            )
            original[row] = pred[test_idx].cpu().numpy()
            perturbed[row] = pred_p[test_idx].cpu().numpy()
    original_flow = runoff_to_flow(
        np.expm1(original * y_sd + y_mu),
        data.area[test_idx],
        data.dates[indices],
    )
    perturbed_flow = runoff_to_flow(
        np.expm1(perturbed * y_sd + y_mu),
        data.area[test_idx],
        data.dates[indices],
    )
    return original_flow, perturbed_flow, all_inputs_equal


def audit_cell(
    data: BaseData,
    cell: Cell,
    months: str,
    device: torch.device,
    cache: GraphCommonCache,
) -> dict[str, Any]:
    started = time.perf_counter()
    params = read_params(cell)
    visible, validation = visible_and_validation(data, cell, params)
    (
        partition_ok,
        normalization_ok,
        training_index_ok,
        hidden_idx,
        obs_perturbed,
        _,
        _,
    ) = partition_and_normalization(data, cell, params, visible, validation)
    indices = time_indices(data, months)

    if cell.kind == "regional":
        replay = regional_replay(data, cell, params, indices, device)
        replay_perturbed = replay.copy()
        canary_input_ok = training_index_ok
    elif cell.kind == "graph":
        replay, replay_perturbed, canary_input_ok = graph_replay_pair(
            data,
            cell,
            params,
            visible,
            validation,
            obs_perturbed,
            indices,
            device,
            cache,
        )
    elif cell.kind == "coupled":
        replay, replay_perturbed, canary_input_ok = coupled_replay_pair(
            data,
            cell,
            params,
            visible,
            validation,
            obs_perturbed,
            indices,
            device,
        )
    else:
        raise ValueError(cell.kind)

    canary_prediction_ok, canary_difference = exact_prediction_pair(
        replay, replay_perturbed
    )
    archived, errors = load_archive(data, cell, indices)
    archive_ok, archive_difference, detail = compare_archive(
        replay, archived, errors
    )
    passed = all(
        [
            partition_ok,
            normalization_ok,
            training_index_ok,
            canary_input_ok,
            canary_prediction_ok,
            archive_ok,
        ]
    )
    return {
        "cell_key": cell.key,
        "scope": cell.scope,
        "method": cell.method,
        "variant": cell.variant,
        "seed": cell.seed,
        "fold": cell.fold,
        "density_percent": cell.density,
        "mask_repeat": cell.repeat,
        "months_audited": len(indices),
        "visible_stations": len(visible),
        "validation_stations": len(validation),
        "externally_hidden_stations": len(hidden_idx),
        "partition_pass": partition_ok,
        "normalization_pass": normalization_ok,
        "training_index_pass": training_index_ok,
        "canary_input_pass": canary_input_ok,
        "canary_prediction_pass": canary_prediction_ok,
        "canary_max_abs_prediction_diff": canary_difference,
        "archive_reproduction_pass": archive_ok,
        "archive_max_abs_prediction_diff": archive_difference,
        "audit_pass": passed,
        "detail": detail,
        "duration_seconds": time.perf_counter() - started,
        "output_directory": str(cell.directory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scopes",
        default="main,ablation,density",
        help=(
            "Comma-separated subset of main,ablation,density,round_sweep,"
            "components,final_pilrn,final_topology,final_coupled,"
            "final_density,final_density_coupled,"
            "final_density_coupled_scratch."
        ),
    )
    parser.add_argument("--months", choices=["all", "smoke"], default="all")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="CPU is the release default because repeated sparse reductions are bit-exact.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Directory for audit rows and summary files.",
    )
    args = parser.parse_args()
    scopes = [value.strip() for value in args.scopes.split(",") if value.strip()]
    unknown = sorted(
        set(scopes)
        - {
            "main",
            "ablation",
            "density",
            "round_sweep",
            "components",
            "final_pilrn",
            "final_topology",
            "final_coupled",
            "final_density",
            "final_density_coupled",
            "final_density_coupled_scratch",
        }
    )
    if unknown:
        raise ValueError(f"Unknown scopes: {unknown}")

    cells = discover_cells(scopes)
    if args.limit is not None:
        cells = cells[: args.limit]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"audit_rows_{args.months}.csv"
    existing_rows = (
        pd.read_csv(output_csv).to_dict("records")
        if args.resume and output_csv.exists()
        else []
    )
    by_key = {str(row["cell_key"]): row for row in existing_rows}
    data = BaseData()
    cache = GraphCommonCache(data)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable.")
    device = torch.device(args.device)

    for position, cell in enumerate(cells, start=1):
        if args.resume and cell.key in by_key and bool(by_key[cell.key]["audit_pass"]):
            print(f"REUSE {position}/{len(cells)} {cell.key}", flush=True)
            continue
        print(f"AUDIT {position}/{len(cells)} {cell.key}", flush=True)
        row = audit_cell(data, cell, args.months, device, cache)
        by_key[cell.key] = row
        pd.DataFrame(list(by_key.values())).to_csv(output_csv, index=False)
        print(
            f"  pass={row['audit_pass']} canary_diff="
            f"{row['canary_max_abs_prediction_diff']:.3g} "
            f"archive_diff={row['archive_max_abs_prediction_diff']:.3g}",
            flush=True,
        )
        if not row["audit_pass"]:
            raise RuntimeError(f"Leakage audit failed: {cell.key}: {row}")

    frame = pd.DataFrame(list(by_key.values()))
    summary = {
        "protocol": "physical_v4",
        "months": args.months,
        "cells_expected": len(cells),
        "cells_audited": int(len(frame)),
        "cells_passed": int(frame.audit_pass.sum()) if len(frame) else 0,
        "all_passed": bool(frame.audit_pass.all()) if len(frame) else False,
        "max_canary_prediction_difference": (
            float(frame.canary_max_abs_prediction_diff.max()) if len(frame) else None
        ),
        "max_archive_reproduction_difference": (
            float(frame.archive_max_abs_prediction_diff.max()) if len(frame) else None
        ),
    }
    (output_dir / f"audit_summary_{args.months}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
