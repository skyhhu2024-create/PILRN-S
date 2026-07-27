# -*- coding: utf-8 -*-
"""Full leakage audit for the Manuscript v3 leave-station-out protocol.

The release claim is deliberately strong, so the audit is exhaustive by default:

* all ten folds;
* all ten reported methods/training regimes;
* all 297 study months;
* the original held-out/excluded discharge is multiplied by 1.5 in m3/s space;
* the baseline pipeline is first checked against the archived manuscript predictions;
* the perturbed pipeline must then produce bit-identical predictions.

The audit does not retrain neural networks. It loads the archived checkpoints. The
random forest is deterministically refit because the original implementation does
not archive a fitted estimator. Results are written to
Single_Prediction_v3/Leakage_Audit as CSV, JSON, and a compact text log.

Examples
--------
Full release audit::

    python audit_leakage.py --folds 1-10 --models all --months all

Fast development smoke test::

    python audit_leakage.py --folds 1,5 --models all --months smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C
import graph_v2
from train_regional_lstm import (
    RegionalLSTM,
    build_pairs,
    flow_to_runoff,
    load_forcing,
    load_obs,
    load_statics,
    runoff_to_flow,
)


MODEL_ORDER = [
    "idw",
    "nearest",
    "rf",
    "regional",
    "coupled",
    "coupled_warm",
    "gat",
    "pilrn_emb",
    "pilrn_res",
    "pilrn_sig",
]
MODEL_LABELS = {
    "idw": "IDW+DAR",
    "nearest": "Nearest+DAR",
    "rf": "RF",
    "regional": "Regional_LSTM",
    "coupled": "Coupled_e2e",
    "coupled_warm": "Coupled_warm",
    "gat": "GAT",
    "pilrn_emb": "PILRN_emb",
    "pilrn_res": "PILRN_res",
    "pilrn_sig": "PILRN_sig",
}
GRAPH_VARIANTS = {
    "gat": ("gat", C.SP_V3 / "GAT_v2"),
    "pilrn_emb": ("pilrn", C.SP_V3 / "PILRN_v2"),
    "pilrn_res": ("pilrn_res", C.SP_V3 / "PILRN_res_v2"),
    "pilrn_sig": ("pilrn_sig", C.SP_V3 / "PILRN_sig_v2"),
}


@dataclass
class AuditContext:
    dyn: np.ndarray
    dates: pd.PeriodIndex
    node_ids: list[str]
    feats: list[str]
    obs: np.ndarray
    sta_raw: np.ndarray
    area: np.ndarray
    id2idx: dict[str, int]
    fold_df: pd.DataFrame
    excluded: set[str]
    trainable: list[str]
    t0: int
    t1: int
    t_tr: int
    t_va: int
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldContext:
    fold: int
    test_st: list[str]
    visible: list[str]
    vis_idx: np.ndarray
    tst_idx: np.ndarray
    excl_idx: np.ndarray
    hidden_idx: np.ndarray
    vis_mask: np.ndarray
    obs_pert: np.ndarray
    perturbed_values: int
    time_idx: np.ndarray
    cache: dict[str, Any] = field(default_factory=dict)


def parse_int_spec(spec: str, minimum: int, maximum: int) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    values = sorted(set(values))
    if not values or values[0] < minimum or values[-1] > maximum:
        raise ValueError(f"Expected values in [{minimum}, {maximum}], got {spec!r}")
    return values


def parse_models(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(MODEL_ORDER)
    values = [x.strip().lower() for x in spec.split(",") if x.strip()]
    unknown = sorted(set(values) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choose from {MODEL_ORDER}")
    return [m for m in MODEL_ORDER if m in values]


def load_context() -> AuditContext:
    dyn, dates, node_ids, feats = load_forcing()
    node_ids = [str(x) for x in node_ids]
    feats = [str(x) for x in feats]
    id2idx = {s: i for i, s in enumerate(node_ids)}
    obs = load_obs(node_ids, dates)
    sta_raw, area = load_statics(node_ids)
    fold_df = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    exclude_file = C.BASIN / "station_exclude.csv"
    excluded = (
        set(pd.read_csv(exclude_file, dtype=str).iloc[:, 0].astype(str))
        if exclude_file.exists()
        else set()
    )
    trainable = [
        s for s in fold_df["station_id"].astype(str)
        if s in id2idx and s not in excluded
    ]
    return AuditContext(
        dyn=dyn,
        dates=dates,
        node_ids=node_ids,
        feats=feats,
        obs=obs,
        sta_raw=sta_raw,
        area=area,
        id2idx=id2idx,
        fold_df=fold_df,
        excluded=excluded,
        trainable=trainable,
        t0=dates.get_loc(pd.Period(C.STUDY_START, "M")),
        t1=dates.get_loc(pd.Period(C.STUDY_END, "M")),
        t_tr=dates.get_loc(pd.Period(C.TRAIN_END, "M")),
        t_va=dates.get_loc(pd.Period(C.VAL_END, "M")),
    )


def make_fold_context(ctx: AuditContext, fold: int, months: str) -> FoldContext:
    trainable_set = set(ctx.trainable)
    test_st = [
        s for s in ctx.fold_df.loc[ctx.fold_df["fold"] == fold, "station_id"].astype(str)
        if s in trainable_set
    ]
    test_set = set(test_st)
    visible = [s for s in ctx.trainable if s not in test_set]
    vis_idx = np.asarray([ctx.id2idx[s] for s in visible], dtype=np.int64)
    tst_idx = np.asarray([ctx.id2idx[s] for s in test_st], dtype=np.int64)
    excl_idx = np.asarray(
        [ctx.id2idx[s] for s in sorted(ctx.excluded) if s in ctx.id2idx],
        dtype=np.int64,
    )
    hidden_idx = np.unique(np.concatenate([tst_idx, excl_idx])).astype(np.int64)
    vis_mask = np.zeros(len(ctx.node_ids), dtype=bool)
    vis_mask[vis_idx] = True

    # Manuscript definition: multiply raw observed discharge by 1.5.
    obs_pert = ctx.obs.copy()
    hidden_values = obs_pert[:, hidden_idx]
    finite = np.isfinite(hidden_values)
    hidden_values[finite] *= 1.5
    obs_pert[:, hidden_idx] = hidden_values

    if months == "all":
        time_idx = np.arange(ctx.t0, ctx.t1 + 1, dtype=np.int64)
    else:
        time_idx = np.unique(np.linspace(ctx.t0, ctx.t1, 12).astype(np.int64))
    return FoldContext(
        fold=fold,
        test_st=test_st,
        visible=visible,
        vis_idx=vis_idx,
        tst_idx=tst_idx,
        excl_idx=excl_idx,
        hidden_idx=hidden_idx,
        vis_mask=vis_mask,
        obs_pert=obs_pert,
        perturbed_values=int(finite.sum()),
        time_idx=time_idx,
    )


def y_log_from_obs(ctx: AuditContext, obs: np.ndarray) -> np.ndarray:
    runoff = flow_to_runoff(obs, ctx.area, ctx.dates)
    return np.log1p(np.clip(runoff, 0, None))


def y_stats(ctx: AuditContext, fc: FoldContext, y_log: np.ndarray) -> tuple[float, float]:
    values = y_log[ctx.t0:ctx.t_tr + 1][:, fc.vis_idx]
    return float(np.nanmean(values)), float(np.nanstd(values)) + 1e-8


def np_exact(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def diff_arrays(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    nan_layout_equal = np.array_equal(np.isnan(a), np.isnan(b))
    mask = np.isfinite(a) & np.isfinite(b)
    max_abs = float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
    return nan_layout_equal and np_exact(a, b), max_abs


def archive_predictions(
    ctx: AuditContext,
    fc: FoldContext,
    fold_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    periods = ctx.dates[fc.time_idx]
    out = np.full((len(periods), len(fc.test_st)), np.nan, dtype=np.float64)
    errors: list[str] = []
    pdir = fold_dir / "predictions_by_station"
    for col, station in enumerate(fc.test_st):
        path = pdir / f"{station}.csv"
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        data = pd.read_csv(path)
        if "date" not in data or "predicted_flow_m3s" not in data:
            errors.append(f"invalid columns {path}")
            continue
        source_periods = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
        source = pd.Series(
            pd.to_numeric(data["predicted_flow_m3s"], errors="coerce").values,
            index=source_periods,
        )
        out[:, col] = source.reindex(periods).values
    return out, errors


def archive_compare(pred: np.ndarray, archived: np.ndarray, errors: list[str]) -> tuple[bool, float, str]:
    if errors:
        return False, float("inf"), "; ".join(errors[:3])
    if pred.shape != archived.shape or not np.array_equal(np.isnan(pred), np.isnan(archived)):
        return False, float("inf"), "shape or NaN layout differs"
    mask = np.isfinite(pred) & np.isfinite(archived)
    max_abs = float(np.max(np.abs(pred[mask] - archived[mask]))) if mask.any() else 0.0
    scale = float(np.max(np.abs(archived[mask]))) if mask.any() else 1.0
    tolerance = 1e-4 + 1e-5 * max(scale, 1.0)
    return max_abs <= tolerance, max_abs, f"tol={tolerance:.3e}"


def partition_pass(ctx: AuditContext, fc: FoldContext) -> bool:
    visible, test, excluded = set(fc.visible), set(fc.test_st), ctx.excluded
    assigned = set(ctx.fold_df.loc[ctx.fold_df["fold"] == fc.fold, "station_id"].astype(str))
    return all([
        bool(test),
        not (test & visible),
        not (excluded & visible),
        not (excluded & test),
        test <= assigned,
        len(ctx.trainable) == len(fc.visible) + len(fc.test_st),
    ])


def model_directory(model: str, fold: int) -> Path | None:
    tag = f"fold_{fold:02d}"
    if model == "regional":
        return C.REGIONAL_OUT / tag
    if model == "coupled":
        return C.SP_V3 / "Coupled_v2" / tag
    if model == "coupled_warm":
        return C.SP_V3 / "Coupled_finetune_v3" / tag
    if model in GRAPH_VARIANTS:
        return GRAPH_VARIANTS[model][1] / tag
    if model in {"idw", "nearest"}:
        return C.SP_V3 / "Baseline_IDW_v2" / model / tag
    if model == "rf":
        return C.SP_V3 / "Baseline_RF_v2" / tag
    return None


def normalization_pass(
    ctx: AuditContext,
    fc: FoldContext,
    model: str,
    y_log: np.ndarray,
    y_log_pert: np.ndarray,
) -> tuple[bool, bool, str]:
    mu, sd = y_stats(ctx, fc, y_log)
    mu_p, sd_p = y_stats(ctx, fc, y_log_pert)
    perturb_invariant = mu == mu_p and sd == sd_p
    if model in {"idw", "nearest", "rf"}:
        return True, perturb_invariant, "no target normalization"

    fold_dir = model_directory(model, fc.fold)
    assert fold_dir is not None
    norm_file = fold_dir / "norm_params.json"
    if not norm_file.exists():
        return False, perturb_invariant, f"missing {norm_file}"
    with open(norm_file, encoding="utf-8") as handle:
        params = json.load(handle)
    # The manuscript says archived normalization values are reproduced
    # bit-exactly, so this is intentionally stricter than an allclose check.
    ok = float(params["y_mu"]) == mu and float(params["y_sd"]) == sd

    expected_test = set(fc.test_st)
    archived_test = set(params.get("test_stations", params.get("holdout", [])))
    checkpoint_partition = archived_test == expected_test

    if model == "regional":
        d_slice = ctx.dyn[ctx.t0:ctx.t_tr + 1][:, fc.vis_idx, :]
        d_mu = np.nanmean(d_slice, axis=(0, 1))
        d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
        s_mu = ctx.sta_raw[fc.vis_idx].mean(axis=0)
        s_sd = ctx.sta_raw[fc.vis_idx].std(axis=0) + 1e-8
        ok = ok and all([
            np.array_equal(np.asarray(params["d_mu"]), d_mu),
            np.array_equal(np.asarray(params["d_sd"]), d_sd),
            np.array_equal(np.asarray(params["s_mu"]), s_mu),
            np.array_equal(np.asarray(params["s_sd"]), s_sd),
        ])
    return ok and checkpoint_partition, perturb_invariant, f"checkpoint_partition={checkpoint_partition}"


def training_index_pass(
    ctx: AuditContext,
    fc: FoldContext,
    model: str,
    y_log: np.ndarray,
    y_log_pert: np.ndarray,
) -> tuple[bool, str]:
    visible = set(fc.visible)
    if model in {"idw", "nearest"}:
        donors = {ctx.node_ids[j] for j in fc.vis_idx}
        return donors == visible and not (donors & set(fc.test_st)), f"donors={len(donors)}"

    if model == "regional":
        dyn_ok = ~np.isnan(ctx.dyn).any(axis=2)
        pairs = build_pairs(
            list(fc.vis_idx), range(ctx.t0, ctx.t_tr + 1),
            dyn_ok, np.isfinite(y_log), C.SEQ_LENGTH,
        )
        pairs_pert = build_pairs(
            list(fc.vis_idx), range(ctx.t0, ctx.t_tr + 1),
            dyn_ok, np.isfinite(y_log_pert), C.SEQ_LENGTH,
        )
        stations = {ctx.node_ids[j] for j, _ in pairs}
        return pairs == pairs_pert and stations <= visible, f"pairs={len(pairs)} stations={len(stations)}"

    if model == "rf":
        stations = {ctx.node_ids[j] for j in fc.vis_idx}
        return stations == visible and not (stations & set(fc.test_st)), f"training_stations={len(stations)}"

    # Graph variants and Coupled use the same deterministic visible-only
    # validation/target construction. Dynamic training targets are sampled only
    # from train_donor.
    rng = np.random.default_rng(C.SEED)
    val_idx = rng.choice(
        fc.vis_idx,
        size=max(1, int(len(fc.vis_idx) * 0.15)),
        replace=False,
    )
    val_set = set(int(x) for x in val_idx)
    train_donor = {int(j) for j in fc.vis_idx if int(j) not in val_set}
    hidden = set(int(x) for x in fc.hidden_idx)
    ok = val_set <= set(int(x) for x in fc.vis_idx)
    ok = ok and train_donor <= set(int(x) for x in fc.vis_idx)
    ok = ok and not (val_set & hidden) and not (train_donor & hidden)
    return ok, f"train_donor={len(train_donor)} validation={len(val_set)}"


def graph_common(ctx: AuditContext, fc: FoldContext) -> dict[str, Any]:
    if "graph_common" in fc.cache:
        return fc.cache["graph_common"]
    y_log = y_log_from_obs(ctx, ctx.obs)
    y_log_pert = y_log_from_obs(ctx, fc.obs_pert)
    y_mu, y_sd = y_stats(ctx, fc, y_log)
    y_mu_p, y_sd_p = y_stats(ctx, fc, y_log_pert)
    y_norm = ((y_log - y_mu) / y_sd).astype(np.float32)
    y_norm_pert = ((y_log_pert - y_mu_p) / y_sd_p).astype(np.float32)

    d_slice = ctx.dyn[ctx.t0:ctx.t_tr + 1][:, fc.vis_idx, :]
    d_mu = np.nanmean(d_slice, axis=(0, 1))
    d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
    dyn_n = np.nan_to_num((ctx.dyn - d_mu) / d_sd, nan=0.0).astype(np.float32)
    s_mu = ctx.sta_raw[fc.vis_idx].mean(axis=0)
    s_sd = ctx.sta_raw[fc.vis_idx].std(axis=0) + 1e-8
    sta_n = ((ctx.sta_raw - s_mu) / s_sd).astype(np.float32)

    tp_i, t2_i = ctx.feats.index("tp"), ctx.feats.index("t2m")
    clim_tp = np.nan_to_num(np.nanmean(ctx.dyn[ctx.t0:ctx.t_tr + 1, :, tp_i], axis=0))
    clim_t2 = np.nan_to_num(np.nanmean(ctx.dyn[ctx.t0:ctx.t_tr + 1, :, t2_i], axis=0))
    edge_index, edge_feat, area_ratio, _, _, _ = graph_v2.build_graph(
        ctx.node_ids, clim_tp, clim_t2,
    )

    reg_flow = np.full((len(ctx.dates), len(ctx.node_ids)), np.nan, dtype=np.float64)
    pdir = C.REGIONAL_OUT / f"fold_{fc.fold:02d}" / "predictions_by_station"
    for path in pdir.glob("*.csv"):
        j = ctx.id2idx.get(path.stem)
        if j is None:
            continue
        data = pd.read_csv(path)
        periods = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
        idx = ctx.dates.get_indexer(periods)
        valid = idx >= 0
        reg_flow[idx[valid], j] = pd.to_numeric(
            data.loc[valid, "predicted_flow_m3s"], errors="coerce"
        ).values
    reg_runoff = flow_to_runoff(reg_flow, ctx.area, ctx.dates)
    reg_norm = np.nan_to_num(
        (np.log1p(np.clip(reg_runoff, 0, None)) - y_mu) / y_sd,
        nan=0.0,
    ).astype(np.float32)
    reg_flag = np.isfinite(reg_runoff).astype(np.float32)

    common = dict(
        y_log=y_log,
        y_log_pert=y_log_pert,
        y_mu=y_mu,
        y_sd=y_sd,
        y_mu_pert=y_mu_p,
        y_sd_pert=y_sd_p,
        y_norm=y_norm,
        y_norm_pert=y_norm_pert,
        y_ok=np.isfinite(y_norm),
        y_ok_pert=np.isfinite(y_norm_pert),
        dyn_n=dyn_n,
        sta_n=sta_n,
        tp_i=tp_i,
        t2_i=t2_i,
        edge_index=edge_index,
        edge_feat=edge_feat,
        area_ratio=area_ratio,
        reg_norm=reg_norm,
        reg_flag=reg_flag,
    )
    fc.cache["graph_common"] = common
    return common


def audit_idw(ctx: AuditContext, fc: FoldContext, model: str) -> dict[str, Any]:
    from baseline_idw_v2 import predict_fold, river_distance_matrix

    if "river_distance" not in ctx.cache:
        ctx.cache["river_distance"] = river_distance_matrix(ctx.node_ids)
    distance = ctx.cache["river_distance"]
    pred = predict_fold(model, fc.tst_idx, fc.vis_idx, ctx.obs, ctx.area, distance)[fc.time_idx]
    pred_pert = predict_fold(model, fc.tst_idx, fc.vis_idx, fc.obs_pert, ctx.area, distance)[fc.time_idx]
    exact, max_diff = diff_arrays(pred, pred_pert)
    archive, errors = archive_predictions(ctx, fc, model_directory(model, fc.fold))
    archive_ok, archive_diff, archive_detail = archive_compare(pred, archive, errors)
    return dict(
        canary_input_pass=True,
        canary_prediction_pass=exact,
        max_abs_prediction_diff=max_diff,
        archive_reproduction_pass=archive_ok,
        archive_max_abs_diff=archive_diff,
        detail=archive_detail,
    )


def rf_features(ctx: AuditContext) -> np.ndarray:
    if "rf_features" not in ctx.cache:
        from baseline_rf_v2 import build_features
        ctx.cache["rf_features"] = build_features(ctx.dyn, ctx.sta_raw, ctx.feats, ctx.t0)
    return ctx.cache["rf_features"]


def rf_fit_predict(
    ctx: AuditContext,
    fc: FoldContext,
    obs: np.ndarray,
    fitted_model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from baseline_rf_v2 import MIN_LEAF, N_TREES

    X = rf_features(ctx)
    y_log = y_log_from_obs(ctx, obs)
    Xtr: list[np.ndarray] = []
    ytr: list[float] = []
    for j in fc.vis_idx:
        for t in range(ctx.t0, ctx.t_tr + 1):
            if np.isfinite(y_log[t, j]) and not np.isnan(X[t, j]).any():
                Xtr.append(X[t, j])
                ytr.append(float(y_log[t, j]))
    Xtr_arr = np.asarray(Xtr)
    ytr_arr = np.asarray(ytr)
    if fitted_model is None:
        rf = RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=MIN_LEAF,
            n_jobs=-1,
            random_state=C.SEED,
        )
        rf.fit(Xtr_arr, ytr_arr)
        # Keep the released parallel fit, but make repeated frozen-model
        # predictions single-threaded so floating-point aggregation order is
        # fixed for the bit-identity canary.
        rf.n_jobs = 1
    else:
        rf = fitted_model
    pred_log = np.full((len(fc.time_idx), len(fc.tst_idx)), np.nan, dtype=np.float64)
    for col, j in enumerate(fc.tst_idx):
        Xte = X[fc.time_idx, j]
        ok = ~np.isnan(Xte).any(axis=1)
        if ok.any():
            pred_log[ok, col] = rf.predict(Xte[ok])
    pred_runoff = np.expm1(pred_log)
    pred_flow = runoff_to_flow(pred_runoff, ctx.area[fc.tst_idx], ctx.dates[fc.time_idx])
    return pred_flow, Xtr_arr, ytr_arr, rf


def audit_rf(ctx: AuditContext, fc: FoldContext) -> dict[str, Any]:
    # Canary tests a frozen trained model.  The estimator is reconstructed once
    # because the original baseline did not archive it; the perturbed path must
    # generate identical training matrices and then uses the same frozen forest.
    pred, Xtr, ytr, rf = rf_fit_predict(ctx, fc, ctx.obs)
    pred_pert, Xtr_p, ytr_p, _ = rf_fit_predict(ctx, fc, fc.obs_pert, fitted_model=rf)
    input_equal = np_exact(Xtr, Xtr_p) and np_exact(ytr, ytr_p)
    exact, max_diff = diff_arrays(pred, pred_pert)
    archive, errors = archive_predictions(ctx, fc, model_directory("rf", fc.fold))
    archive_ok, archive_diff, archive_detail = archive_compare(pred, archive, errors)
    return dict(
        canary_input_pass=input_equal,
        canary_prediction_pass=exact,
        max_abs_prediction_diff=max_diff,
        archive_reproduction_pass=archive_ok,
        archive_max_abs_diff=archive_diff,
        detail=f"training_rows={len(ytr)}; {archive_detail}",
    )


def regional_predict(
    ctx: AuditContext,
    fc: FoldContext,
    device: Any,
    y_mu: float,
    y_sd: float,
) -> np.ndarray:
    import torch

    fold_dir = model_directory("regional", fc.fold)
    assert fold_dir is not None
    with open(fold_dir / "norm_params.json", encoding="utf-8") as handle:
        params = json.load(handle)
    d_mu = np.asarray(params["d_mu"], dtype=np.float32)
    d_sd = np.asarray(params["d_sd"], dtype=np.float32)
    s_mu = np.asarray(params["s_mu"], dtype=np.float32)
    s_sd = np.asarray(params["s_sd"], dtype=np.float32)
    dyn_n = (ctx.dyn - d_mu) / d_sd
    sta_n = (ctx.sta_raw - s_mu) / s_sd
    clim = np.nanmean(dyn_n[ctx.t0:ctx.t_tr + 1], axis=0)
    tp_i, t2_i = ctx.feats.index("tp"), ctx.feats.index("t2m")
    sta_n = np.concatenate([sta_n, clim[:, [tp_i, t2_i]]], axis=1).astype(np.float32)
    dyn_ok = ~np.isnan(ctx.dyn).any(axis=2)

    model = RegionalLSTM(ctx.dyn.shape[2] + sta_n.shape[1]).to(device)
    model.load_state_dict(torch.load(fold_dir / "checkpoint.pth", map_location=device, weights_only=True))
    model.eval()
    pred_n = np.full((len(fc.time_idx), len(fc.tst_idx)), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for row, t in enumerate(fc.time_idx):
            valid_cols = [
                col for col, j in enumerate(fc.tst_idx)
                if dyn_ok[t - C.SEQ_LENGTH + 1:t + 1, j].all()
            ]
            if not valid_cols:
                continue
            nodes = [fc.tst_idx[col] for col in valid_cols]
            seq = np.stack([
                np.concatenate([
                    dyn_n[t - C.SEQ_LENGTH + 1:t + 1, j],
                    np.repeat(sta_n[j][None, :], C.SEQ_LENGTH, axis=0),
                ], axis=1)
                for j in nodes
            ]).astype(np.float32, copy=False)
            values = model(torch.from_numpy(seq).to(device)).cpu().numpy()
            pred_n[row, valid_cols] = values
    pred_runoff = np.expm1(pred_n * y_sd + y_mu)
    return runoff_to_flow(pred_runoff, ctx.area[fc.tst_idx], ctx.dates[fc.time_idx])


def audit_regional(ctx: AuditContext, fc: FoldContext, device: Any) -> dict[str, Any]:
    y_log = y_log_from_obs(ctx, ctx.obs)
    y_log_p = y_log_from_obs(ctx, fc.obs_pert)
    mu, sd = y_stats(ctx, fc, y_log)
    mu_p, sd_p = y_stats(ctx, fc, y_log_p)
    pred = regional_predict(ctx, fc, device, mu, sd)
    pred_pert = regional_predict(ctx, fc, device, mu_p, sd_p)
    exact, max_diff = diff_arrays(pred, pred_pert)
    archive, errors = archive_predictions(ctx, fc, model_directory("regional", fc.fold))
    archive_ok, archive_diff, archive_detail = archive_compare(pred, archive, errors)
    return dict(
        canary_input_pass=mu == mu_p and sd == sd_p,
        canary_prediction_pass=exact,
        max_abs_prediction_diff=max_diff,
        archive_reproduction_pass=archive_ok,
        archive_max_abs_diff=archive_diff,
        detail=archive_detail,
    )


def graph_predict_pair(
    ctx: AuditContext,
    fc: FoldContext,
    model_name: str,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, bool]:
    import torch
    from train_pilrn_v2 import GATNet, PILRN, PILRNSignal, SIG_ROUNDS

    variant, base_dir = GRAPH_VARIANTS[model_name]
    common = graph_common(ctx, fc)
    N = len(ctx.node_ids)
    src = torch.from_numpy(common["edge_index"][0]).to(device)
    dst = torch.from_numpy(common["edge_index"][1]).to(device)
    ef = torch.from_numpy(common["edge_feat"]).to(device)
    lar = torch.from_numpy(np.log(common["area_ratio"])).to(device)
    sta_t = torch.from_numpy(common["sta_n"]).to(device)
    dyn_t2 = torch.from_numpy(common["dyn_n"][:, :, common["t2_i"]]).to(device)
    reg_norm = torch.from_numpy(common["reg_norm"]).to(device)
    reg_flag = torch.from_numpy(common["reg_flag"]).to(device)
    S = common["sta_n"].shape[1]
    use_res = variant == "pilrn_res"
    n_ch = 6 if use_res else 5
    if variant == "pilrn_sig":
        model = PILRNSignal(S, rounds=SIG_ROUNDS).to(device)
    else:
        model_cls = GATNet if variant == "gat" else PILRN
        model = model_cls(n_ch + 1 + S).to(device)
    checkpoint = base_dir / f"fold_{fc.fold:02d}" / "checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    pred = np.full((len(fc.time_idx), len(fc.tst_idx)), np.nan, dtype=np.float32)
    pred_p = np.full_like(pred, np.nan)
    all_inputs_equal = True

    def make_signal_input(y_norm: np.ndarray, y_ok: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        ok = y_ok[t] & fc.vis_mask
        obs = np.where(ok, y_norm[t], 0.0).astype(np.float32, copy=False)
        flag = ok.astype(np.float32)
        obs[fc.tst_idx] = 0.0
        flag[fc.tst_idx] = 0.0
        return obs, flag

    def make_embedding_input(y_norm: np.ndarray, y_ok: np.ndarray, t: int) -> np.ndarray:
        x = np.zeros((N, n_ch), dtype=np.float32)
        x[:, 0] = common["reg_norm"][t]
        x[:, 1] = common["reg_flag"][t]
        obs, flag = make_signal_input(y_norm, y_ok, t)
        x[:, 2] = obs
        x[:, 3] = flag
        x[:, 4] = common["dyn_n"][t, :, common["tp_i"]]
        if use_res:
            x[:, 5] = (obs - common["reg_norm"][t]) * flag * common["reg_flag"][t]
        return x

    with torch.inference_mode():
        for row, t in enumerate(fc.time_idx):
            if variant == "pilrn_sig":
                obs, flag = make_signal_input(common["y_norm"], common["y_ok"], t)
                obs_p, flag_p = make_signal_input(common["y_norm_pert"], common["y_ok_pert"], t)
                all_inputs_equal = all_inputs_equal and np_exact(obs, obs_p) and np_exact(flag, flag_p)
                p, _, _ = model(
                    torch.from_numpy(obs).to(device), torch.from_numpy(flag).to(device),
                    reg_norm[t], reg_flag[t], sta_t, src, dst, ef, lar, N,
                )
                pp, _, _ = model(
                    torch.from_numpy(obs_p).to(device), torch.from_numpy(flag_p).to(device),
                    reg_norm[t], reg_flag[t], sta_t, src, dst, ef, lar, N,
                )
            else:
                x = make_embedding_input(common["y_norm"], common["y_ok"], t)
                xp = make_embedding_input(common["y_norm_pert"], common["y_ok_pert"], t)
                all_inputs_equal = all_inputs_equal and np_exact(x, xp)
                xt = torch.cat([torch.from_numpy(x).to(device), dyn_t2[t].unsqueeze(-1), sta_t], dim=-1)
                xpt = torch.cat([torch.from_numpy(xp).to(device), dyn_t2[t].unsqueeze(-1), sta_t], dim=-1)
                p, _, _ = model(xt, src, dst, ef, lar, N)
                pp, _, _ = model(xpt, src, dst, ef, lar, N)
                if use_res:
                    p = p + reg_norm[t]
                    pp = pp + reg_norm[t]
            pred[row] = p[fc.tst_idx].cpu().numpy()
            pred_p[row] = pp[fc.tst_idx].cpu().numpy()

    pred_flow = runoff_to_flow(
        np.expm1(pred * common["y_sd"] + common["y_mu"]),
        ctx.area[fc.tst_idx], ctx.dates[fc.time_idx],
    )
    pred_flow_p = runoff_to_flow(
        np.expm1(pred_p * common["y_sd_pert"] + common["y_mu_pert"]),
        ctx.area[fc.tst_idx], ctx.dates[fc.time_idx],
    )
    return pred_flow, pred_flow_p, all_inputs_equal


def audit_graph(ctx: AuditContext, fc: FoldContext, model: str, device: Any) -> dict[str, Any]:
    pred, pred_pert, input_equal = graph_predict_pair(ctx, fc, model, device)
    exact, max_diff = diff_arrays(pred, pred_pert)
    archive, errors = archive_predictions(ctx, fc, model_directory(model, fc.fold))
    archive_ok, archive_diff, archive_detail = archive_compare(pred, archive, errors)
    return dict(
        canary_input_pass=input_equal,
        canary_prediction_pass=exact,
        max_abs_prediction_diff=max_diff,
        archive_reproduction_pass=archive_ok,
        archive_max_abs_diff=archive_diff,
        detail=archive_detail,
    )


def coupled_predict_pair(
    ctx: AuditContext,
    fc: FoldContext,
    device: Any,
    model_key: str = "coupled",
) -> tuple[np.ndarray, np.ndarray, bool]:
    import torch
    from train_coupled_v2 import CoupledNet, L

    common = graph_common(ctx, fc)
    dyn_n = common["dyn_n"]
    clim = np.nanmean(dyn_n[ctx.t0:ctx.t_tr + 1], axis=0)
    sta_n = np.concatenate([
        common["sta_n"], clim[:, [common["tp_i"], common["t2_i"]]],
    ], axis=1).astype(np.float32)
    N = len(ctx.node_ids)
    src = torch.from_numpy(common["edge_index"][0]).to(device)
    dst = torch.from_numpy(common["edge_index"][1]).to(device)
    ef = torch.from_numpy(common["edge_feat"]).to(device)
    lar = torch.from_numpy(np.log(common["area_ratio"])).to(device)
    sta_t = torch.from_numpy(sta_n).to(device)
    sta_rep = np.repeat(sta_n[None, :, :], L, axis=0).transpose(1, 0, 2)

    model = CoupledNet(dyn_n.shape[2] + sta_n.shape[1], sta_n.shape[1]).to(device)
    checkpoint = model_directory(model_key, fc.fold) / "checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    pred = np.full((len(fc.time_idx), len(fc.tst_idx)), np.nan, dtype=np.float32)
    pred_p = np.full_like(pred, np.nan)
    all_inputs_equal = True

    def make_obs(y_norm: np.ndarray, y_ok: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        ok = y_ok[t] & fc.vis_mask
        obs = np.where(ok, y_norm[t], 0.0).astype(np.float32, copy=False)
        flag = ok.astype(np.float32)
        obs[fc.tst_idx] = 0.0
        flag[fc.tst_idx] = 0.0
        return obs, flag

    with torch.inference_mode():
        for row, t in enumerate(fc.time_idx):
            seq = dyn_n[t - L + 1:t + 1].transpose(1, 0, 2)
            seq_t = torch.from_numpy(np.concatenate([seq, sta_rep], axis=2)).to(device)
            obs, flag = make_obs(common["y_norm"], common["y_ok"], t)
            obs_p, flag_p = make_obs(common["y_norm_pert"], common["y_ok_pert"], t)
            all_inputs_equal = all_inputs_equal and np_exact(obs, obs_p) and np_exact(flag, flag_p)
            p, _, _, _ = model(
                seq_t, torch.from_numpy(obs).to(device), torch.from_numpy(flag).to(device),
                sta_t, src, dst, ef, lar, N,
            )
            pp, _, _, _ = model(
                seq_t, torch.from_numpy(obs_p).to(device), torch.from_numpy(flag_p).to(device),
                sta_t, src, dst, ef, lar, N,
            )
            pred[row] = p[fc.tst_idx].cpu().numpy()
            pred_p[row] = pp[fc.tst_idx].cpu().numpy()

    pred_flow = runoff_to_flow(
        np.expm1(pred * common["y_sd"] + common["y_mu"]),
        ctx.area[fc.tst_idx], ctx.dates[fc.time_idx],
    )
    pred_flow_p = runoff_to_flow(
        np.expm1(pred_p * common["y_sd_pert"] + common["y_mu_pert"]),
        ctx.area[fc.tst_idx], ctx.dates[fc.time_idx],
    )
    return pred_flow, pred_flow_p, all_inputs_equal


def audit_coupled(
    ctx: AuditContext,
    fc: FoldContext,
    device: Any,
    model_key: str = "coupled",
) -> dict[str, Any]:
    pred, pred_pert, input_equal = coupled_predict_pair(
        ctx, fc, device, model_key=model_key,
    )
    exact, max_diff = diff_arrays(pred, pred_pert)
    archive, errors = archive_predictions(ctx, fc, model_directory(model_key, fc.fold))
    archive_ok, archive_diff, archive_detail = archive_compare(pred, archive, errors)
    return dict(
        canary_input_pass=input_equal,
        canary_prediction_pass=exact,
        max_abs_prediction_diff=max_diff,
        archive_reproduction_pass=archive_ok,
        archive_max_abs_diff=archive_diff,
        detail=archive_detail,
    )


def clean_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "leakage_audit_summary.csv", index=False)
    records = [{k: clean_number(v) for k, v in row.items()} for row in rows]
    payload = dict(metadata)
    payload["all_passed"] = bool(rows) and all(bool(row.get("status")) for row in rows)
    payload["records"] = records
    with open(output_dir / "leakage_audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "leakage_audit.log", "w", encoding="utf-8") as handle:
        handle.write(
            f"Leakage audit generated {metadata['generated_at']}\n"
            f"Folds: {metadata['folds']}\nModels: {metadata['models']}\n"
            f"Months: {metadata['months']}\n\n"
        )
        for row in rows:
            handle.write(
                f"fold={row['fold']:02d} model={row['model']:<14} "
                f"status={'PASS' if row['status'] else 'FAIL'} "
                f"max_diff={row.get('max_abs_prediction_diff')} "
                f"archive_diff={row.get('archive_max_abs_diff')} "
                f"detail={row.get('detail', '')}\n"
            )


def audit_one(
    ctx: AuditContext,
    fc: FoldContext,
    model: str,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    y_log = y_log_from_obs(ctx, ctx.obs)
    y_log_pert = y_log_from_obs(ctx, fc.obs_pert)
    part_ok = partition_pass(ctx, fc)
    norm_ok, norm_perturb_ok, norm_detail = normalization_pass(
        ctx, fc, model, y_log, y_log_pert,
    )
    train_ok, train_detail = training_index_pass(ctx, fc, model, y_log, y_log_pert)

    if model in {"idw", "nearest"}:
        outcome = audit_idw(ctx, fc, model)
    elif model == "rf":
        outcome = audit_rf(ctx, fc)
    elif model == "regional":
        outcome = audit_regional(ctx, fc, device)
    elif model in {"coupled", "coupled_warm"}:
        outcome = audit_coupled(ctx, fc, device, model_key=model)
    else:
        outcome = audit_graph(ctx, fc, model, device)

    status = all([
        part_ok,
        norm_ok,
        norm_perturb_ok,
        train_ok,
        bool(outcome["canary_input_pass"]),
        bool(outcome["canary_prediction_pass"]),
        bool(outcome["archive_reproduction_pass"]),
    ])
    return dict(
        fold=fc.fold,
        model=MODEL_LABELS[model],
        model_key=model,
        n_test_stations=len(fc.test_st),
        n_months=len(fc.time_idx),
        perturbed_values=fc.perturbed_values,
        partition_pass=part_ok,
        normalization_pass=norm_ok,
        normalization_perturbation_invariant=norm_perturb_ok,
        training_index_pass=train_ok,
        canary_input_pass=bool(outcome["canary_input_pass"]),
        canary_prediction_pass=bool(outcome["canary_prediction_pass"]),
        max_abs_prediction_diff=float(outcome["max_abs_prediction_diff"]),
        archive_reproduction_pass=bool(outcome["archive_reproduction_pass"]),
        archive_max_abs_diff=float(outcome["archive_max_abs_diff"]),
        status=status,
        runtime_seconds=round(time.perf_counter() - started, 3),
        detail=f"{norm_detail}; {train_detail}; {outcome.get('detail', '')}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None, help="Legacy single-fold option")
    parser.add_argument("--folds", default="1-10", help="e.g. 1-10 or 1,5")
    parser.add_argument("--models", default="all", help="all or comma-separated model keys")
    parser.add_argument("--months", choices=["all", "smoke"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    if args.root:
        os.environ["JOH_ROOT"] = args.root

    import torch

    folds = [args.fold] if args.fold is not None else parse_int_spec(args.folds, 1, 10)
    models = parse_models(args.models)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device(args.device)

    output_dir = Path(args.output) if args.output else C.SP_V3 / "Leakage_Audit"
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    metadata = dict(
        generated_at=generated_at,
        folds=folds,
        models=[MODEL_LABELS[m] for m in models],
        months=args.months,
        study_months=297,
        perturbation="raw observed discharge at test and excluded stations multiplied by 1.5",
        device=str(device),
    )

    print("=" * 78)
    print("Manuscript v3 full leakage audit")
    print(f"folds={folds} models={models} months={args.months} device={device}")
    print("=" * 78)
    ctx = load_context()
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fc = make_fold_context(ctx, fold, args.months)
        print(
            f"\n--- fold {fold:02d}: visible={len(fc.visible)} test={len(fc.test_st)} "
            f"months={len(fc.time_idx)} perturbed_values={fc.perturbed_values} ---"
        )
        for model in models:
            try:
                row = audit_one(ctx, fc, model, device)
            except Exception as exc:  # Keep auditing the remaining matrix.
                traceback.print_exc()
                row = dict(
                    fold=fold,
                    model=MODEL_LABELS[model],
                    model_key=model,
                    n_test_stations=len(fc.test_st),
                    n_months=len(fc.time_idx),
                    perturbed_values=fc.perturbed_values,
                    partition_pass=partition_pass(ctx, fc),
                    normalization_pass=False,
                    normalization_perturbation_invariant=False,
                    training_index_pass=False,
                    canary_input_pass=False,
                    canary_prediction_pass=False,
                    max_abs_prediction_diff=float("inf"),
                    archive_reproduction_pass=False,
                    archive_max_abs_diff=float("inf"),
                    status=False,
                    runtime_seconds=0.0,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            print(
                f"  [{'PASS' if row['status'] else 'FAIL'}] {row['model']:<14} "
                f"canary_max={row['max_abs_prediction_diff']:.3e} "
                f"archive_max={row['archive_max_abs_diff']:.3e} "
                f"({row['runtime_seconds']:.1f}s)"
            )
            write_outputs(output_dir, rows, metadata)

    failures = [row for row in rows if not row["status"]]
    print("\n" + "=" * 78)
    print(f"Audit result: {len(rows) - len(failures)}/{len(rows)} model-fold rows passed")
    print(f"Evidence: {output_dir}")
    if failures:
        print("Failures:")
        for row in failures:
            print(f"  fold {row['fold']:02d} {row['model']}: {row['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
