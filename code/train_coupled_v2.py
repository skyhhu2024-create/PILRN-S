# -*- coding: utf-8 -*-
"""
端到端耦合基线 (协议 v2): 共享 LSTM 编码器 + 信号级物理路由, 联合训练

与 pilrn_sig 的唯一区别 = 优化策略:
  - pilrn_sig: Stage 1 (regional LSTM) 单独训练并冻结 -> Stage 2 路由单独训练 (解耦)
  - 本模型:   LSTM 局地先验 与 路由 联合端到端训练 (耦合)
架构、输入信息、训练协议(动态 mask/观测目标/测试站隔离/空间验证早停)完全一致,
直接回应 R2 Lines 723-726 对"耦合基线被人为削弱"的批评,并干净检验解耦的必要性。

basin-wide 设计: 一个共享 LSTM 吃 (11 动态驱动 + 12 静态属性), 对全部 307 节点
(含无观测节点)输出局地先验 —— 不依赖单站模型, 无资料节点天然可预测。
辅助损失(可见未mask站的局地先验 vs 观测)给耦合模型公平的学习信号。

用法:
    python train_coupled_v2.py --folds 1,2
    python train_coupled_v2.py               # 全部 10 折
输出: Single_Prediction/Coupled_v2/fold_XX/ + kfold_summary.csv
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C
import graph_v2
from train_regional_lstm import (load_forcing, load_obs, load_statics,
                                 flow_to_runoff, runoff_to_flow, metrics)
from train_pilrn_v2 import PILRNSignal

import torch
import torch.nn as nn

L = C.SEQ_LENGTH
ENC_HIDDEN = 128
LR = 1e-3
MAX_EPOCHS = 150     # 与 PILRN-S 对齐
PATIENCE = 20        # 与 PILRN-S 对齐
BATCH_T = 64         # 与 PILRN-S 对齐
DYN_MASK_RATIO = 0.15
SPATIAL_VAL_RATIO = 0.15
AUX_W = 1.0          # 局地先验辅助损失权重(联合训练中 Stage-1 目标的对应物)
STRICT_VAL = True
OUT = C.SP_V3 / "Coupled_scratch_physical_v4"
FT_OUT = C.SP_V3 / "Coupled_warm_physical_v4"


class CoupledNet(nn.Module):
    """共享 LSTM 编码器 -> 局地先验(标量) -> 信号级物理路由 (端到端)"""

    def __init__(
        self,
        in_dim,
        s_dim,
        y_mu,
        y_sd,
        router_rounds=2,
        router_edge_mode="full",
        router_area_correction=True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, ENC_HIDDEN, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(ENC_HIDDEN, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
        self.router = PILRNSignal(
            s_dim,
            y_mu=y_mu,
            y_sd=y_sd,
            rounds=router_rounds,
            edge_mode=router_edge_mode,
            area_correction=router_area_correction,
        )

    def forward(self, seq, obs, vflag, sta, src, dst, ef, lar, n):
        h, _ = self.lstm(seq)                       # [N,L,H]
        prior = self.head(h[:, -1, :]).squeeze(-1)  # [N] 局地先验 (归一化 log 径流深)
        ones = torch.ones_like(prior)
        pred, alpha, scale = self.router(obs, vflag, prior, ones, sta,
                                         src, dst, ef, lar, n)
        return pred, prior, alpha, scale


def run(
    fold,
    seed=C.SEED,
    tag=None,
    init_stage1_tag=None,
    stage1_root=None,
    visible_ids=None,
    out_root=None,
    match_router_init=False,
    router_rounds=2,
    router_edge_mode="full",
    router_area_correction=True,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dyn, dates, node_ids, feats = load_forcing()
    N = len(node_ids)
    id2idx = {s: i for i, s in enumerate(node_ids)}
    obs = load_obs(node_ids, dates)
    sta_raw, area = load_statics(node_ids)

    t0 = dates.get_loc(pd.Period(C.STUDY_START, "M"))
    t1 = dates.get_loc(pd.Period(C.STUDY_END, "M"))
    t_tr = dates.get_loc(pd.Period(C.TRAIN_END, "M"))
    t_va = dates.get_loc(pd.Period(C.VAL_END, "M"))

    fold_df = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    excl = set()
    f_ex = C.BASIN / "station_exclude.csv"
    if f_ex.exists():
        excl = set(pd.read_csv(f_ex, dtype=str).iloc[:, 0])
    trainable = [s for s in fold_df["station_id"] if s in id2idx and s not in excl]
    test_st = [s for s in fold_df.loc[fold_df["fold"] == fold, "station_id"]
               if s in set(trainable)]
    outer_visible = [s for s in trainable if s not in set(test_st)]
    if visible_ids is None:
        visible = outer_visible
    else:
        requested = [str(value) for value in visible_ids]
        invalid = sorted(set(requested) - set(outer_visible))
        if invalid:
            raise ValueError(
                "Density-specific visible stations are not a subset of the "
                f"outer-training population: {invalid[:10]}"
            )
        # Preserve the canonical fold-file order for deterministic validation
        # sampling and normalization.
        requested_set = set(requested)
        visible = [value for value in outer_visible if value in requested_set]
        if not visible:
            raise ValueError("Density-specific visible-station set is empty.")
    vis_idx = np.array([id2idx[s] for s in visible])
    tst_idx = np.array([id2idx[s] for s in test_st])
    vis_mask = np.zeros(N, dtype=bool)
    vis_mask[vis_idx] = True                 # donor 只允许可见站
    rng = np.random.default_rng(seed)
    val_st = rng.choice(
        vis_idx, size=max(1, int(len(vis_idx) * SPATIAL_VAL_RATIO)), replace=False
    )
    train_donor = np.array([j for j in vis_idx if j not in set(val_st)])
    if STRICT_VAL:
        vis_mask[val_st] = False
        print(f"  [STRICT_VAL] 验证站 ({len(val_st)}) 训练全程隐藏")
    tag = tag or (f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}")
    print(f"[coupled {tag}] 可见 {len(visible)} / 测试 {len(test_st)} / device={device}")

    # 归一化 (与 pilrn_sig 相同口径)
    runoff = flow_to_runoff(obs, area, dates)
    y_log = np.log1p(np.clip(runoff, 0, None))
    sl = y_log[t0:t_tr + 1][:, vis_idx]
    y_mu = float(np.nanmean(sl)); y_sd = float(np.nanstd(sl)) + 1e-8
    y_norm = ((y_log - y_mu) / y_sd).astype(np.float32)
    y_ok = np.isfinite(y_norm)

    d_slice = dyn[t0:t_tr + 1][:, vis_idx, :]
    d_mu = np.nanmean(d_slice, axis=(0, 1)); d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
    dyn_n = np.nan_to_num((dyn - d_mu) / d_sd, nan=0.0).astype(np.float32)
    s_mu = sta_raw[vis_idx].mean(axis=0); s_sd = sta_raw[vis_idx].std(axis=0) + 1e-8
    sta_n = (sta_raw - s_mu) / s_sd
    tp_i, t2_i = feats.index("tp"), feats.index("t2m")
    clim = np.nanmean(dyn_n[t0:t_tr + 1], axis=0)
    sta_n = np.concatenate([sta_n, clim[:, [tp_i, t2_i]]], axis=1).astype(np.float32)
    S = sta_n.shape[1]

    clim_tp = np.nan_to_num(np.nanmean(dyn[t0:t_tr + 1, :, tp_i], axis=0))
    clim_t2 = np.nan_to_num(np.nanmean(dyn[t0:t_tr + 1, :, t2_i], axis=0))
    edge_index, edge_feat, area_ratio, river_dist, hc_w, _ = \
        graph_v2.build_graph(node_ids, clim_tp, clim_t2)
    E = edge_index.shape[1]
    src = torch.from_numpy(edge_index[0]).to(device)
    dst = torch.from_numpy(edge_index[1]).to(device)
    ef = torch.from_numpy(edge_feat).to(device)
    lar = torch.from_numpy(np.log(area_ratio)).to(device)
    sta_t = torch.from_numpy(sta_n).to(device)

    # 预构造 [T,N,L,Fd+S] 太大 -> 每个时间步现做序列 (numpy 切片, 快)
    Fd = dyn_n.shape[2]
    sta_rep = np.repeat(sta_n[None, :, :], L, axis=0).transpose(1, 0, 2)  # [N,L,S]

    def make_seq(t):
        seq = dyn_n[t - L + 1: t + 1].transpose(1, 0, 2)                  # [N,L,Fd]
        return torch.from_numpy(np.concatenate([seq, sta_rep], axis=2)).to(device)

    model = CoupledNet(
        Fd + S,
        S,
        y_mu=y_mu,
        y_sd=y_sd,
        router_rounds=router_rounds,
        router_edge_mode=router_edge_mode,
        router_area_correction=router_area_correction,
    ).to(device)
    if init_stage1_tag is not None:
        checkpoint_root = (
            Path(stage1_root) if stage1_root is not None else C.REGIONAL_OUT
        )
        stage1_checkpoint = checkpoint_root / init_stage1_tag / "checkpoint.pth"
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing Stage-1 checkpoint for coupled fine-tuning: {stage1_checkpoint}"
            )
        stage1_state = torch.load(
            stage1_checkpoint, map_location="cpu", weights_only=True
        )
        incompatible = model.load_state_dict(stage1_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        non_router_missing = [
            key for key in incompatible.missing_keys if not key.startswith("router.")
        ]
        if unexpected or non_router_missing:
            raise RuntimeError(
                "Stage-1 checkpoint is not architecture-compatible: "
                f"unexpected={unexpected}, non_router_missing={non_router_missing}"
            )
        if match_router_init:
            # Reproduce the router initialization used by the decoupled PILRN-S
            # run for the same seed. Resetting here removes RNG consumption by
            # construction of the encoder/head as a source of mismatch.
            torch.manual_seed(seed)
            router_template = PILRNSignal(
                S,
                y_mu=y_mu,
                y_sd=y_sd,
                rounds=router_rounds,
                edge_mode=router_edge_mode,
                area_correction=router_area_correction,
            )
            model.router.load_state_dict(router_template.state_dict())
            del router_template
        print(
            f"[coupled {tag}] initialized encoder/head from "
            f"{checkpoint_root.name}/{init_stage1_tag}"
        )
    nparam = sum(p.numel() for p in model.parameters())
    print(f"CoupledNet 参数量: {nparam}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    tr_times = list(range(max(t0, L - 1), t_tr + 1))
    va_times = list(range(t_tr + 1, t_va + 1))

    def forward_t(t, hidden_idx):
        ok = y_ok[t] & vis_mask
        o = np.where(ok, y_norm[t], 0.0).copy()
        vf = ok.astype(np.float32).copy()
        o[hidden_idx] = 0.0; vf[hidden_idx] = 0.0
        return model(make_seq(t), torch.from_numpy(o).to(device),
                     torch.from_numpy(vf).to(device), sta_t, src, dst, ef, lar, N)

    batch_graph_cache = {}

    def forward_batch(times, hidden_idx):
        batch_size = len(times)
        sequences = torch.cat([make_seq(t) for t in times], dim=0)
        obs_rows = []
        flag_rows = []
        for t in times:
            ok = y_ok[t] & vis_mask
            obs_row = np.where(ok, y_norm[t], 0.0).copy()
            flag_row = ok.astype(np.float32).copy()
            obs_row[hidden_idx] = 0.0
            flag_row[hidden_idx] = 0.0
            obs_rows.append(obs_row)
            flag_rows.append(flag_row)
        if batch_size not in batch_graph_cache:
            offsets = (
                torch.arange(batch_size, device=device, dtype=src.dtype) * N
            ).repeat_interleave(E)
            batch_graph_cache[batch_size] = (
                src.repeat(batch_size) + offsets,
                dst.repeat(batch_size) + offsets,
                ef.repeat(batch_size, 1),
                lar.repeat(batch_size, 1),
            )
        src_b, dst_b, ef_b, lar_b = batch_graph_cache[batch_size]
        pred, prior, alpha, scale = model(
            sequences,
            torch.from_numpy(np.concatenate(obs_rows)).to(device),
            torch.from_numpy(np.concatenate(flag_rows)).to(device),
            sta_t.repeat(batch_size, 1),
            src_b,
            dst_b,
            ef_b,
            lar_b,
            batch_size * N,
        )
        return (
            pred.view(batch_size, N),
            prior.view(batch_size, N),
            alpha,
            scale,
        )

    def epoch_pass(times, epoch=0, train=True, hidden_eval=None):
        model.train(train)
        total, count = 0.0, 0
        for i in range(0, len(times), BATCH_T):
            bt = times[i:i + BATCH_T]
            if train:
                k = max(1, int(len(train_donor) * DYN_MASK_RATIO))
                targets = rng.choice(train_donor, size=k, replace=False)
            else:
                targets = hidden_eval
            hidden = np.concatenate([targets, tst_idx]) if len(tst_idx) else targets
            valid_times = []
            valid_targets = []
            for t in bt:
                tm = targets[y_ok[t, targets]]
                if len(tm):
                    valid_times.append(t)
                    valid_targets.append(tm)
            if not valid_times:
                continue
            pred_batch, prior_batch, _, _ = forward_batch(valid_times, hidden)
            hidden_set = set(int(value) for value in hidden)
            losses = []
            for row, (t, tm) in enumerate(zip(valid_times, valid_targets)):
                yt = torch.from_numpy(y_norm[t, tm]).to(device)
                loss = ((pred_batch[row, tm] - yt) ** 2).mean()
                if train:
                    aux_idx = np.array([
                        j for j in train_donor
                        if int(j) not in hidden_set and y_ok[t, j]
                    ])
                    if len(aux_idx):
                        ya = torch.from_numpy(y_norm[t, aux_idx]).to(device)
                        loss = loss + AUX_W * (
                            (prior_batch[row, aux_idx] - ya) ** 2
                        ).mean()
                losses.append(loss)
            loss_b = torch.stack(losses).mean()
            nb = len(losses)
            if train:
                opt.zero_grad(); loss_b.backward(); opt.step()
            total += loss_b.item() * nb; count += nb
        return total / max(count, 1)

    output_root = Path(out_root) if out_root is not None else OUT
    out_dir = output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_state, wait, log = np.inf, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        tl = epoch_pass(tr_times, epoch=ep, train=True)
        with torch.no_grad():
            vl = epoch_pass(va_times, epoch=ep, train=False, hidden_eval=val_st)
        log.append(dict(epoch=ep, train=tl, val=vl))
        print(f"  epoch {ep:02d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop @ {ep}"); break
    model.load_state_dict(best_state)
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(model.state_dict(), out_dir / "checkpoint.pth")

    # masked 推理 + 评估
    model.eval()
    pred_n = np.full((len(dates), N), np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(max(t0, L - 1), t1 + 1):
            p, _, _, _ = forward_t(t, tst_idx)
            pred_n[t] = p.cpu().numpy()
    pred_flow = runoff_to_flow(np.expm1(pred_n * y_sd + y_mu), area, dates)

    pdir = out_dir / "predictions_by_station"; pdir.mkdir(exist_ok=True)
    rows = []
    didx = dates.to_timestamp()
    slf = slice(t0, t1 + 1); te = slice(t_va + 1, t1 + 1)
    for s in test_st:
        j = id2idx[s]
        pd.DataFrame({"date": didx[slf], "predicted_flow_m3s": pred_flow[slf, j],
                      "observed_flow_m3s": obs[slf, j]}).to_csv(pdir / f"{s}.csv", index=False)
        m_te = metrics(obs[te, j], pred_flow[te, j])
        m_fu = metrics(obs[slf, j], pred_flow[slf, j])
        if m_te and m_fu:
            rows.append(dict(station_id=s, fold=fold,
                             **{f"{k}_test": v for k, v in m_te.items()},
                             **{f"{k}_full": v for k, v in m_fu.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "evaluation_holdout.csv", index=False)
    if len(df):
        print(f"[coupled holdout] n={len(df)} | test NSE median={df.NSE_test.median():.3f} "
              f"| full NSE median={df.NSE_full.median():.3f}")
    json.dump(dict(protocol="physical_v4",
                   y_mu=y_mu, y_sd=y_sd, seed=seed, fold=fold, n_param=int(nparam),
                   aux_w=AUX_W, test_stations=test_st,
                   learned_beta=float(model.router.area.beta().detach().cpu()),
                   rounds=int(model.router.rounds),
                   router_edge_mode=str(model.router.edge_mode),
                   router_area_correction=bool(model.router.area.enabled),
                   strict_validation=bool(STRICT_VAL),
                   visible_stations=visible,
                   validation_stations=[node_ids[int(j)] for j in val_st],
                   initialization=("regional_finetune" if init_stage1_tag is not None
                                   else "scratch"),
                   stage1_tag=init_stage1_tag,
                   stage1_root=(
                       str(Path(stage1_root))
                       if stage1_root is not None
                       else str(C.REGIONAL_OUT)
                   ),
                   matched_router_initialization=bool(match_router_init)),
              open(out_dir / "norm_params.json", "w"), indent=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize encoder/head from the seed-matched Regional LSTM checkpoint.",
    )
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(1, 11))
    dfs = []
    for k in folds:
        stage1_tag = (
            f"fold_{k:02d}" if a.seed == C.SEED
            else f"fold_{k:02d}_seed{a.seed}"
        )
        dfs.append(
            run(
                fold=k,
                seed=a.seed,
                init_stage1_tag=stage1_tag if a.warm_start else None,
                out_root=FT_OUT if a.warm_start else OUT,
                match_router_init=a.warm_start,
            )
        )
    df = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    output_root = FT_OUT if a.warm_start else OUT
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "kfold_summary.csv", index=False)
    print(f"\n{'=' * 60}\nCoupled v2 (端到端) 汇总 (真值=观测)")
    for per in ("test", "full"):
        n = df[f"NSE_{per}"]
        print(f"[{per}] n={len(n)} NSE med={n.median():.3f} "
              f">0.5: {(n > 0.5).mean() * 100:.0f}%  <0: {(n < 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
