# -*- coding: utf-8 -*-
"""
PILRN v2: 物理引导可学习路由 (Stage 2, 协议 v2)

与原实现的关键区别:
  1. 训练目标 = 观测径流(log 径流深空间), 不再是留出站自己的 LSTM 预测
  2. fold 测试站全程隔离: 不进 loss、不参与早停、观测通道置零
  3. 训练用"动态随机 mask": 每个 batch 从可见站中随机抽 15% 作临时无资料目标
  4. Stage 1 = 对应 fold 的 regional LSTM (fold 模型未见过测试站 -> 无泄露);
     目标站自己的 regional 预测是合法输入(无需局地观测), 路由学习"邻站观测修正局地先验"
  5. 边特征基于官方校正面积 (graph_v2.py)
  6. 评估与保存来自同一次 masked 推理, 真值=观测

用法:
    python train_pilrn_v2.py --fold 1
    python run_kfold_pilrn.py            # 全部 10 折
输出: Single_Prediction/PILRN_v2/fold_XX/
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

import torch
import torch.nn as nn

# ---------------- Stage 2 超参 ----------------
HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 150
PATIENCE = 20
BATCH_T = 64            # 每 batch 时间步数
DYN_MASK_RATIO = 0.15   # 每 batch 动态 mask 的可见站比例
SPATIAL_VAL_RATIO = 0.15
PHYSICAL_VERSION = "physical_v4"
SIG_ROUNDS = 2          # PILRNSignal 传播轮数 (消融: 1/2/3)
STRICT_VAL = True       # 空间验证站在训练全程不作 donor
PILRN_OUT = C.SP_V3 / "PILRN_physical_v4"


# ---------------- 模型 ----------------
def softmax_per_target(pos, dst, n_nodes):
    """per-target softmax; pos>=0 (Softplus 输出), 全局减 max 保证稳定"""
    e = torch.exp(pos - pos.max())
    denom = torch.zeros(n_nodes, device=pos.device).index_add_(0, dst, e)
    return e / (denom[dst] + 1e-12)


class EdgeWeightNet(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus())

    def forward(self, ef):
        return self.mlp(ef).squeeze(-1)


class GlobalAreaExponent(nn.Module):
    """One bounded generalized-DAR exponent shared by all directed edges."""

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            self.theta = nn.Parameter(torch.tensor(0.0))  # beta=1 at initialization

    def beta(self):
        if not self.enabled:
            return torch.tensor(1.0)
        return 0.7 + 0.6 * torch.sigmoid(self.theta)

    def forward(self, log_ratio):
        beta = self.beta().to(log_ratio.device)
        return torch.exp((beta - 1.0) * log_ratio).squeeze(-1)


class PhysicsRoutingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src, dst, alpha, scale):
        msg = x[src] * (alpha * scale).unsqueeze(-1)          # [E,dim]
        agg = torch.zeros_like(x).index_add_(0, dst, msg)     # [N,dim]
        g = torch.sigmoid(self.gate(torch.cat([x, agg], dim=-1)))
        return self.norm(g * x + (1 - g) * agg)


class PILRN(nn.Module):
    """Matched embedding-level control; edge weights act on latent states."""

    def __init__(self, in_dim, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(DROPOUT))
        self.ewn = EdgeWeightNet()
        self.layers = nn.ModuleList([PhysicsRoutingLayer(hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                     nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, x, src, dst, edge_feat, log_area_ratio, n_nodes):
        alpha = softmax_per_target(self.ewn(edge_feat), dst, n_nodes)
        scale = torch.ones_like(log_area_ratio).squeeze(-1)
        h = self.proj(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, src, dst, alpha, scale)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return self.readout(h).squeeze(-1), alpha, scale


class GATRoutingLayer(nn.Module):
    """点积/加性注意力版路由层: 权重来自节点嵌入, 不用物理边特征 (基线)"""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(dim, dim * heads, bias=False)
        self.att = nn.Parameter(torch.empty(heads, 2 * dim))
        nn.init.xavier_uniform_(self.att)
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src, dst, n_nodes):
        N, dim = x.shape
        Wh = self.W(x).view(N, self.heads, dim)                    # [N,H,dim]
        cat = torch.cat([Wh[src], Wh[dst]], dim=-1)                # [E,H,2dim]
        e = torch.nn.functional.leaky_relu((cat * self.att).sum(-1), 0.2)  # [E,H]
        scores = torch.exp(e - e.max(dim=0, keepdim=True).values)
        denom = torch.zeros(n_nodes, self.heads, device=x.device).index_add_(
            0, dst, scores
        )
        attention = scores / (denom[dst] + 1e-12)
        messages = Wh[src] * attention.unsqueeze(-1)
        agg = torch.zeros(
            n_nodes, self.heads, dim, device=x.device
        ).index_add_(0, dst, messages).mean(dim=1)
        g = torch.sigmoid(self.gate(torch.cat([x, agg], dim=-1)))
        return self.norm(g * x + (1 - g) * agg)


class GATNet(nn.Module):
    """Decoupled GAT 基线: 与 PILRN 相同输入/门控/读出, 仅权重机制不同"""
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(DROPOUT))
        self.layers = nn.ModuleList([GATRoutingLayer(hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                     nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, x, src, dst, edge_feat, log_area_ratio, n_nodes):
        h = self.proj(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, src, dst, n_nodes)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        E = src.shape[0]
        one = torch.ones(E, device=x.device)
        return self.readout(h).squeeze(-1), one / n_nodes, one


class PILRNSignal(nn.Module):
    """Signal-level routing whose amplitude operations occur in runoff-depth space."""

    def __init__(
        self,
        s_dim,
        y_mu,
        y_sd,
        rounds=2,
        edge_mode="full",
        area_correction=True,
    ):
        super().__init__()
        if edge_mode not in {"full", "uniform"}:
            raise ValueError(f"Unknown edge_mode={edge_mode}")
        self.edge_mode = edge_mode
        self.ewn = EdgeWeightNet() if edge_mode == "full" else None
        self.area = GlobalAreaExponent(enabled=area_correction)
        self.gate = nn.Sequential(nn.Linear(s_dim + 3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.rounds = int(rounds)
        self.register_buffer("y_mu", torch.tensor(float(y_mu), dtype=torch.float32))
        self.register_buffer("y_sd", torch.tensor(float(y_sd), dtype=torch.float32))

    def to_runoff(self, z):
        log_runoff = torch.clamp(z * self.y_sd + self.y_mu, min=0.0, max=15.0)
        return torch.expm1(log_runoff)

    def to_normalized_log(self, runoff):
        return (torch.log1p(runoff.clamp_min(0.0)) - self.y_mu) / self.y_sd

    def forward(self, obs, vflag, reg, regflag, sta, src, dst, ef, lar, n):
        if self.edge_mode == "full":
            w = self.ewn(ef)
        else:
            w = torch.ones(src.shape[0], device=obs.device)
        sc = self.area(lar)

        obs_r = self.to_runoff(obs)
        reg_r = torch.where(regflag > 0, self.to_runoff(reg), torch.zeros_like(reg))
        s = torch.where(vflag > 0, obs_r, reg_r)
        a = (vflag > 0).float()
        alpha_last = torch.zeros_like(w)
        for _ in range(self.rounds):
            contrib = w * a[src]
            denom = torch.zeros(n, device=s.device).index_add_(0, dst, contrib)
            num = torch.zeros(n, device=s.device).index_add_(0, dst, contrib * sc * s[src])
            n_est = num / (denom + 1e-8)
            has = (denom > 1e-8).float()
            alpha_last = contrib / (denom[dst] + 1e-12)
            s_z = self.to_normalized_log(s)
            n_z = self.to_normalized_log(n_est)
            g = torch.sigmoid(self.gate(torch.cat(
                [sta, a.unsqueeze(1), s_z.unsqueeze(1), n_z.unsqueeze(1)], dim=1))).squeeze(-1)
            fused = g * s + (1 - g) * n_est
            updated = torch.where(has > 0, fused, s)
            s = torch.where(vflag > 0, obs_r, updated)
            a = torch.maximum(a, has)
        return self.to_normalized_log(s), alpha_last, sc


VARIANT_OUT = {
    "pilrn": C.SP_V3 / "PILRN_embed_matched_physical_v4",
    "pilrn_sig": C.SP_V3 / "PILRN_sig_physical_v4",
    "pilrn_sig_no_area": C.SP_V3 / "PILRN_sig_no_area_physical_v4",
    "pilrn_sig_topology": C.SP_V3 / "PILRN_sig_topology_physical_v4",
    "pilrn_sig_shuffled": C.SP_V3 / "PILRN_sig_shuffled_physical_v4",
    "pilrn_sig_r1": C.SP_V3 / "PILRN_sig_r1_physical_v4",
    "pilrn_sig_r3": C.SP_V3 / "PILRN_sig_r3_physical_v4",
    "gat": C.SP_V3 / "GAT_physical_comparator_v4",
}


# ---------------- 主流程 ----------------
def run(
    fold,
    seed=C.SEED,
    stage1_tag=None,
    tag=None,
    variant="pilrn",
    visible_frac=1.0,
    visible_ids=None,
    stage1_root=None,
    output_root=None,
    rounds_override=None,
):
    """Run a declared embedding, attention, or physical signal-routing variant.
    visible_frac<1: 站网密度敏感性实验 —— 从可见站中抽取该比例作为 donor/训练目标,
    其余站完全隐藏(非donor非目标非验证); 抽样只依赖 fold, 各 variant 用同一抽样。
    Stage-1 先验保持全可见 fold 模型不变, 以隔离路由阶段的密度效应。"""
    assert variant in VARIANT_OUT
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

    # ---- 站点集合(与 regional 完全一致的折与剔除) ----
    fold_df = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    excl = set()
    excl_file = C.BASIN / "station_exclude.csv"
    if excl_file.exists():
        excl = set(pd.read_csv(excl_file, dtype=str).iloc[:, 0])
    trainable = [s for s in fold_df["station_id"] if s in id2idx and s not in excl]
    test_st = [s for s in fold_df.loc[fold_df["fold"] == fold, "station_id"]
               if s in set(trainable)]
    visible = [s for s in trainable if s not in set(test_st)]
    if visible_ids is not None:
        allowed = set(str(value) for value in visible_ids)
        visible = [s for s in visible if s in allowed]
    elif visible_frac < 1.0:
        rng_d = np.random.default_rng(9000 + fold)      # 只依赖 fold -> sig/GAT 同抽样
        kd = max(5, int(round(len(visible) * visible_frac)))
        keep = sorted(rng_d.choice(len(visible), size=kd, replace=False))
        visible = [visible[i] for i in keep]
        print(f"  [密度敏感性] visible_frac={visible_frac:.2f} -> {len(visible)} 站")
    vis_idx = np.array([id2idx[s] for s in visible])
    tst_idx = np.array([id2idx[s] for s in test_st])
    # donor 只允许可见站: 剔除站/未入折站/测试站的观测一律不进入输入
    vis_mask = np.zeros(N, dtype=bool)
    vis_mask[vis_idx] = True
    rng = np.random.default_rng(seed)
    val_st = rng.choice(
        vis_idx, size=max(1, int(len(vis_idx) * SPATIAL_VAL_RATIO)), replace=False
    )
    train_donor = np.array([j for j in vis_idx if j not in set(val_st)])
    if STRICT_VAL:
        vis_mask[val_st] = False
        print(f"  [STRICT_VAL] 验证站 ({len(val_st)}) 训练全程隐藏")
    base_tag = f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}"
    if visible_frac < 1.0:
        base_tag += f"_d{int(round(visible_frac * 100))}"
    tag = tag or base_tag
    print(f"[{tag}] 可见 {len(visible)} / 测试 {len(test_st)} / device={device}")

    # ---- Stage 1: 对应 fold 的 regional LSTM 预测 (优先同 seed 的运行) ----
    stage1_root = Path(stage1_root) if stage1_root is not None else C.REGIONAL_OUT
    if stage1_tag is None:
        stage1_tag = base_tag if (stage1_root / base_tag).exists() else f"fold_{fold:02d}"
    s1_dir = stage1_root / stage1_tag / "predictions_by_station"
    if not s1_dir.exists():
        sys.exit(f"[Error] 缺少 Stage 1 预测 {s1_dir}, 先运行 run_kfold_regional.py")
    reg_flow = np.full((len(dates), N), np.nan, dtype=np.float64)
    for f in s1_dir.glob("*.csv"):
        j = id2idx.get(f.stem)
        if j is None:
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        idx = dates.get_indexer(pd.PeriodIndex(d["date"], freq="M"))
        ok = idx >= 0
        reg_flow[idx[ok], j] = d["predicted_flow_m3s"].values[ok]

    # ---- 归一化 (log 径流深空间, 统计只用 可见站 x 训练期) ----
    runoff = flow_to_runoff(obs, area, dates)
    y_log = np.log1p(np.clip(runoff, 0, None))
    # Reuse the full outer-training normalization convention of the frozen
    # Stage-1 model; the internal validation stations remain hidden as donors.
    norm_idx = vis_idx
    sl = y_log[t0:t_tr + 1][:, norm_idx]
    y_mu = float(np.nanmean(sl)); y_sd = float(np.nanstd(sl)) + 1e-8
    y_norm = ((y_log - y_mu) / y_sd).astype(np.float32)
    y_ok = np.isfinite(y_norm)

    reg_runoff = flow_to_runoff(reg_flow, area, dates)
    reg_norm = ((np.log1p(np.clip(reg_runoff, 0, None)) - y_mu) / y_sd).astype(np.float32)
    reg_norm = np.nan_to_num(reg_norm, nan=0.0)
    reg_flag = np.isfinite(reg_runoff).astype(np.float32)

    # 静态特征 + 气候均值 (与 regional 相同)
    s_mu = sta_raw[norm_idx].mean(axis=0); s_sd = sta_raw[norm_idx].std(axis=0) + 1e-8
    sta_n = (sta_raw - s_mu) / s_sd
    tp_i, t2_i = feats.index("tp"), feats.index("t2m")
    d_slice = dyn[t0:t_tr + 1][:, norm_idx, :]
    d_mu = np.nanmean(d_slice, axis=(0, 1)); d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
    dyn_n = np.nan_to_num((dyn - d_mu) / d_sd, nan=0.0).astype(np.float32)

    # ---- 图 + 边特征 (官方面积) ----
    clim_tp = np.nanmean(dyn[t0:t_tr + 1, :, tp_i], axis=0)
    clim_t2 = np.nanmean(dyn[t0:t_tr + 1, :, t2_i], axis=0)
    edge_index, edge_feat, area_ratio, river_dist, hc_w, feat_stats = \
        graph_v2.build_graph(node_ids, np.nan_to_num(clim_tp), np.nan_to_num(clim_t2))
    E = edge_index.shape[1]
    active_n = int(np.unique(edge_index).size)
    print(
        f"图: {active_n} 个活跃路由节点 / {N} 条节点记录, "
        f"{E} 条有向边"
    )

    src = torch.from_numpy(edge_index[0]).to(device)
    dst = torch.from_numpy(edge_index[1]).to(device)
    ef = torch.from_numpy(edge_feat).to(device)
    if variant == "pilrn_sig_shuffled":
        shuffle_rng = np.random.default_rng(47000 + int(fold))
        permutation = torch.from_numpy(shuffle_rng.permutation(E)).to(device)
        ef_router = ef[permutation]
    else:
        ef_router = ef
    lar = torch.from_numpy(np.log(area_ratio)).to(device)

    sta_t = torch.from_numpy(sta_n.astype(np.float32)).to(device)   # [N,S]
    S = sta_n.shape[1]
    use_res = (variant == "pilrn_res")
    n_ch = 6 if use_res else 5

    def build_x(t, hidden_idx):
        """节点特征: reg_pred, reg_flag, obs(仅可见donor), vis_flag, tp [, 残差] + t2m + statics"""
        x = np.zeros((N, n_ch), dtype=np.float32)
        x[:, 0] = reg_norm[t]
        x[:, 1] = reg_flag[t]
        ok = y_ok[t] & vis_mask
        o = np.where(ok, y_norm[t], 0.0).copy()
        vflag = ok.astype(np.float32).copy()
        o[hidden_idx] = 0.0; vflag[hidden_idx] = 0.0
        x[:, 2] = o; x[:, 3] = vflag
        x[:, 4] = dyn_n[t, :, tp_i]
        if use_res:
            x[:, 5] = (o - reg_norm[t]) * vflag * reg_flag[t]   # 可见站残差(邻站修正信号)
        xt = torch.from_numpy(x).to(device)
        return torch.cat([xt, dyn_n_t2[t].unsqueeze(-1), sta_t], dim=-1)

    dyn_n_t2 = torch.from_numpy(dyn_n[:, :, t2_i]).to(device)
    reg_norm_dev = torch.from_numpy(reg_norm).to(device)
    reg_flag_dev = torch.from_numpy(reg_flag).to(device)
    y_norm_np = y_norm; y_ok_np = y_ok

    signal_variants = {
        "pilrn_sig",
        "pilrn_sig_no_area",
        "pilrn_sig_topology",
        "pilrn_sig_shuffled",
        "pilrn_sig_r1",
        "pilrn_sig_r3",
    }
    if variant in signal_variants:
        declared_rounds = {"pilrn_sig_r1": 1, "pilrn_sig_r3": 3}.get(
            variant, SIG_ROUNDS
        )
        rounds = (
            int(rounds_override)
            if rounds_override is not None
            else declared_rounds
        )
        if rounds < 1:
            raise ValueError(f"Signal-routing rounds must be >= 1, got {rounds}.")
        edge_mode = "uniform" if variant == "pilrn_sig_topology" else "full"
        area_correction = variant not in {
            "pilrn_sig_no_area",
            "pilrn_sig_topology",
        }
        model = PILRNSignal(
            S,
            y_mu=y_mu,
            y_sd=y_sd,
            rounds=rounds,
            edge_mode=edge_mode,
            area_correction=area_correction,
        ).to(device)

        def forward_t(t, hidden_idx):
            ok = y_ok_np[t] & vis_mask
            o = np.where(ok, y_norm_np[t], 0.0).copy()
            vf = ok.astype(np.float32).copy()
            o[hidden_idx] = 0.0; vf[hidden_idx] = 0.0
            return model(torch.from_numpy(o).to(device), torch.from_numpy(vf).to(device),
                         reg_norm_dev[t], reg_flag_dev[t], sta_t, src, dst,
                         ef_router, lar, N)

        signal_batch_graph_cache = {}

        def forward_signal_batch(times, hidden_idx):
            batch_size = len(times)
            obs_rows = []
            flag_rows = []
            for t in times:
                ok = y_ok_np[t] & vis_mask
                obs_row = np.where(ok, y_norm_np[t], 0.0).copy()
                flag_row = ok.astype(np.float32).copy()
                obs_row[hidden_idx] = 0.0
                flag_row[hidden_idx] = 0.0
                obs_rows.append(obs_row)
                flag_rows.append(flag_row)
            if batch_size not in signal_batch_graph_cache:
                offsets = (
                    torch.arange(batch_size, device=device, dtype=src.dtype) * N
                ).repeat_interleave(E)
                signal_batch_graph_cache[batch_size] = (
                    src.repeat(batch_size) + offsets,
                    dst.repeat(batch_size) + offsets,
                    ef_router.repeat(batch_size, 1),
                    lar.repeat(batch_size, 1),
                )
            src_b, dst_b, ef_b, lar_b = signal_batch_graph_cache[batch_size]
            pred, alpha, scale = model(
                torch.from_numpy(np.concatenate(obs_rows)).to(device),
                torch.from_numpy(np.concatenate(flag_rows)).to(device),
                reg_norm_dev[times].reshape(-1),
                reg_flag_dev[times].reshape(-1),
                sta_t.repeat(batch_size, 1),
                src_b,
                dst_b,
                ef_b,
                lar_b,
                batch_size * N,
            )
            return pred.view(batch_size, N), alpha, scale
    else:
        net_cls = GATNet if variant == "gat" else PILRN
        model = net_cls(n_ch + 1 + S).to(device)   # +1: t2m 列

        def forward_t(t, hidden_idx):
            pred, alpha, scale = model(
                build_x(t, hidden_idx), src, dst, ef_router, lar, N
            )
            if use_res:
                pred = pred + reg_norm_dev[t]      # 残差路由: regional 先验 + 修正
            return pred, alpha, scale

        batch_graph_cache = {}

        def forward_embedding_batch(times, hidden_idx):
            batch_size = len(times)
            x = torch.cat([build_x(t, hidden_idx) for t in times], dim=0)
            if batch_size not in batch_graph_cache:
                offsets = (
                    torch.arange(batch_size, device=device, dtype=src.dtype) * N
                ).repeat_interleave(E)
                batch_graph_cache[batch_size] = (
                    src.repeat(batch_size) + offsets,
                    dst.repeat(batch_size) + offsets,
                    ef_router.repeat(batch_size, 1),
                    lar.repeat(batch_size, 1),
                )
            src_b, dst_b, ef_b, lar_b = batch_graph_cache[batch_size]
            pred, alpha, scale = model(
                x, src_b, dst_b, ef_b, lar_b, batch_size * N
            )
            return pred.view(batch_size, N), alpha, scale
    nparam = sum(p.numel() for p in model.parameters())
    print(f"PILRN v2 参数量: {nparam}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    tr_times = [t for t in range(t0, t_tr + 1)]
    va_times = [t for t in range(t_tr + 1, t_va + 1)]

    def epoch_pass(times, epoch, train=True, hidden_eval=None):
        """train=True: 动态 mask + 反传; False: 固定 hidden_eval 站为目标算 val loss"""
        model.train(train)
        total, count = 0.0, 0
        batches = [times[i:i + BATCH_T] for i in range(0, len(times), BATCH_T)]
        for bt in batches:
            if train:
                k = max(1, int(len(train_donor) * DYN_MASK_RATIO))
                targets = rng.choice(train_donor, size=k, replace=False)
            else:
                targets = hidden_eval
            hidden = np.concatenate([targets, tst_idx]) if len(tst_idx) else targets
            if variant == "gat" or variant == "pilrn":
                valid_times = []
                valid_targets = []
                for t in bt:
                    tm = targets[y_ok[t, targets]]
                    if len(tm):
                        valid_times.append(t)
                        valid_targets.append(tm)
                if not valid_times:
                    continue
                pred_batch, _, _ = forward_embedding_batch(valid_times, hidden)
                losses = []
                for row, (t, tm) in enumerate(zip(valid_times, valid_targets)):
                    yt = torch.from_numpy(y_norm[t, tm]).to(device)
                    losses.append(((pred_batch[row, tm] - yt) ** 2).mean())
                loss_b = torch.stack(losses).mean()
                nb = len(losses)
                if train:
                    opt.zero_grad(); loss_b.backward(); opt.step()
                total += loss_b.item() * nb; count += nb
                continue
            if variant in signal_variants:
                valid_times = []
                valid_targets = []
                for t in bt:
                    tm = targets[y_ok[t, targets]]
                    if len(tm):
                        valid_times.append(t)
                        valid_targets.append(tm)
                if not valid_times:
                    continue
                pred_batch, _, _ = forward_signal_batch(valid_times, hidden)
                losses = []
                for row, (t, tm) in enumerate(zip(valid_times, valid_targets)):
                    yt = torch.from_numpy(y_norm[t, tm]).to(device)
                    losses.append(((pred_batch[row, tm] - yt) ** 2).mean())
                loss_b = torch.stack(losses).mean()
                nb = len(losses)
                if train:
                    opt.zero_grad(); loss_b.backward(); opt.step()
                total += loss_b.item() * nb; count += nb
                continue
            loss_b = torch.tensor(0.0, device=device)
            nb = 0
            for t in bt:
                tm = targets[y_ok[t, targets]]
                if len(tm) == 0:
                    continue
                pred, alpha, scale = forward_t(t, hidden)
                yt = torch.from_numpy(y_norm[t, tm]).to(device)
                mse = ((pred[tm] - yt) ** 2).mean()
                loss_b = loss_b + mse
                nb += 1
            if nb == 0:
                continue
            loss_b = loss_b / nb
            if train:
                opt.zero_grad(); loss_b.backward(); opt.step()
            total += loss_b.item() * nb; count += nb
        return total / max(count, 1)

    variant_root = Path(output_root) if output_root is not None else VARIANT_OUT[variant]
    out_dir = variant_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_state, wait, log = np.inf, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        tl = epoch_pass(tr_times, ep, train=True)
        with torch.no_grad():
            vl = epoch_pass(va_times, ep, train=False, hidden_eval=val_st)
        log.append(dict(epoch=ep, train=tl, val=vl))
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:03d}  train {tl:.4f}  val {vl:.4f}")
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

    # ---- masked 推理 (测试站隐藏观测) + 评估 ----
    model.eval()
    pred_n = np.full((len(dates), N), np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(t0, t1 + 1):
            p, alpha, scale = forward_t(t, tst_idx)
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
            rows.append(dict(station_id=s,
                             **{f"{k}_test": v for k, v in m_te.items()},
                             **{f"{k}_full": v for k, v in m_fu.items()}))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "evaluation_holdout.csv", index=False)
        print(f"[holdout] n={len(df)} | test NSE median={df.NSE_test.median():.3f} "
              f"| full NSE median={df.NSE_full.median():.3f} "
              f"| PBIAS med={df.PBIAS_test.median():.1f}%")

    # 边权重存档 (可解释性分析素材)
    with torch.no_grad():
        _, alpha, scale = forward_t(t_va, tst_idx)
    pd.DataFrame({
        "src": [node_ids[i] for i in edge_index[0]],
        "dst": [node_ids[i] for i in edge_index[1]],
        "river_dist_km": river_dist, "area_ratio": area_ratio[:, 0],
        "alpha": alpha.cpu().numpy(), "area_scale": scale.cpu().numpy(),
        "hc_weight": hc_w,
    }).to_csv(out_dir / "edge_weights.csv", index=False)

    learned_beta = None
    if isinstance(model, PILRNSignal):
        learned_beta = float(model.area.beta().detach().cpu())
    json.dump(dict(
                   protocol=PHYSICAL_VERSION,
                   y_mu=y_mu, y_sd=y_sd, seed=seed, fold=fold, n_param=int(nparam),
                   variant=variant, rounds=int(getattr(model, "rounds", N_LAYERS)),
                   best_validation_loss=float(best),
                   learned_beta=learned_beta,
                   strict_validation=bool(STRICT_VAL),
                   visible_stations=visible,
                   validation_stations=[node_ids[int(j)] for j in val_st],
                   stage1_root=str(stage1_root),
                   stage1_tag=str(stage1_tag),
                   test_stations=test_st, feat_stats=feat_stats),
              open(out_dir / "norm_params.json", "w"), indent=1)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", default="pilrn", choices=list(VARIANT_OUT.keys()))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--visible-frac", type=float, default=1.0)
    ap.add_argument("--strict-val", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--output-root", default=None)
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    if a.strict_val:
        STRICT_VAL = True
    run(
        fold=a.fold,
        seed=a.seed,
        variant=a.variant,
        visible_frac=a.visible_frac,
        output_root=a.output_root,
        rounds_override=a.rounds,
    )
