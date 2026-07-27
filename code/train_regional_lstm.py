# -*- coding: utf-8 -*-
"""
Regional LSTM (Rebuild_v2 第二步)

一个共享 LSTM, 输入 = 动态驱动(ERA5-Land 9 变量) + 静态属性, 目标 = 径流深 mm/month。
天然支持无资料预测: 留出站不参与训练/验证/归一化, 直接用其驱动+属性推理。

用法:
    python train_regional_lstm.py --fold 1          # 留出第 1 折的站 (协议同原论文折分组)
    python train_regional_lstm.py --holdout 6935450,6935340
    python train_regional_lstm.py --all-visible     # 不留出, 产出全部站预测(作新 Stage 1)
可选: --seed 42  --root <项目根>

输出 (Regional_LSTM/fold_XX 或 all_visible/):
    predictions_by_station/<sid>.csv   date, predicted_flow_m3s, observed_flow_m3s
    evaluation_holdout.csv / evaluation_visible.csv   (NSE/KGE/PBIAS/FHV/r, 统一测试期+全期)
    checkpoint.pth, norm_params.json, train_log.csv
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_forcing():
    z = np.load(C.FORCING_NPZ, allow_pickle=True)
    dates = pd.PeriodIndex(z["dates"], freq="M")
    return z["data"].astype(np.float32), dates, list(z["node_ids"]), list(z["features"])


def load_obs(node_ids, dates):
    """观测流量矩阵 [T,N] (m3/s); 混合日期格式/占位行/负值均视为缺测"""
    T, N = len(dates), len(node_ids)
    obs = np.full((T, N), np.nan, dtype=np.float32)
    for j, nid in enumerate(node_ids):
        f = C.GRDC_DIR / f"{nid}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        date_col = "data" if "data" in d.columns else d.columns[0]
        flow_col = "MEAN" if "MEAN" in d.columns else d.columns[-1]
        ts = pd.to_datetime(d[date_col], format="mixed", errors="coerce")
        flow = pd.to_numeric(d[flow_col], errors="coerce")
        keep = ts.notna() & (ts >= pd.Timestamp("1950-01-01"))
        s = pd.Series(flow[keep].values, index=pd.PeriodIndex(ts[keep], freq="M"))
        s = s[~s.index.duplicated(keep="last")]
        s[s < 0] = np.nan
        common = dates.intersection(s.index)
        obs[dates.get_indexer(common), j] = s.reindex(common).values
    return obs


def load_statics(node_ids):
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str}).set_index("node_id")
    nodes = nodes.reindex(node_ids)
    area = nodes["catchment_area_km2"].values.astype(np.float64)
    cols = []
    for c in C.STATIC_COLS:
        v = nodes[c].astype(np.float64)
        v = v.fillna(v.median())
        if c in C.LOG_STATIC:
            v = np.log1p(v.clip(lower=0))
        cols.append(v.values)
    return np.stack(cols, axis=1).astype(np.float32), area   # [N,S], [N]


def flow_to_runoff(q, area, dates):
    """m3/s -> mm/month (逐月精确天数)"""
    days = np.array([p.days_in_month for p in dates], dtype=np.float64)
    return q * 86.4 * days[:, None] / np.maximum(area[None, :], 1e-6)


def runoff_to_flow(r, area, dates):
    days = np.array([p.days_in_month for p in dates], dtype=np.float64)
    return r * np.maximum(area[None, :], 1e-6) / (86.4 * days[:, None])


# ----------------------------------------------------------------------
# 样本集: (station, t) -> (seq[L,Fd+S], y)
# ----------------------------------------------------------------------
class RegionalDataset(Dataset):
    def __init__(self, dyn, sta, y, pairs, L):
        self.dyn, self.sta, self.y, self.pairs, self.L = dyn, sta, y, pairs, L

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        j, t = self.pairs[i]
        seq = self.dyn[t - self.L + 1: t + 1, j, :]                 # [L,Fd]
        stat = np.repeat(self.sta[j][None, :], self.L, axis=0)      # [L,S]
        x = np.concatenate([seq, stat], axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.tensor(self.y[t, j], dtype=torch.float32)


def build_pairs(stations, t_range, dyn_ok, y_ok, L):
    pairs = []
    for j in stations:
        for t in t_range:
            if t - L + 1 < 0:
                continue
            if y_ok[t, j] and dyn_ok[t - L + 1: t + 1, j].all():
                pairs.append((j, t))
    return pairs


class RegionalLSTM(nn.Module):
    def __init__(self, in_dim, hidden=C.HIDDEN_DIM, layers=C.NUM_LAYERS, drop=C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def metrics(o, p):
    m = ~(np.isnan(o) | np.isnan(p))
    if m.sum() < 10:
        return None
    o, p = o[m], p[m]
    nse = 1 - ((o - p) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-12)
    r = float(np.corrcoef(o, p)[0, 1])
    beta = p.mean() / max(o.mean(), 1e-12)
    gamma = (p.std() / max(p.mean(), 1e-12)) / max(o.std() / max(o.mean(), 1e-12), 1e-12)
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    pbias = 100 * (p - o).sum() / max(o.sum(), 1e-12)
    thr = np.quantile(o, 0.9)
    pk = o >= thr
    fhv = 100 * (p[pk] - o[pk]).sum() / max(o[pk].sum(), 1e-12) if pk.sum() > 2 else np.nan
    return dict(NSE=nse, KGE=kge, PBIAS=pbias, FHV=fhv, r=r, n=int(m.sum()))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run(
    fold=None,
    holdout_ids=None,
    seed=C.SEED,
    tag=None,
    visible_ids=None,
    output_root=None,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dyn, dates, node_ids, feats = load_forcing()
    id2idx = {s: i for i, s in enumerate(node_ids)}
    obs = load_obs(node_ids, dates)
    sta_raw, area = load_statics(node_ids)

    # 研究窗口索引
    t_tr_end = dates.get_loc(pd.Period(C.TRAIN_END, "M"))
    t_val_end = dates.get_loc(pd.Period(C.VAL_END, "M"))
    t0 = dates.get_loc(pd.Period(C.STUDY_START, "M"))
    t1 = dates.get_loc(pd.Period(C.STUDY_END, "M"))
    tr_rng = range(t0, t_tr_end + 1)
    va_rng = range(t_tr_end + 1, t_val_end + 1)

    # 可训练站 = 固化折分组里的 182 站, 减去 QC 剔除名单(station_exclude.csv, 见 qc_stations.py)
    fold_df = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    excl_file = C.BASIN / "station_exclude.csv"
    excluded = set()
    if excl_file.exists():
        excluded = set(pd.read_csv(excl_file, dtype=str).iloc[:, 0])
        print(f"[QC] 剔除 {len(excluded)} 个数据错配站: {sorted(excluded)}")
    trainable = [s for s in fold_df["station_id"] if s in id2idx and s not in excluded]

    if fold is not None:
        holdout = [s for s in fold_df.loc[fold_df["fold"] == fold, "station_id"] if s in set(trainable)]
        tag = tag or (f"fold_{fold:02d}" if seed == C.SEED else f"fold_{fold:02d}_seed{seed}")
    elif holdout_ids:
        holdout = [s for s in holdout_ids if s in id2idx]
        tag = tag or "custom_holdout"
    else:
        holdout = []
        tag = tag or "all_visible"
    visible = [s for s in trainable if s not in set(holdout)]
    if visible_ids is not None:
        allowed = set(str(value) for value in visible_ids)
        visible = [s for s in visible if s in allowed]
    vis_idx = [id2idx[s] for s in visible]
    print(f"[{tag}] 可见 {len(visible)} 站 / 留出 {len(holdout)} 站 / device={device}")

    # 目标: 径流深 -> log1p -> z-score (统计只用 可见站 x 训练期)
    runoff = flow_to_runoff(obs, area, dates)
    y_log = np.log1p(np.clip(runoff, 0, None))
    tr_slice = y_log[t0:t_tr_end + 1][:, vis_idx]
    y_mu = float(np.nanmean(tr_slice)); y_sd = float(np.nanstd(tr_slice)) + 1e-8
    y_norm = ((y_log - y_mu) / y_sd).astype(np.float32)
    y_ok = ~np.isnan(y_norm)

    # 动态特征 z-score (统计只用 可见站 x 训练期)
    dyn_ok = ~np.isnan(dyn).any(axis=2)
    d_slice = dyn[t0:t_tr_end + 1][:, vis_idx, :]
    d_mu = np.nanmean(d_slice, axis=(0, 1)); d_sd = np.nanstd(d_slice, axis=(0, 1)) + 1e-8
    dyn_n = (dyn - d_mu) / d_sd

    # 静态特征 z-score (可见站统计) + 训练期气候均值两列
    s_mu = sta_raw[vis_idx].mean(axis=0); s_sd = sta_raw[vis_idx].std(axis=0) + 1e-8
    sta_n = (sta_raw - s_mu) / s_sd
    clim = np.nanmean(dyn_n[t0:t_tr_end + 1], axis=0)            # [N,Fd] 已归一化
    tp_i, t2_i = feats.index("tp"), feats.index("t2m")
    sta_n = np.concatenate([sta_n, clim[:, [tp_i, t2_i]]], axis=1).astype(np.float32)

    L = C.SEQ_LENGTH
    tr_pairs = build_pairs(vis_idx, tr_rng, dyn_ok, y_ok, L)
    va_pairs = build_pairs(vis_idx, va_rng, dyn_ok, y_ok, L)
    print(f"训练样本 {len(tr_pairs)}, 验证样本 {len(va_pairs)}")

    tr_ld = DataLoader(RegionalDataset(dyn_n, sta_n, y_norm, tr_pairs, L),
                       batch_size=C.BATCH_SIZE, shuffle=True, drop_last=False)
    va_ld = DataLoader(RegionalDataset(dyn_n, sta_n, y_norm, va_pairs, L),
                       batch_size=C.BATCH_SIZE, shuffle=False)

    model = RegionalLSTM(dyn_n.shape[2] + sta_n.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    lossf = nn.MSELoss()

    regional_root = Path(output_root) if output_root is not None else C.REGIONAL_OUT
    out_dir = regional_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_state, wait, log = np.inf, None, 0, []
    for ep in range(1, C.MAX_EPOCHS + 1):
        model.train(); tl = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(y)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va_ld:
                vl += lossf(model(x.to(device)), y.to(device)).item() * len(y)
        tl /= len(tr_pairs); vl /= max(len(va_pairs), 1)
        log.append(dict(epoch=ep, train=tl, val=vl))
        print(f"  epoch {ep:02d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best - 1e-5:
            best, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                print(f"  early stop @ {ep}"); break
    model.load_state_dict(best_state)
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(model.state_dict(), out_dir / "checkpoint.pth")

    # ------- 全节点推理 (含无观测 POI 节点; 有驱动+属性即可预测) -------
    model.eval()
    pred_n = np.full_like(y_norm, np.nan)
    all_idx = list(range(len(node_ids)))
    with torch.no_grad():
        for t in range(max(t0, L - 1), t1 + 1):
            ok = [j for j in all_idx if dyn_ok[t - L + 1: t + 1, j].all()]
            if not ok:
                continue
            seq = np.stack([np.concatenate(
                [dyn_n[t - L + 1: t + 1, j, :],
                 np.repeat(sta_n[j][None, :], L, axis=0)], axis=1) for j in ok])
            out = model(torch.from_numpy(seq).to(device)).cpu().numpy()
            pred_n[t, ok] = out
    pred_runoff = np.expm1(pred_n * y_sd + y_mu)
    pred_flow = runoff_to_flow(pred_runoff, area, dates)

    # ------- 保存与评估 (真值 = 观测; 统一测试期 + 全期) -------
    pdir = out_dir / "predictions_by_station"
    pdir.mkdir(exist_ok=True)
    rows_h, rows_v = [], []
    didx = dates.to_timestamp()
    hold_set = set(holdout)
    # 非训练站(POI/剔除站)只保存预测文件, 不参与评估
    for s in node_ids:
        if s in set(trainable):
            continue
        j = id2idx[s]
        sl = slice(t0, t1 + 1)
        pd.DataFrame({"date": didx[sl], "predicted_flow_m3s": pred_flow[sl, j],
                      "observed_flow_m3s": obs[sl, j]}).to_csv(pdir / f"{s}.csv", index=False)
    for s in trainable:
        j = id2idx[s]
        sl = slice(t0, t1 + 1)
        pd.DataFrame({"date": didx[sl], "predicted_flow_m3s": pred_flow[sl, j],
                      "observed_flow_m3s": obs[sl, j]}).to_csv(pdir / f"{s}.csv", index=False)
        te = slice(t_val_end + 1, t1 + 1)
        m_te = metrics(obs[te, j], pred_flow[te, j])
        m_fu = metrics(obs[sl, j], pred_flow[sl, j])
        if m_te is None or m_fu is None:
            continue
        row = dict(station_id=s, **{f"{k}_test": v for k, v in m_te.items()},
                   **{f"{k}_full": v for k, v in m_fu.items()})
        (rows_h if s in hold_set else rows_v).append(row)

    for name, rows in [("evaluation_holdout.csv", rows_h), ("evaluation_visible.csv", rows_v)]:
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / name, index=False)
            print(f"[{name[:-4]}] n={len(df)} | test NSE median={df.NSE_test.median():.3f} "
                  f"| full NSE median={df.NSE_full.median():.3f} "
                  f"| test FHV median={df.FHV_test.median():.1f}%")

    with open(out_dir / "norm_params.json", "w") as fp:
        json.dump(dict(y_mu=y_mu, y_sd=y_sd, d_mu=d_mu.tolist(), d_sd=d_sd.tolist(),
                       s_mu=s_mu.tolist(), s_sd=s_sd.tolist(), features=[str(x) for x in feats],
                       seq_length=L, seed=seed, holdout=holdout,
                       training_visible=visible,
                       output_root=str(regional_root)), fp, indent=1)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--holdout", default=None, help="逗号分隔的站 ID")
    ap.add_argument("--all-visible", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    hold = a.holdout.split(",") if a.holdout else None
    run(fold=a.fold, holdout_ids=hold, seed=a.seed)
