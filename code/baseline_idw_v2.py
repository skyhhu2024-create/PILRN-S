# -*- coding: utf-8 -*-
"""
经典基线 (协议 v2, 无训练): 邻站观测插值
  idw     : 沿河距离倒数加权 + DAR 面积缩放 (官方面积), 面积比过滤 [1/10, 10]
  nearest : 最近单 donor + DAR 缩放

与深度模型完全同折、同剔除、同真值(观测)、同评估时段。
用法:
    python baseline_idw_v2.py                 # 两种方法 x 10 折
    python baseline_idw_v2.py --folds 1,2
输出: Single_Prediction/Baseline_IDW_v2/{idw,nearest}_kfold_summary.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C
from train_regional_lstm import load_forcing, load_obs, load_statics, metrics

DAR_EXP = 0.8
IDW_POWER = 1.0
AREA_RATIO_MAX = 10.0
OUT = C.SP_V3 / "Baseline_IDW_v2"


def river_distance_matrix(node_ids):
    """Floyd-Warshall 沿河最短距离 (双向, km)"""
    id2idx = {s: i for i, s in enumerate(node_ids)}
    N = len(node_ids)
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0.0)
    edges = pd.read_csv(C.EDGES_CSV, dtype={"upstream_id": str, "downstream_id": str})
    for _, r in edges.iterrows():
        u, d = id2idx.get(r["upstream_id"]), id2idx.get(r["downstream_id"])
        if u is None or d is None:
            continue
        km = float(r["river_dist_km"])
        D[u, d] = min(D[u, d], km); D[d, u] = min(D[d, u], km)
    for k in range(N):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def predict_fold(method, targets, donors, obs, area, D):
    """targets/donors: 节点索引; 返回 pred [T, len(targets)]"""
    T = obs.shape[0]
    pred = np.full((T, len(targets)), np.nan)
    for ti, tgt in enumerate(targets):
        # 合格 donors: 有限河距 + 面积比过滤
        cand = [j for j in donors
                if np.isfinite(D[j, tgt])
                and 1.0 / AREA_RATIO_MAX <= area[tgt] / area[j] <= AREA_RATIO_MAX]
        if not cand:   # 放宽: 仅要求有限河距
            cand = [j for j in donors if np.isfinite(D[j, tgt])]
        if not cand:
            continue
        cand = np.array(cand)
        dar = (area[tgt] / area[cand]) ** DAR_EXP
        if method == "nearest":
            order = np.argsort(D[cand, tgt])
            for t in range(T):
                for k in order:                       # 最近的有观测 donor
                    q = obs[t, cand[k]]
                    if np.isfinite(q):
                        pred[t, ti] = q * dar[k]
                        break
        else:
            w = 1.0 / (D[cand, tgt] + 1.0) ** IDW_POWER
            qs = obs[:, cand] * dar[None, :]          # [T, n_cand]
            valid = np.isfinite(qs)
            wsum = (valid * w[None, :]).sum(axis=1)
            num = np.nansum(np.where(valid, qs * w[None, :], 0.0), axis=1)
            with np.errstate(invalid="ignore"):
                pred[:, ti] = np.where(wsum > 0, num / wsum, np.nan)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(1, 11))

    _, dates, node_ids, _ = load_forcing()
    id2idx = {s: i for i, s in enumerate(node_ids)}
    obs = load_obs(node_ids, dates)
    _, area = load_statics(node_ids)          # 官方校正面积
    D = river_distance_matrix(node_ids)
    print(f"河距矩阵完成, 有限距离对比例 {np.isfinite(D).mean()*100:.0f}%")

    t0 = dates.get_loc(pd.Period(C.STUDY_START, "M"))
    t1 = dates.get_loc(pd.Period(C.STUDY_END, "M"))
    t_va = dates.get_loc(pd.Period(C.VAL_END, "M"))

    fold_df = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    excl = set()
    f_ex = C.BASIN / "station_exclude.csv"
    if f_ex.exists():
        excl = set(pd.read_csv(f_ex, dtype=str).iloc[:, 0])
    trainable = [s for s in fold_df["station_id"] if s in id2idx and s not in excl]

    OUT.mkdir(parents=True, exist_ok=True)
    for method in ("idw", "nearest"):
        rows = []
        for k in folds:
            test_st = [s for s in fold_df.loc[fold_df["fold"] == k, "station_id"]
                       if s in set(trainable)]
            visible = [s for s in trainable if s not in set(test_st)]
            tgt_idx = [id2idx[s] for s in test_st]
            don_idx = [id2idx[s] for s in visible]
            pred = predict_fold(method, tgt_idx, don_idx, obs, area, D)
            fdir = OUT / method / f"fold_{k:02d}" / "predictions_by_station"
            fdir.mkdir(parents=True, exist_ok=True)
            didx = dates.to_timestamp()
            for ti, s in enumerate(test_st):
                j = id2idx[s]
                sl = slice(t0, t1 + 1); te = slice(t_va + 1, t1 + 1)
                pd.DataFrame({"date": didx[sl],
                              "predicted_flow_m3s": pred[sl, ti],
                              "observed_flow_m3s": obs[sl, j]}) \
                    .to_csv(fdir / f"{s}.csv", index=False)
                m_te = metrics(obs[te, j], pred[te, ti])
                m_fu = metrics(obs[sl, j], pred[sl, ti])
                if m_te and m_fu:
                    rows.append(dict(station_id=s, fold=k,
                                     **{f"{x}_test": v for x, v in m_te.items()},
                                     **{f"{x}_full": v for x, v in m_fu.items()}))
            print(f"[{method}] fold {k} 完成 ({len(test_st)} 站)")
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{method}_kfold_summary.csv", index=False)
        for per in ("test", "full"):
            n = df[f"NSE_{per}"]
            print(f"[{method} {per}] n={len(n)} NSE med={n.median():.3f} "
                  f">0.5: {(n > 0.5).mean()*100:.0f}%  <0: {(n < 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
