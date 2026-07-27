# -*- coding: utf-8 -*-
"""
随机森林区域化基线 (协议 v2, 非深度学习)

输入特征 = 当月 11 个驱动 + 滞后聚合(tp 近3/12月累计, t2m 近3月均值) + 12 静态属性
目标 = log 径流深 (与深度模型同口径); 同折留站评估, 真值=观测。
需要 scikit-learn (pip install scikit-learn)。

用法:
    python baseline_rf_v2.py                # 10 折
    python baseline_rf_v2.py --folds 1,2
输出: Single_Prediction/Baseline_RF_v2/kfold_summary.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C
from train_regional_lstm import (load_forcing, load_obs, load_statics,
                                 flow_to_runoff, runoff_to_flow, metrics)

OUT = C.SP_V3 / "Baseline_RF_v2"
N_TREES = 300
MIN_LEAF = 5


def build_features(dyn, sta, feats, t0):
    """[T,N,Fd] -> [T,N,Fd+3+S]: 当月驱动 + tp_sum3/tp_sum12/t2m_mean3 + 静态"""
    T, N, Fd = dyn.shape
    tp = dyn[:, :, feats.index("tp")]
    t2 = dyn[:, :, feats.index("t2m")]
    cs = np.nancumsum(np.nan_to_num(tp), axis=0)
    tp3 = np.full((T, N), np.nan); tp12 = np.full((T, N), np.nan)
    t23 = np.full((T, N), np.nan)
    for t in range(T):
        if t >= 2:
            tp3[t] = cs[t] - (cs[t - 3] if t >= 3 else 0)
            t23[t] = np.nanmean(t2[t - 2: t + 1], axis=0)
        if t >= 11:
            tp12[t] = cs[t] - (cs[t - 12] if t >= 12 else 0)
    S = np.repeat(sta[None, :, :], T, axis=0)
    X = np.concatenate([dyn, tp3[:, :, None], tp12[:, :, None], t23[:, :, None], S],
                       axis=2).astype(np.float32)
    return X


def main():
    from sklearn.ensemble import RandomForestRegressor

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(1, 11))

    dyn, dates, node_ids, feats = load_forcing()
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

    runoff = flow_to_runoff(obs, area, dates)
    y_log = np.log1p(np.clip(runoff, 0, None))
    X = build_features(dyn, sta_raw, feats, t0)
    print(f"特征张量: {X.shape}")

    rows = []
    for k in folds:
        test_st = [s for s in fold_df.loc[fold_df["fold"] == k, "station_id"]
                   if s in set(trainable)]
        visible = [s for s in trainable if s not in set(test_st)]
        vis_idx = [id2idx[s] for s in visible]
        tst_idx = [id2idx[s] for s in test_st]

        # 训练样本: 可见站 x 训练期
        Xtr, ytr = [], []
        for j in vis_idx:
            for t in range(t0, t_tr + 1):
                if np.isfinite(y_log[t, j]) and not np.isnan(X[t, j]).any():
                    Xtr.append(X[t, j]); ytr.append(y_log[t, j])
        rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=MIN_LEAF,
                                   n_jobs=-1, random_state=a.seed)
        rf.fit(np.array(Xtr), np.array(ytr))
        print(f"[RF fold {k}] 训练样本 {len(ytr)}")

        didx = dates.to_timestamp()
        fdir = OUT / f"fold_{k:02d}" / "predictions_by_station"
        fdir.mkdir(parents=True, exist_ok=True)
        for s in test_st:
            j = id2idx[s]
            Xte = X[t0:t1 + 1, j]
            ok = ~np.isnan(Xte).any(axis=1)
            pred_log = np.full(len(Xte), np.nan)
            if ok.sum():
                pred_log[ok] = rf.predict(Xte[ok])
            pr = np.expm1(pred_log)                      # mm/month
            pf = runoff_to_flow(pr[:, None], area[j:j + 1], dates[t0:t1 + 1])[:, 0]
            o = obs[t0:t1 + 1, j]
            pd.DataFrame({"date": didx[t0:t1 + 1], "predicted_flow_m3s": pf,
                          "observed_flow_m3s": o}).to_csv(fdir / f"{s}.csv", index=False)
            n_te = t_va + 1 - t0
            m_te = metrics(o[n_te:], pf[n_te:])
            m_fu = metrics(o, pf)
            if m_te and m_fu:
                rows.append(dict(station_id=s, fold=k,
                                 **{f"{x}_test": v for x, v in m_te.items()},
                                 **{f"{x}_full": v for x, v in m_fu.items()}))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "kfold_summary.csv", index=False)
    for per in ("test", "full"):
        n = df[f"NSE_{per}"]
        print(f"[RF {per}] n={len(n)} NSE med={n.median():.3f} "
              f">0.5: {(n > 0.5).mean() * 100:.0f}%  <0: {(n < 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
