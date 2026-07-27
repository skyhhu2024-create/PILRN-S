# -*- coding: utf-8 -*-
"""
河网图 + 8 维物理边特征 (Rebuild_v2, 基于官方校正后的节点属性)

与原论文的 8 维边特征定义一致, 但全部基于 river_network_nodes_v2.csv 的
GRDC 官方面积/高程重新计算(原版边特征被错误面积污染)。

边特征 (v_j -> v_i):
  f1 log 集水面积比        log(Ac_j / Ac_i)
  f2 log 局地面积比        log(Al_j / Al_i)
  f3 归一化高程差          (h_j - h_i) / std
  f4 归一化坡度差          (s_j - s_i) / std
  f5 倒数河距              1 / (d_ji + 1)
  f6 归一化 log 河距       log(d_ji + 1) / std
  f7 降水相似度            1 - |tp_j - tp_i| / range
  f8 温度相似度            1 - |t2m_j - t2m_i| / range
f1-f6 逐维 z-score; f7/f8 已在 [0,1]。
"""
import numpy as np
import pandas as pd

import config_v2 as C


def build_graph(node_ids, clim_mean_tp, clim_mean_t2m):
    """
    Args:
        node_ids: 节点顺序 (与 forcing npz 一致)
        clim_mean_tp / clim_mean_t2m: [N] 训练期气候均值 (物理单位)
    Returns:
        edge_index [2,E] int64 (双向, E=2*边数), src->dst
        edge_feat  [E,8] float32 (f1-f6 z-scored)
        area_ratio [E,1] float32 (A_dst / A_src, AreaScalingNet 输入)
        river_dist [E]   float32 (km)
        hc_weight  [E]   float32 手工先验权重 (per-target 归一化, Eq.11)
        feat_stats dict  (z-score 常数, 存档用)
    """
    id2idx = {s: i for i, s in enumerate(node_ids)}
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str}).set_index("node_id") \
              .reindex(node_ids)
    edges = pd.read_csv(C.EDGES_CSV, dtype={"upstream_id": str, "downstream_id": str})

    Ac = nodes["catchment_area_km2"].astype(float).copy()
    Al = nodes["local_area_km2"].astype(float).copy()
    h = nodes["altitude"].astype(float).copy()
    s = nodes["slope"].astype(float).copy()
    for v in (Ac, Al, h, s):
        v.fillna(v.median(), inplace=True)
    Ac = np.maximum(Ac.values, 1e-3)
    Al = np.maximum(Al.values, 1e-3)
    h, s = h.values, s.values

    src, dst, dist = [], [], []
    for _, r in edges.iterrows():
        u, d = id2idx.get(r["upstream_id"]), id2idx.get(r["downstream_id"])
        if u is None or d is None:
            continue
        km = float(r["river_dist_km"])
        src += [u, d]; dst += [d, u]; dist += [km, km]      # 双向
    src = np.array(src); dst = np.array(dst); dist = np.array(dist, dtype=np.float64)
    E = len(src)

    f = np.zeros((E, 8), dtype=np.float64)
    f[:, 0] = np.log(Ac[src] / Ac[dst])
    f[:, 1] = np.log(Al[src] / Al[dst])
    f[:, 2] = h[src] - h[dst]
    f[:, 3] = s[src] - s[dst]
    f[:, 4] = 1.0 / (dist + 1.0)
    f[:, 5] = np.log(dist + 1.0)
    dtp = np.abs(clim_mean_tp[src] - clim_mean_tp[dst])
    dt2 = np.abs(clim_mean_t2m[src] - clim_mean_t2m[dst])
    f[:, 6] = 1.0 - dtp / max(dtp.max(), 1e-8)
    f[:, 7] = 1.0 - dt2 / max(dt2.max(), 1e-8)

    mu = f[:, :6].mean(axis=0); sd = f[:, :6].std(axis=0) + 1e-8
    f[:, :6] = (f[:, :6] - mu) / sd

    area_ratio = (Ac[dst] / Ac[src]).reshape(-1, 1)

    # 手工先验 (Eq.11): 1/(d+1) * log(1+A_src)/log(1+A_max), per-target softmax 归一化
    raw = (1.0 / (dist + 1.0)) * (np.log1p(Ac[src]) / np.log1p(Ac.max()))
    hc = np.zeros(E)
    for t in np.unique(dst):
        m = dst == t
        hc[m] = raw[m] / max(raw[m].sum(), 1e-12)

    stats = dict(f16_mu=mu.tolist(), f16_sd=sd.tolist(),
                 dtp_max=float(dtp.max()), dt2_max=float(dt2.max()))
    return (np.stack([src, dst]).astype(np.int64), f.astype(np.float32),
            area_ratio.astype(np.float32), dist.astype(np.float32),
            hc.astype(np.float32), stats)
