# -*- coding: utf-8 -*-
"""
站点-流域属性一致性 QC (Rebuild_v2 Step 0.5)

检查每个测站的观测流量与其挂接的 HydroRIVERS 河段/流域属性是否物理自洽:
  flag_spec_high : 比流量 > 100 l/s/km2  -> 观测远超该面积产流能力(面积偏小/挂错大流量站)
  flag_spec_low  : 比流量 < 2 l/s/km2    -> 观测远低于该面积(微型研究流域被挂到大河段/引水渠)
  flag_hyriv     : 观测均流量 与 HydroRIVERS 模拟流量 之比 >8 或 <1/8

已确认的典型错例(2026-07 核查):
  6435060 LOBITH   莱茵出口(~160,000 km2) 被挂到 270 km2 支流
  6335601 LAUFFEN  内卡河(~7,600 km2) 被挂到 117 km2
  6935450/6935340/6935345/6935318/6935315 瑞士微型研究流域(几 km2)被挂到数百 km2 河段
    —— 即原论文 7 个"失败站"中的 5 个, 属数据错配而非模型失败
  6335606 WENDLINGEN (KLA) 为污水处理厂出流站, 建议剔除

用法:  python qc_stations.py        输出 <BASIN>/station_qc.csv
处理:  人工核对 GRDC 官方元数据面积后, 把要剔除的站写入 <BASIN>/station_exclude.csv
      (单列 station_id), 训练脚本会自动读取并剔除。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C

SPEC_HIGH = 100.0   # l/s/km2
SPEC_LOW = 2.0
HYRIV_RATIO = 8.0


def load_obs_mean():
    rows = {}
    for f in C.GRDC_DIR.glob("*.csv"):
        sid = f.stem
        if not sid[0].isdigit():
            continue
        d = pd.read_csv(f)
        date_col = "data" if "data" in d.columns else d.columns[0]
        flow_col = "MEAN" if "MEAN" in d.columns else d.columns[-1]
        ts = pd.to_datetime(d[date_col], format="mixed", errors="coerce")
        flow = pd.to_numeric(d[flow_col], errors="coerce")
        keep = ts.notna() & (ts >= pd.Timestamp(C.STUDY_START + "-01")) & \
               (ts <= pd.Timestamp(C.STUDY_END + "-31")) & (flow >= 0)
        if keep.sum() >= 24:
            rows[sid] = float(flow[keep].mean())
    return rows


def main():
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str}).set_index("node_id")
    fold = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    in182 = set(fold["station_id"])
    qmean = load_obs_mean()

    recs = []
    for sid, q in qmean.items():
        if sid not in nodes.index:
            continue
        r = nodes.loc[sid]
        A = float(r["catchment_area_km2"])
        hyq = float(r["river_flow_m3s"])
        spec = q * 1000.0 / max(A, 1e-6)                  # l/s/km2
        ratio = q / max(hyq, 1e-6)
        flags = []
        if spec > SPEC_HIGH:
            flags.append("spec_high")
        if spec < SPEC_LOW:
            flags.append("spec_low")
        if ratio > HYRIV_RATIO or ratio < 1.0 / HYRIV_RATIO:
            flags.append("hyriv_mismatch")
        recs.append(dict(
            station_id=sid, name=r.get("name", ""), in_182=sid in in182,
            area_km2=A, obs_mean_m3s=round(q, 2), spec_lskm2=round(spec, 1),
            hyriv_q_m3s=round(hyq, 2), obs_vs_hyriv=round(ratio, 2),
            flags=";".join(flags)))
    df = pd.DataFrame(recs).sort_values(["flags", "spec_lskm2"], ascending=[False, False])
    out = C.BASIN / "station_qc.csv"
    df.to_csv(out, index=False)
    bad = df[df["flags"] != ""]
    print(f"共检查 {len(df)} 站, 可疑 {len(bad)} 站 (在182名单内 {int(bad['in_182'].sum())} 站)")
    print(bad.to_string(index=False))
    print(f"\n已保存 -> {out}")
    print("下一步: 核对 GRDC 官方面积, 将确认错配且无法修复的站写入 station_exclude.csv")


if __name__ == "__main__":
    main()
