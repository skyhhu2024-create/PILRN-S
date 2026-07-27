# -*- coding: utf-8 -*-
"""
v3 河网重建: 元数据约束重吸附 + 拓扑/河距/局地面积全部重算

设计红线(无泄露保证): 吸附打分只使用 官方坐标 + 官方集水面积 + HydroRIVERS 段属性,
全程不接触任何流量观测。

打分: score = dist_km + LAMBDA * |log10(UPLAND_SKM / A_official)|
      (LAMBDA=3: 面积每差一个数量级等价于 3 km 距离惩罚), 搜索半径 5 km。
官方面积 < 10 km² 的站标记 unmappable(HydroRIVERS 不解析此尺度)。
POI(电站)节点保留原吸附段。

输出: river_network_nodes_v3.csv / river_network_edges_v3.csv / snap_report_v3.csv
用法: python rebuild_network_v3.py
"""
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C

SHP_BASE = C.BASIN / "HydroRIVERS_v10_eu_clip_2040023010"
LAMBDA = 3.0        # km per decade of area mismatch
RADIUS_KM = 5.0
MIN_AREA = 10.0     # km², 小于此为 unmappable


# ---------------- 纯 Python shp/dbf 读取 ----------------
def read_dbf(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        n_rec = struct.unpack("<I", hdr[4:8])[0]
        hdr_size = struct.unpack("<H", hdr[8:10])[0]
        rec_size = struct.unpack("<H", hdr[10:12])[0]
        fields = []
        while True:
            fd = f.read(32)
            if fd[0:1] == b"\r":
                break
            fields.append((fd[:11].split(b"\x00")[0].decode(), fd[11:12].decode(), fd[16]))
        f.seek(hdr_size)
        rows = []
        for _ in range(n_rec):
            rec = f.read(rec_size)
            if rec[0:1] == b"*":
                continue
            off = 1
            row = {}
            for name, ftype, flen in fields:
                raw = rec[off:off + flen].decode("latin1").strip()
                row[name] = float(raw) if (ftype == "N" and raw not in ("", "*")) else raw
                off += flen
            rows.append(row)
    return pd.DataFrame(rows)


def read_shp_polylines(path):
    """返回 list[np.ndarray (n,2) lon/lat] 与 shp 记录顺序一致"""
    out = []
    with open(path, "rb") as f:
        f.seek(100)
        data = f.read()
    i = 0
    while i < len(data):
        length = struct.unpack(">i", data[i + 4:i + 8])[0] * 2
        rec = data[i + 8:i + 8 + length]
        shp_type = struct.unpack("<i", rec[0:4])[0]
        if shp_type == 3:
            n_parts = struct.unpack("<i", rec[36:40])[0]
            n_pts = struct.unpack("<i", rec[40:44])[0]
            pts_off = 44 + 4 * n_parts
            pts = np.frombuffer(rec[pts_off:pts_off + 16 * n_pts], dtype="<f8").reshape(-1, 2)
            out.append(pts.copy())
        else:
            out.append(np.zeros((0, 2)))
        i += 8 + length
    return out


def main():
    print("[1] 读取 HydroRIVERS ...")
    dbf = read_dbf(str(SHP_BASE) + ".dbf")
    lines = read_shp_polylines(str(SHP_BASE) + ".shp")
    assert len(dbf) == len(lines), (len(dbf), len(lines))
    dbf["HYRIV_ID"] = dbf["HYRIV_ID"].astype(int)
    dbf["NEXT_DOWN"] = dbf["NEXT_DOWN"].astype(int)
    seg = dbf.set_index("HYRIV_ID")
    seg_pts = {hid: lines[i] for i, hid in enumerate(dbf["HYRIV_ID"])}
    print(f"    段数 {len(seg)}")

    # 顶点扁平化, 用于近邻搜索
    all_xy, all_id = [], []
    for hid, pts in seg_pts.items():
        if len(pts):
            all_xy.append(pts)
            all_id.append(np.full(len(pts), hid))
    all_xy = np.vstack(all_xy)
    all_id = np.concatenate(all_id)

    print("[2] 读取站点 ...")
    md = pd.read_csv(C.BASIN / "grdc_official_metadata.csv",
                     dtype={"station_id": str}).set_index("station_id")
    old = pd.read_csv(C.BASIN / "river_network_nodes.csv",
                      dtype={"node_id": str}).set_index("node_id")

    print("[3] 元数据约束重吸附 (207 gauges) ...")
    up = seg["UPLAND_SKM"].to_dict()
    rows, report = [], []
    for sid, m in md.iterrows():
        lat, lon, A = m["lat"], m["lon"], m["area_official"]
        clat = np.cos(np.deg2rad(lat))
        d2 = (all_xy[:, 1] - lat) ** 2 + ((all_xy[:, 0] - lon) * clat) ** 2
        近 = d2 < (RADIUS_KM / 111.0) ** 2
        cand = {}
        for hid, dd in zip(all_id[近], d2[近]):
            if hid not in cand or dd < cand[hid]:
                cand[hid] = dd
        best, best_score, best_d = None, np.inf, np.nan
        for hid, dd in cand.items():
            dist_km = np.sqrt(dd) * 111.0
            u = max(up.get(hid, 0.0), 1e-3)
            score = dist_km + LAMBDA * abs(np.log10(u / max(A, 1e-3)))
            if score < best_score:
                best, best_score, best_d = hid, score, dist_km
        unmappable = (A < MIN_AREA) or (best is None)
        old_hid = int(old.loc[sid, "snapped_HYRIV_ID"]) if sid in old.index else -1
        report.append(dict(
            station_id=sid, name=m["station"], area_official=A,
            old_HYRIV=old_hid, new_HYRIV=best if best else -1,
            moved=(best is not None and best != old_hid),
            snap_dist_km=round(best_d, 2) if best else np.nan,
            new_upland=round(up.get(best, np.nan), 1) if best else np.nan,
            unmappable=unmappable))
        rows.append(dict(node_id=sid, node_type="gauge", name=m["station"],
                         lat=lat, lon=lon, hyriv=best if best else -1,
                         area=A, unmappable=unmappable))

    print("[4] POI 节点保留原吸附段 ...")
    for nid, r in old[old["node_type"] != "gauge"].iterrows():
        rows.append(dict(node_id=nid, node_type=r["node_type"], name=r.get("name", ""),
                         lat=r["lat"], lon=r["lon"], hyriv=int(r["snapped_HYRIV_ID"]),
                         area=float(r["catchment_area_km2"]), unmappable=False))
    nodes = pd.DataFrame(rows).set_index("node_id")

    print("[5] 重建拓扑与河距 ...")
    nxt = seg["NEXT_DOWN"].to_dict()
    length = seg["LENGTH_KM"].to_dict()
    seg_nodes = defaultdict(list)                     # hyriv -> [node_id] 按面积升序
    for nid, r in nodes.iterrows():
        if r["hyriv"] > 0 and not r["unmappable"]:
            seg_nodes[r["hyriv"]].append(nid)
    for hid in seg_nodes:
        seg_nodes[hid].sort(key=lambda n: nodes.loc[n, "area"])

    edges = []
    for nid, r in nodes.iterrows():
        if r["hyriv"] <= 0 or r["unmappable"]:
            continue
        hid = r["hyriv"]
        same = seg_nodes[hid]
        k = same.index(nid)
        if k + 1 < len(same):                          # 同段更大面积节点 = 直接下游
            edges.append((nid, same[k + 1], round(length.get(hid, 1.0) / 2, 2)))
            continue
        dist = length.get(hid, 1.0) / 2               # 从段中点走到段尾
        cur = nxt.get(hid, 0)
        hops = 0
        while cur and cur > 0 and hops < 4000:
            if seg_nodes.get(cur):
                edges.append((nid, seg_nodes[cur][0], round(dist + length.get(cur, 1.0) / 2, 2)))
                break
            dist += length.get(cur, 1.0)
            cur = nxt.get(cur, 0)
            hops += 1
    edf = pd.DataFrame(edges, columns=["upstream_id", "downstream_id", "river_dist_km"])
    edf["upstream_type"] = [nodes.loc[u, "node_type"] for u in edf.upstream_id]
    edf["downstream_type"] = [nodes.loc[d, "node_type"] for d in edf.downstream_id]

    print("[6] 局地面积 / 属性合成 ...")
    up_sum = edf.groupby("downstream_id")["upstream_id"].apply(
        lambda ups: sum(nodes.loc[u, "area"] for u in ups))
    out = []
    for nid, r in nodes.iterrows():
        A = r["area"]
        local = max(A - float(up_sum.get(nid, 0.0)), 1.0)
        hid = r["hyriv"] if r["hyriv"] > 0 else None
        o = old.loc[nid] if nid in old.index else None
        out.append(dict(
            node_id=nid, node_type=r["node_type"], name=r["name"],
            lat=r["lat"], lon=r["lon"], snapped_HYRIV_ID=r["hyriv"],
            local_area_km2=round(local, 1), catchment_area_km2=A,
            river_flow_m3s=round(float(seg.loc[hid, "DIS_AV_CMS"]), 3) if hid else np.nan,
            upland_skm=round(float(seg.loc[hid, "UPLAND_SKM"]), 1) if hid else np.nan,
            strahler_order=int(seg.loc[hid, "ORD_STRA"]) if hid else np.nan,
            altitude=(md.loc[nid, "altitude_official"] if nid in md.index
                      else (o["altitude"] if o is not None else np.nan)),
            slope=o["slope"] if o is not None else np.nan,
            snd=o["snd"] if o is not None else np.nan,
            slt=o["slt"] if o is not None else np.nan,
            scl=o["scl"] if o is not None else np.nan,
            unmappable=r["unmappable"]))
    ndf = pd.DataFrame(out)

    ndf.to_csv(C.BASIN / "river_network_nodes_v3.csv", index=False)
    edf[["upstream_id", "downstream_id", "upstream_type", "downstream_type",
         "river_dist_km"]].to_csv(C.BASIN / "river_network_edges_v3.csv", index=False)
    rep = pd.DataFrame(report)
    rep.to_csv(C.BASIN / "snap_report_v3.csv", index=False)

    # ---------------- 验证摘要 ----------------
    g = rep[~rep.unmappable]
    print("\n===== 重吸附摘要 =====")
    print(f"gauges 可映射 {len(g)}/{len(rep)}, 换段 {int(g.moved.sum())} 站")
    print(f"吸附距离 median {g.snap_dist_km.median():.2f} km, max {g.snap_dist_km.max():.2f} km")
    ratio = g.new_upland / g.area_official
    print(f"UPLAND/官方面积 比: P5={ratio.quantile(.05):.2f} P50={ratio.quantile(.5):.2f} "
          f"P95={ratio.quantile(.95):.2f}  >30%偏差: {int(((ratio>1.3)|(ratio<1/1.3)).sum())} 站")
    for sid, nm in [("6435060", "LOBITH"), ("6335601", "LAUFFEN"),
                    ("6935600", "CHUR"), ("6335695", "PLOCHINGEN")]:
        rr = rep[rep.station_id == sid]
        if len(rr):
            rr = rr.iloc[0]
            print(f"  {nm:11s} A_off={rr.area_official:9.1f}  新段UPLAND={rr.new_upland:9.1f}"
                  f"  moved={rr.moved}  dist={rr.snap_dist_km} km")
    print(f"边数 (有向前身) {len(edf)}")


if __name__ == "__main__":
    main()
