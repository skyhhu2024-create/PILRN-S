# -*- coding: utf-8 -*-
"""
ERA5-Land 月均数据预处理 (Rebuild_v2 第一步)

用法(在你的工作站上运行, 需要 xarray + netCDF4):
    python prepare_forcing.py              # 仅检查文件结构并打印摘要
    python prepare_forcing.py --extract    # 检查 + 提取逐站序列, 输出 Forcing_v2/
    python prepare_forcing.py --extract --csv   # 同时输出逐站 CSV 便于人工核对
可选: --nc <路径> 覆盖默认 NC 文件路径; --root <项目根目录>

输出:
    Forcing_v2/forcing_v2.npz    data[T,N,F], dates, node_ids, features
    Forcing_v2/forcing_summary.txt
    Forcing_v2/station_csv/<node_id>.csv   (--csv 时)
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as C


def open_dataset(nc_path):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    # 时间坐标名
    tname = next((c for c in ("valid_time", "time", "date") if c in ds.coords or c in ds.dims), None)
    if tname is None:
        raise RuntimeError(f"找不到时间坐标, coords={list(ds.coords)}")
    lon = "longitude" if "longitude" in ds.coords else "lon"
    lat = "latitude" if "latitude" in ds.coords else "lat"
    # 新版 CDS 可能带 expver / number 维度 -> 压掉
    for extra in ("expver", "number"):
        if extra in ds.dims:
            ds = ds.mean(dim=extra, skipna=True) if ds.sizes[extra] > 1 else ds.squeeze(extra, drop=True)
    return ds, tname, lat, lon


def resolve_vars(ds):
    """canonical 名 -> 实际变量名; 返回 (found, missing)"""
    found, missing = {}, []
    for canon, cands in C.VAR_CANDIDATES.items():
        name = next((v for v in cands if v in ds.data_vars), None)
        if name:
            found[canon] = name
        else:
            missing.append(canon)
    return found, missing


def inspect(ds, tname, lat, lon, found, missing, report_lines):
    def w(s):
        print(s)
        report_lines.append(s)

    t = pd.DatetimeIndex(ds[tname].values)
    w(f"[时间] {t[0]:%Y-%m} ~ {t[-1]:%Y-%m}  共 {len(t)} 步, 推断频率: {pd.infer_freq(t)}")
    w(f"[空间] lon {float(ds[lon].min()):.2f}~{float(ds[lon].max()):.2f}, "
      f"lat {float(ds[lat].min()):.2f}~{float(ds[lat].max()):.2f}, "
      f"分辨率 {abs(float(ds[lon][1] - ds[lon][0])):.3f} 度, "
      f"网格 {ds.sizes[lat]} x {ds.sizes[lon]}")
    w(f"[变量] 找到 {len(found)}/{len(C.VAR_CANDIDATES)}: {found}")
    if missing:
        w(f"[警告] 缺少变量: {missing} —— 请核对下载配置!")
    for canon, name in found.items():
        da = ds[name]
        v0 = da.isel({tname: 0}).values
        nan_pct = 100 * np.isnan(v0).mean()
        w(f"  {canon:6s} <- {name:35s} units={da.attrs.get('units', '?'):12s} "
          f"首月范围 [{np.nanmin(v0):.4g}, {np.nanmax(v0):.4g}]  NaN {nan_pct:.1f}%")
    return t


def convert_units(canon, arr, days_in_month):
    """arr: [T, ...]; days_in_month: [T] 广播到空间维"""
    kind = C.VAR_KIND[canon]
    d = days_in_month.reshape([-1] + [1] * (arr.ndim - 1))
    if kind == "temp":
        return arr - 273.15                     # K -> degC
    if kind == "flux":
        return arr * 1000.0 * d                 # m/day -> mm/month
    if kind == "flux_neg":
        return -arr * 1000.0 * d                # 蒸发取正值
    if kind == "state_m":
        return arr * 1000.0                     # m -> mm
    return arr                                   # state 原样


def extract(ds, tname, lat, lon, found, out_dir, write_csv, report_lines):
    def w(s):
        print(s)
        report_lines.append(s)

    t = pd.DatetimeIndex(ds[tname].values).to_period("M").to_timestamp()
    days = np.array([pd.Period(x, "M").days_in_month for x in t], dtype=np.float64)
    lats, lons = ds[lat].values.astype(np.float64), ds[lon].values.astype(np.float64)

    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str})
    N, T, F = len(nodes), len(t), len(found)
    w(f"[提取] {N} 节点 x {T} 月 x {F} 特征")

    # 有效格点掩膜(ERA5-Land 海上为 NaN): 用第一个变量首月
    first = ds[list(found.values())[0]].isel({tname: 0}).values
    valid = ~np.isnan(first)                                    # [lat, lon]
    LG, LT = np.meshgrid(lons, lats)                            # [lat, lon]
    vy, vx = np.where(valid)
    vlat, vlon = LT[vy, vx], LG[vy, vx]

    # 每个节点 -> 最近有效格点(经度按 cos(lat) 加权)
    idx_y = np.empty(N, dtype=int)
    idx_x = np.empty(N, dtype=int)
    far = []
    for i, r in nodes.iterrows():
        clat = np.cos(np.deg2rad(r["lat"]))
        d2 = (vlat - r["lat"]) ** 2 + ((vlon - r["lon"]) * clat) ** 2
        j = int(np.argmin(d2))
        idx_y[i], idx_x[i] = vy[j], vx[j]
        if np.sqrt(d2[j]) > 0.5:
            far.append((r["node_id"], round(float(np.sqrt(d2[j])), 2)))
    if far:
        w(f"[警告] {len(far)} 个节点最近有效格点超过 0.5 度: {far[:5]} ...")

    data = np.full((T, N, F), np.nan, dtype=np.float32)
    for k, (canon, name) in enumerate(found.items()):
        arr = ds[name].values.astype(np.float64)                # [T, lat, lon]
        arr = convert_units(canon, arr, days)
        data[:, :, k] = arr[:, idx_y, idx_x]
        col = data[:, :, k]
        w(f"  {canon:6s} 提取后范围 [{np.nanmin(col):.4g}, {np.nanmax(col):.4g}] "
          f"NaN {100 * np.isnan(col).mean():.2f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "forcing_v2.npz",
        data=data,
        dates=np.array([f"{x:%Y-%m}" for x in t]),
        node_ids=nodes["node_id"].astype(str).values,
        features=np.array(list(found.keys())),
    )
    w(f"[OK] 保存 {out_dir / 'forcing_v2.npz'}  shape={data.shape}")

    if write_csv:
        csv_dir = out_dir / "station_csv"
        csv_dir.mkdir(exist_ok=True)
        cols = list(found.keys())
        for i, nid in enumerate(nodes["node_id"].astype(str)):
            pd.DataFrame(data[:, i, :], index=t, columns=cols).rename_axis("date") \
                .to_csv(csv_dir / f"{nid}.csv", float_format="%.4f")
        w(f"[OK] 逐站 CSV -> {csv_dir}")


def main():
    global C

    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    if a.root:
        os.environ["JOH_ROOT"] = a.root
    C = importlib.reload(C)
    nc = Path(a.nc) if a.nc else C.NC_PATH
    if not nc.exists():
        nearby = []
        era5_dir = C.BASIN / "ERA5"
        if era5_dir.exists():
            nearby = [p.name for p in sorted(era5_dir.glob("*.nc"))]
        if nearby:
            sys.exit(f"[Error] NC 文件不存在: {nc}\n可用文件: {nearby}")
        sys.exit(f"[Error] NC 文件不存在: {nc}")

    report = [f"NC 文件: {nc} ({nc.stat().st_size / 1e6:.0f} MB)"]
    ds, tname, lat, lon = open_dataset(nc)
    found, missing = resolve_vars(ds)
    inspect(ds, tname, lat, lon, found, missing, report)

    if a.extract:
        if missing:
            print(f"[警告] 缺少 {missing}, 仅提取已有变量继续")
        extract(ds, tname, lat, lon, found, C.FORCING_DIR, a.csv, report)

    C.FORCING_DIR.mkdir(parents=True, exist_ok=True)
    (C.FORCING_DIR / "forcing_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\n摘要已写入 {C.FORCING_DIR / 'forcing_summary.txt'}")


if __name__ == "__main__":
    main()
