# -*- coding: utf-8 -*-
"""Rebuild_v2 global configuration.

Two layouts are supported:

1. the original research workspace, where data live under
   ``<JOH_ROOT>/processed_data/EU/Rine_Basin``;
2. a standalone GitHub clone, where redistributable configuration lives in
   ``<repository>/data`` and generated files are written below
   ``<repository>/outputs``.

``JOH_ROOT`` keeps its original meaning and overrides automatic detection.
Raw licensed inputs can be supplied independently with ``GRDC_DIR`` and
``NC_PATH``.  ``JOH_OUTPUT_ROOT`` optionally overrides the generated-output
directory.
"""
import os
from pathlib import Path


HERE = Path(__file__).resolve()
REPOSITORY_ROOT = HERE.parents[1]
RELEASE_DATA = REPOSITORY_ROOT / "data"


def project_root() -> tuple[Path, bool]:
    env = os.environ.get("JOH_ROOT")
    if env:
        return Path(env).expanduser().resolve(), False

    # Prefer a self-contained release clone even when it happens to be nested
    # inside a larger local workspace that also contains ``processed_data``.
    if (
        (RELEASE_DATA / "river_network_nodes_v3.csv").exists()
        and (RELEASE_DATA / "kfold_station_assignment.csv").exists()
    ):
        return REPOSITORY_ROOT, True

    for p in HERE.parents:
        if (p / "processed_data").exists():
            return p, False

    # Fallback to the standalone-release convention so that missing raw inputs
    # produce clear file errors rather than pointing outside the repository.
    return REPOSITORY_ROOT, True


ROOT, RELEASE_LAYOUT = project_root()
BASIN = (
    RELEASE_DATA
    if RELEASE_LAYOUT
    else ROOT / "processed_data" / "EU" / "Rine_Basin"
)


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def resolve_nc_path() -> Path:
    """优先使用已存在的 ERA5 nc 文件；兼容文件名为 Rine_1990-2025_all.nc 等情况。"""
    configured = os.environ.get("NC_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    era5_dir = BASIN / "ERA5"
    if era5_dir.exists():
        for name in ("Rine_1990-2025_all.nc", "Rine_1995-2025_all.nc"):
            p = era5_dir / name
            if p.exists():
                return p
        matches = sorted(era5_dir.glob("*.nc"))
        if matches:
            return matches[0]

    return era5_dir / "Rine_1990-2025_all.nc"


def resolve_nodes_csv() -> Path:
    """优先级: v3(重吸附+重建拓扑) > v2(仅属性校正) > v1(原始)。"""
    for name in ("river_network_nodes_v3.csv", "river_network_nodes_v2.csv",
                 "river_network_nodes.csv"):
        p = BASIN / name
        if p.exists():
            return p
    return BASIN / "river_network_nodes.csv"


def resolve_edges_csv() -> Path:
    v3 = BASIN / "river_network_edges_v3.csv"
    return v3 if v3.exists() else BASIN / "river_network_edges.csv"


# ---------------- 数据路径 ----------------
NC_PATH = resolve_nc_path()                             # ERA5-Land 月均下载文件
NODES_CSV = resolve_nodes_csv()                         # 节点静态属性(优先 v3)
EDGES_CSV = resolve_edges_csv()                         # 河网边(优先 v3)
GRDC_DIR = env_path("GRDC_DIR", BASIN / "GRDC_Rine")   # 逐站月观测 (data, MEAN)
FOLD_CSV = (
    BASIN / "kfold_station_assignment.csv"
    if RELEASE_LAYOUT
    else BASIN / "Single_Prediction" / "kfold_station_assignment.csv"
)

OUTPUT_ROOT = env_path(
    "JOH_OUTPUT_ROOT",
    REPOSITORY_ROOT / "outputs" if RELEASE_LAYOUT else BASIN,
)
FORCING_DIR = OUTPUT_ROOT / "Forcing_v3"                # 预处理输出(v3 节点表坐标)
FORCING_NPZ = FORCING_DIR / "forcing_v2.npz"
SP_V3 = OUTPUT_ROOT / "Single_Prediction_v3"            # v3 干净重跑的输出根目录
REGIONAL_OUT = SP_V3 / "Regional_LSTM"

# ---------------- 研究时段(与原论文一致,保证可比) ----------------
STUDY_START = "1995-04"      # 观测评估起点
STUDY_END = "2019-12"        # 观测评估终点(297 个月)
TRAIN_END = "2012-06"        # 70%: 1995-04 ~ 2012-06 (207 个月)
VAL_END = "2016-03"          # 15%: 2012-07 ~ 2016-03 (45 个月)
# 测试期: 2016-04 ~ 2019-12 (45 个月) —— 与原 KFold 评估的 valid_points=45 对齐

# ---------------- 动态特征(canonical 名 -> NC 中可能的变量名) ----------------
VAR_CANDIDATES = {
    "t2m":   ["t2m", "2m_temperature"],
    "tp":    ["tp", "total_precipitation"],
    "sf":    ["sf", "snowfall"],
    "sd":    ["sd", "snow_depth_water_equivalent", "sde"],
    "smlt":  ["smlt", "snowmelt"],
    "e":     ["e", "total_evaporation", "evaporation"],
    "pev":   ["pev", "potential_evaporation"],
    "swvl1": ["swvl1", "volumetric_soil_water_layer_1"],
    "swvl2": ["swvl2", "volumetric_soil_water_layer_2"],
    "swvl3": ["swvl3", "volumetric_soil_water_layer_3"],
    "swvl4": ["swvl4", "volumetric_soil_water_layer_4"],
}
# 单位换算类型: flux = 月均日累积(m/day)->mm/month(x1000x当月天数, e/pev 取负号使蒸发为正)
#              state_m = 米->毫米; state = 原样; temp = K->degC
VAR_KIND = {
    "t2m": "temp", "tp": "flux", "sf": "flux", "smlt": "flux",
    "e": "flux_neg", "pev": "flux_neg", "sd": "state_m",
    "swvl1": "state", "swvl2": "state", "swvl3": "state", "swvl4": "state",
}
DYN_FEATURES = list(VAR_CANDIDATES.keys())   # 顺序即特征顺序

# ---------------- 静态特征(取自 river_network_nodes.csv,另加训练期气候均值) ----------------
STATIC_COLS = ["catchment_area_km2", "local_area_km2", "altitude", "slope",
               "snd", "slt", "scl", "strahler_order", "lat", "lon"]
LOG_STATIC = {"catchment_area_km2", "local_area_km2"}    # 取 log1p

# ---------------- Regional LSTM 超参 ----------------
SEQ_LENGTH = 12          # 回看 12 个月(覆盖完整水文年)
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 8
SEED = 42
