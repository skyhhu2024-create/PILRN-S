"""Train and evaluate the revised PILRN-S on the published Rhine random-gauge folds.

Example: python code/run_pilrn_s.py --folds 1 --seeds 42
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import config_v2 as C


def inner_groups(visible, seed, fold):
    """Reproduce the five groups in the order of the released fold inventory."""
    rng = np.random.default_rng(70000 + 100 * fold + seed % 1000)
    return [[visible[i] for i in group] for group in np.array_split(rng.permutation(len(visible)), 5)]


def inputs():
    with np.load(C.FORCING_NPZ, allow_pickle=True) as data:
        nodes = list(map(str, data["node_ids"]))
        features = list(map(str, data["features"]))
    if features != C.DYN_FEATURES:
        raise ValueError(f"Expected the eleven published ERA5-Land features in order, got {features}")
    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate node IDs in the forcing archive")
    folds = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    if folds.station_id.duplicated().any():
        raise ValueError("Each gauge must occur once in the outer-fold inventory")
    excluded_file = C.BASIN / "station_exclude.csv"
    excluded = set(pd.read_csv(excluded_file, dtype=str).iloc[:, 0])
    folds = folds[folds.station_id.isin(set(nodes) - excluded)]
    files = [C.FORCING_NPZ, C.FOLD_CSV, C.NODES_CSV, C.EDGES_CSV, excluded_file]
    observation_files = [C.GRDC_DIR / f"{s}.csv" for s in nodes]
    if not any(f.is_file() for f in observation_files):
        raise FileNotFoundError(f"No gauge discharge CSV files found in {C.GRDC_DIR}")
    files += [f for f in observation_files if f.is_file()]
    code = Path(__file__).resolve().parent
    files += [code / name for name in ("config_v2.py", "graph_v2.py", "train_regional_lstm.py",
                                       "train_pilrn_v2.py", "pilrn_similarity.py", "run_pilrn_s.py")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return nodes, folds, digest.hexdigest()


def materialize(destination, signature, expected, produce):
    """Publish one completed fit atomically. Only reuse an identical fit."""
    def validate():
        marker = destination / "run_signature.json"
        if not marker.is_file() or json.loads(marker.read_text()) != signature:
            raise ValueError(f"Incompatible run in {destination}. Select a different --output-root.")
        for relative in expected:
            path = destination / relative
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Incomplete saved run: {path}")

    if destination.exists():
        validate()
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary)
        produce(root)
        staged = root / destination.name
        for relative in expected:
            path = staged / relative
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing output from fit: {relative}")
        (staged / "run_signature.json").write_text(json.dumps(signature, indent=2))
        try:
            staged.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            validate()  # A concurrent invocation may have finished first.


def fit_regional(directory, seed, fold, training, held, nodes, digest):
    import train_regional_lstm as regional

    def produce(root):
        path = regional.run(fold=fold, seed=seed, visible_ids=training,
                            output_root=root, tag=directory.name)
        metadata_path = path / "norm_params.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["training_visible"] != training or metadata["holdout"] != held:
            raise RuntimeError(f"Stage-1 station split mismatch at {directory}")
        metadata["output_root"] = str(directory.parent)
        metadata_path.write_text(json.dumps(metadata, indent=1))

    expected = ["checkpoint.pth", "norm_params.json"] + [f"predictions_by_station/{s}.csv" for s in nodes]
    materialize(directory, dict(seed=seed, fold=fold, training=training, held=held, inputs=digest),
                expected, produce)


def run_cell(root, nodes, folds, digest, fold, seed):
    import train_pilrn_v2 as router
    from pilrn_similarity import similarity_router

    held = folds.loc[folds.fold == fold, "station_id"].tolist()
    visible = folds.loc[folds.fold != fold, "station_id"].tolist()
    if not held or len(visible) < 5:
        raise ValueError(f"Fold {fold} needs held-out gauges and five nonempty inner groups")
    tag = f"fold_{fold:02d}_seed{seed}"
    outer = root / "regional" / tag
    fit_regional(outer, seed, fold, visible, held, nodes, digest)
    groups = inner_groups(visible, seed, fold)
    membership = {station: k for k, group in enumerate(groups, 1) for station in group}
    inner = {}
    for k, group in enumerate(groups, 1):
        directory = root / "inner" / f"{tag}_inner{k}"
        hidden = set(group)
        training = [s for s in visible if s not in hidden]
        fit_regional(directory, seed, fold, training, held, nodes, digest)
        inner[k] = directory

    signature = dict(seed=seed, fold=fold, inputs=digest, inner_group=membership, outer_test=held)
    priors = root / "priors" / tag

    def assemble(staging):
        folder = staging / tag / "predictions_by_station"
        folder.mkdir(parents=True)
        for station in nodes:
            model = inner[membership[station]] if station in membership else outer
            source = model / "predictions_by_station" / f"{station}.csv"
            # Keep the original numerical representation and discard any observation column.
            with source.open(newline="") as src, (folder / source.name).open("w", newline="") as dst:
                writer = csv.writer(dst)
                writer.writerow(["date", "predicted_flow_m3s"])
                writer.writerows((r["date"], r["predicted_flow_m3s"]) for r in csv.DictReader(src))

    materialize(priors, signature, [f"predictions_by_station/{s}.csv" for s in nodes], assemble)

    def fit_router(staging):
        with similarity_router(k=8):
            router.run(fold=fold, seed=seed, variant="pilrn_sig_no_area", rounds_override=3,
                       stage1_root=priors.parent, stage1_tag=tag, output_root=staging, tag=tag)

    result = root / "router" / tag
    expected = ["checkpoint.pth", "norm_params.json", "edge_weights.csv"]
    expected += [f"predictions_by_station/{s}.csv" for s in held]
    materialize(result, signature, expected, fit_router)
    print(f"Completed {tag}: {result}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026, 7, 2024])
    parser.add_argument("--output-root", type=Path, default=C.SP_V3 / "PILRN_S_revised")
    args = parser.parse_args()
    nodes, folds, digest = inputs()
    for seed in args.seeds:
        for fold in args.folds:
            run_cell(args.output_root.resolve(), nodes, folds, digest, fold, seed)


if __name__ == "__main__":
    main()
