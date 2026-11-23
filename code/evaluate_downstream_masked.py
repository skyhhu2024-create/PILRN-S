"""Replay the final 45 Rhine months at the published random-fold evaluation gauges.

For each held-out target, remove every downstream gauge's same-month observation
from the complete input, including observations that could enter through a
similarity edge. Other eligible observations remain available. Training, the
frozen priors and model weights are identical in both observation scenarios.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

import config_v2 as C
from pilrn_similarity import PILRNSignalSimilarity, build_graph_with_similarity
from train_regional_lstm import flow_to_runoff, load_forcing, load_obs, load_statics, metrics


def downstream_sets(nodes):
    index = {node: i for i, node in enumerate(nodes)}
    links = pd.read_csv(C.EDGES_CSV, dtype={"upstream_id": str, "downstream_id": str})
    direct = defaultdict(list)
    for row in links.itertuples():
        direct[index[row.upstream_id]].append(index[row.downstream_id])
    result = []
    for target in range(len(nodes)):
        pending = list(direct[target])
        below = set()
        while pending:
            station = pending.pop()
            if station not in below:
                below.add(station)
                pending.extend(direct[station])
        if target in below:
            raise ValueError(f"Cycle in the directed river network at {nodes[target]}")
        result.append(below)
    return result


def read_series(folder, stations, months):
    columns = []
    for station in stations:
        path = folder / "predictions_by_station" / f"{station}.csv"
        data = pd.read_csv(path, usecols=["date", "predicted_flow_m3s"])
        data.index = pd.PeriodIndex(data.date, freq="M")
        if data.index.has_duplicates or not months.isin(data.index).all():
            raise ValueError(f"Incomplete or duplicate date index in {path}")
        columns.append(data.loc[months, "predicted_flow_m3s"].to_numpy())
    return np.column_stack(columns)


class RhineEvaluation:
    def __init__(self):
        dynamic, calendar, self.nodes, features = load_forcing()
        self.index = {node: i for i, node in enumerate(self.nodes)}
        training = (calendar >= C.STUDY_START) & (calendar <= C.TRAIN_END)
        climate = [np.nan_to_num(np.nanmean(dynamic[training, :, features.index(name)], axis=0))
                   for name in ("tp", "t2m")]
        indices, descriptors, ratios, *_ = build_graph_with_similarity(self.nodes, *climate, k=8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph = [torch.as_tensor(value, device=self.device) for value in
                      (indices[0], indices[1], descriptors, np.log(ratios))]
        self.below = downstream_sets(self.nodes)
        self.months = pd.period_range("2016-04", "2019-12", freq="M")
        self.static, self.area = load_statics(self.nodes)
        self.obs = load_obs(self.nodes, self.months)
        self.depth = flow_to_runoff(self.obs, self.area, self.months)
        self.scale = flow_to_runoff(np.ones_like(self.obs), self.area, self.months)
        self.scored = pd.read_csv(C.REPOSITORY_ROOT / "results" / "station_seed_metrics.csv",
                                  usecols=["fold", "station_id"], dtype={"station_id": str}).drop_duplicates()

    @torch.inference_mode()
    def score(self, router, priors, seed, fold):
        cfg = json.loads((router / "norm_params.json").read_text(encoding="utf-8"))
        if (cfg["seed"], cfg["fold"], cfg["variant"], cfg["rounds"], cfg["strict_validation"]) != (
                seed, fold, "pilrn_sig_no_area", 3, True):
            raise ValueError(f"Incompatible checkpoint in {router}")
        targets = sorted(self.scored.loc[self.scored.fold == fold, "station_id"])
        held = set(cfg["test_stations"])
        visible = set(cfg["visible_stations"])
        validation = set(cfg["validation_stations"])
        if not targets or not set(targets) <= held or visible & held or not validation <= visible:
            raise ValueError(f"Invalid target or donor roster in {router}")
        allowed = {self.index[s] for s in visible - validation}
        reference = read_series(router, targets, self.months)
        prior = read_series(priors, self.nodes, self.months) * self.scale
        mu, sd = cfg["y_mu"], cfg["y_sd"]
        normalized = (np.log1p(np.maximum(prior, 0)) - mu) / sd
        y = (np.log1p(np.maximum(self.depth, 0)) - mu) / sd
        visible_rows = self.static[[self.index[s] for s in cfg["visible_stations"]]]
        static = (self.static - visible_rows.mean(axis=0)) / (visible_rows.std(axis=0) + 1e-8)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        static = tensor(static)
        zprior = tensor(np.nan_to_num(normalized, nan=0))
        prior_flag = tensor(np.isfinite(prior))
        model = PILRNSignalSimilarity(static.shape[1], mu, sd, rounds=3, edge_mode="full",
                                      area_correction=False).to(self.device)
        model.load_state_dict(torch.load(router / "checkpoint.pth", map_location=self.device, weights_only=True))
        model.eval()

        target_cols = [self.index[s] for s in targets]
        groups = defaultdict(list)
        for col in target_cols:
            groups[tuple(sorted(allowed & self.below[col]))].append(col)
        scenarios = [("available", (), target_cols)]
        scenarios.extend(("masked", blocked, cols) for blocked, cols in groups.items())
        rows = []
        for scenario, blocked, cols in scenarios:
            donor_flag = np.zeros(len(self.nodes), dtype=bool)
            donor_flag[list(allowed.difference(blocked))] = True
            flags = np.isfinite(y) & donor_flag
            observations = tensor(np.where(flags, y, 0))
            flags = tensor(flags)
            values = []
            for t in range(len(self.months)):
                z, _, _ = model(observations[t], flags[t], zprior[t], prior_flag[t], static,
                                *self.graph, len(self.nodes))
                values.append(z[cols].cpu().numpy())
            predictions = np.expm1(np.stack(values) * sd + mu) / self.scale[:, cols]
            if scenario == "available":
                np.testing.assert_allclose(predictions, reference, rtol=4e-5, atol=3e-4,
                                           err_msg=f"Unmasked replay differs from {router}")
            for j, col in enumerate(cols):
                scores = metrics(self.obs[:, col], predictions[:, j])
                if scores is None:
                    raise ValueError(f"Insufficient observations or predictions for {self.nodes[col]}")
                rows.append(dict(seed=seed, fold=fold, station_id=self.nodes[col], downstream=scenario,
                                 observed_months=scores["n"], R=scores["r"], NSE=scores["NSE"],
                                 KGE2012=scores["KGE"],
                                 excluded_downstream_donor_ids=",".join(self.nodes[i] for i in blocked)))
        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026, 7, 2024])
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--output-root", type=Path, default=C.SP_V3 / "PILRN_S_revised",
                        help="Training output root created by run_pilrn_s.py")
    args = parser.parse_args()
    torch.set_num_threads(2)
    replay = RhineEvaluation()
    directory = args.output_root / "downstream_replay"
    directory.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for fold in args.folds:
            tag = f"fold_{fold:02d}_seed{seed}"
            scores = replay.score(args.output_root / "router" / tag, args.output_root / "priors" / tag, seed, fold)
            output = directory / f"{tag}.csv"
            with NamedTemporaryFile(dir=directory, delete=False) as tmp:
                temp = Path(tmp.name)
            try:
                scores.to_csv(temp, index=False)
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)
            print(f"{output}: {scores.station_id.nunique()} gauges, both observation scenarios")


if __name__ == "__main__":
    main()
