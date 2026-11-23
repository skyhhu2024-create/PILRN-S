"""Direct observed-gauge similarity edges for the revised PILRN-S router."""
from contextlib import contextmanager
from functools import partial

import numpy as np
import pandas as pd
import torch

import config_v2 as C
import graph_v2
import train_pilrn_v2


K_SIM = 8
_river_graph = graph_v2.build_graph


def _gauge_indices(node_ids):
    folds = pd.read_csv(C.FOLD_CSV, dtype={"station_id": str})
    exclusions = C.BASIN / "station_exclude.csv"
    excluded = set(pd.read_csv(exclusions, dtype=str).iloc[:, 0]) if exclusions.exists() else set()
    index = {station: i for i, station in enumerate(node_ids)}
    return np.asarray([index[s] for s in folds.station_id.drop_duplicates()
                       if s in index and s not in excluded], dtype=np.int64)


def similarity_edges(node_ids, clim_tp, clim_t2m, stats, existing, k=K_SIM):
    """Rank gauges using catchment attributes and training-period climate only.

    Select up to k other gauges first, then skip existing river edges without
    refilling them. Training and monthly observation masks act on this fixed graph.
    """
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    nodes = pd.read_csv(C.NODES_CSV, dtype={"node_id": str}).set_index("node_id").reindex(node_ids)

    def column(name):
        values = nodes[name].astype(float)
        return values.fillna(values.median()).to_numpy()

    area = np.maximum(column("catchment_area_km2"), 1e-3)
    local_area = np.maximum(column("local_area_km2"), 1e-3)
    elevation, slope = column("altitude"), column("slope")
    attributes = np.column_stack([np.log(area), elevation, slope, column("snd"), column("slt"),
                                  column("scl"), column("strahler_order"), clim_tp, clim_t2m])
    gauges = _gauge_indices(node_ids)
    sources, targets = [], []
    if gauges.size and k:
        standardized = (attributes - attributes[gauges].mean(0)) / (attributes[gauges].std(0) + 1e-8)
        for i in range(len(node_ids)):
            distances = np.sqrt(((standardized[gauges] - standardized[i]) ** 2).sum(1))
            distances[gauges == i] = np.inf
            order = np.argsort(distances)
            candidates = gauges[order[np.isfinite(distances[order])][:k]]
            for j in candidates:
                if (j, i) not in existing:
                    sources.append(j)
                    targets.append(i)

    src = np.asarray(sources, dtype=np.int64)
    dst = np.asarray(targets, dtype=np.int64)
    lat, lon = np.radians(column("lat")), np.radians(column("lon"))
    dlat, dlon = lat[dst] - lat[src], lon[dst] - lon[src]
    haversine = (np.sin(dlat / 2) ** 2
                 + np.cos(lat[src]) * np.cos(lat[dst]) * np.sin(dlon / 2) ** 2)
    km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    features = np.column_stack([
        np.log(area[src] / area[dst]), np.log(local_area[src] / local_area[dst]),
        elevation[src] - elevation[dst], slope[src] - slope[dst], 1.0 / (km + 1.0),
        np.log(km + 1.0), 1.0 - np.abs(clim_tp[src] - clim_tp[dst]) / max(stats["dtp_max"], 1e-8),
        1.0 - np.abs(clim_t2m[src] - clim_t2m[dst]) / max(stats["dt2_max"], 1e-8)
    ])
    features[:, :6] = (features[:, :6] - stats["f16_mu"]) / stats["f16_sd"]
    return src, dst, features, area[src] / area[dst], km


def build_graph_with_similarity(node_ids, clim_tp, clim_t2m, k=K_SIM):
    """Append similarity edges to the physical graph and flag them in column 9."""
    edges, features, ratios, distances, reference_weights, stats = _river_graph(
        node_ids, clim_tp, clim_t2m)
    existing = set(zip(edges[0], edges[1]))
    src, dst, sim_features, sim_ratio, km = similarity_edges(
        node_ids, np.asarray(clim_tp, float), np.asarray(clim_t2m, float), stats, existing, k)
    river_count = edges.shape[1]
    edges = np.concatenate([edges, np.stack([src, dst])], axis=1)
    features = np.vstack([features, sim_features.astype(features.dtype)])
    edge_type = np.r_[np.zeros(river_count), np.ones(src.size)]
    features = np.column_stack([features, edge_type.astype(features.dtype)])
    # Preserve the archived similarity-edge fields. The final model has beta=1.
    ratios = np.concatenate([ratios, sim_ratio.reshape(-1, *ratios.shape[1:]).astype(ratios.dtype)])
    distances = np.r_[distances, km.astype(distances.dtype)]
    reference_weights = np.r_[reference_weights, np.zeros(src.size, dtype=reference_weights.dtype)]
    stats = dict(stats, similarity_k=int(k), n_network_edges=river_count, n_similarity_edges=src.size)
    return edges, features, ratios, distances, reference_weights, stats


class PILRNSignalSimilarity(train_pilrn_v2.PILRNSignal):
    """River neighbors can propagate updates; similarity donors require a direct observation."""

    def __init__(self, s_dim, y_mu, y_sd, rounds=3, edge_mode="full", area_correction=False):
        super().__init__(s_dim, y_mu, y_sd, rounds, edge_mode, area_correction)
        if self.edge_mode == "full":
            # Keep the initialization order of the published experiment and checkpoints.
            self.ewn = train_pilrn_v2.EdgeWeightNet(in_dim=9)

    def forward(self, obs, vflag, reg, regflag, sta, src, dst, ef, lar, n):
        weights = self.ewn(ef) if self.edge_mode == "full" else obs.new_ones(src.shape[0])
        scale = self.area(lar)
        observed = (vflag > 0).float()
        available = observed
        obs_runoff = self.to_runoff(obs)
        prior = torch.where(regflag > 0, self.to_runoff(reg), torch.zeros_like(reg))
        signal = torch.where(vflag > 0, obs_runoff, prior)
        is_similarity = ef[:, 8] > 0.5
        shares = torch.zeros_like(weights)
        for _ in range(self.rounds):
            support = torch.where(is_similarity, observed[src], available[src])
            contribution = weights * support
            denominator = torch.zeros_like(signal).index_add_(0, dst, contribution)
            numerator = torch.zeros_like(signal).index_add_(0, dst, contribution * scale * signal[src])
            neighbor = numerator / (denominator + 1e-8)
            has_neighbor = (denominator > 1e-8).float()
            shares = contribution / (denominator[dst] + 1e-12)
            gate_inputs = torch.cat([sta, available.unsqueeze(1),
                                     self.to_normalized_log(signal).unsqueeze(1),
                                     self.to_normalized_log(neighbor).unsqueeze(1)], dim=1)
            gate = torch.sigmoid(self.gate(gate_inputs)).squeeze(-1)
            merged = gate * signal + (1 - gate) * neighbor
            updated = torch.where(has_neighbor > 0, merged, signal)
            signal = torch.where(vflag > 0, obs_runoff, updated)
            available = torch.maximum(available, has_neighbor)
        return self.to_normalized_log(signal), shares, scale


@contextmanager
def similarity_router(k=K_SIM):
    """Use the augmented graph with the released single-process training functions."""
    import train_coupled_v2

    previous = graph_v2.build_graph, train_pilrn_v2.PILRNSignal, train_coupled_v2.PILRNSignal
    graph_v2.build_graph = partial(build_graph_with_similarity, k=k)
    train_pilrn_v2.PILRNSignal = train_coupled_v2.PILRNSignal = PILRNSignalSimilarity
    try:
        yield
    finally:
        graph_v2.build_graph, train_pilrn_v2.PILRNSignal, train_coupled_v2.PILRNSignal = previous
