"""Numerical regression tests for direct observed-donor similarity edges."""
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import pilrn_similarity as sim
from test_physical_router_v4 import force_neighbour_gate, normalized, runoff


class SimilarityRouterTests(TestCase):
    def test_hidden_intermediate_can_relay_only_on_a_river_edge(self):
        model = sim.PILRNSignalSimilarity(1, 1.2, 0.8, edge_mode="uniform")
        force_neighbour_gate(model)
        # The measured 80 reaches node 1 by similarity. Node 3's 50 is
        # routed back to node 0 to check that its observed 80 is reimposed.
        source = torch.tensor([0, 1, 3])
        target = torch.tensor([1, 2, 0])
        for is_similarity, expected_last in [(True, 30.0), (False, 80.0)]:
            with self.subTest(is_similarity=is_similarity):
                descriptors = torch.zeros(3, 9)
                descriptors[:, 8] = torch.tensor([1, is_similarity, 0])
                prediction, shares, _ = model(
                    normalized(torch.tensor([80., 0., 0., 50.])), torch.tensor([1., 0., 0., 1.]),
                    normalized(torch.tensor([10., 20., 30., 40.])), torch.ones(4),
                    torch.zeros(4, 1), source, target, descriptors, torch.zeros(3, 1), 4)
                torch.testing.assert_close(runoff(prediction), torch.tensor([80., 80., expected_last, 50.]))
                self.assertAlmostEqual(shares[1].item(), 0.0 if is_similarity else 1.0)

    def test_final_model_size(self):
        model = sim.PILRNSignalSimilarity(s_dim=10, y_mu=0., y_sd=1.)
        self.assertEqual(model.rounds, 3)
        self.assertFalse(model.area.enabled)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1106)

    def test_small_candidate_pool_skips_self_and_river_duplicates(self):
        # All nine ranking attributes increase together, so B is C's nearest donor.
        ids = ["A", "B", "C"]
        cols = ["catchment_area_km2", "local_area_km2", "altitude", "slope", "snd", "slt",
                "scl", "strahler_order", "lat", "lon"]
        attributes = pd.DataFrame({name: [1., 2., 3.] for name in cols})
        attributes["node_id"] = ids
        climate = np.array([1., 2., 3.])
        stats = dict(f16_mu=[0.] * 6, f16_sd=[1.] * 6, dtp_max=1., dt2_max=1.)
        with patch.object(sim, "_gauge_indices", return_value=np.array([0, 1])), \
                patch.object(sim.pd, "read_csv", return_value=attributes):
            def pairs(k):
                src, dst, *_ = sim.similarity_edges(ids, climate, climate, stats, {(1, 2)}, k)
                return set(zip(src, dst))
            self.assertEqual(pairs(8), {(0, 1), (1, 0), (0, 2)})
            self.assertEqual(pairs(1), {(0, 1), (1, 0)})


if __name__ == "__main__":
    main()
