# -*- coding: utf-8 -*-
"""Numerical regression tests for the physical-space signal router."""
from __future__ import annotations

import math
import unittest

import torch

from train_pilrn_v2 import PILRNSignal


def normalized(runoff: torch.Tensor, mu=1.2, sd=0.8) -> torch.Tensor:
    return (torch.log1p(runoff) - mu) / sd


def runoff(z: torch.Tensor, mu=1.2, sd=0.8) -> torch.Tensor:
    return torch.expm1(z * sd + mu).clamp_min(0.0)


def force_neighbour_gate(model: PILRNSignal) -> None:
    with torch.no_grad():
        for parameter in model.gate.parameters():
            parameter.zero_()
        model.gate[-1].bias.fill_(-30.0)


class PhysicalRouterTests(unittest.TestCase):
    def test_generalized_dar_is_exact_in_runoff_depth_space(self):
        model = PILRNSignal(
            s_dim=1,
            y_mu=1.2,
            y_sd=0.8,
            rounds=1,
            edge_mode="uniform",
            area_correction=True,
        )
        force_neighbour_gate(model)
        target_beta = 0.8
        probability = (target_beta - 0.7) / 0.6
        with torch.no_grad():
            model.area.theta.fill_(math.log(probability / (1.0 - probability)))

        source_runoff = torch.tensor([100.0, 5.0])
        obs = normalized(source_runoff)
        visible = torch.tensor([1.0, 0.0])
        regional = normalized(torch.tensor([10.0, 20.0]))
        regional_flag = torch.ones(2)
        static = torch.zeros(2, 1)
        src = torch.tensor([0])
        dst = torch.tensor([1])
        edge_features = torch.zeros(1, 8)
        log_ratio = torch.log(torch.tensor([[4.0]]))

        prediction, _, _ = model(
            obs,
            visible,
            regional,
            regional_flag,
            static,
            src,
            dst,
            edge_features,
            log_ratio,
            2,
        )
        expected = 100.0 * 4.0 ** (target_beta - 1.0)
        self.assertAlmostEqual(float(runoff(prediction)[1]), expected, places=4)
        self.assertAlmostEqual(float(runoff(prediction)[0]), 100.0, places=4)

    def test_no_neighbour_retains_regional_prior(self):
        model = PILRNSignal(
            s_dim=1,
            y_mu=1.2,
            y_sd=0.8,
            rounds=2,
            edge_mode="uniform",
            area_correction=False,
        )
        force_neighbour_gate(model)
        regional_runoff = torch.tensor([12.0, 34.0])
        prediction, _, _ = model(
            normalized(torch.zeros(2)),
            torch.zeros(2),
            normalized(regional_runoff),
            torch.ones(2),
            torch.zeros(2, 1),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.zeros(1, 8),
            torch.zeros(1, 1),
            2,
        )
        torch.testing.assert_close(runoff(prediction), regional_runoff)

    def test_second_round_can_cross_one_unobserved_intermediate_node(self):
        common = dict(
            s_dim=1,
            y_mu=1.2,
            y_sd=0.8,
            edge_mode="uniform",
            area_correction=False,
        )
        one_round = PILRNSignal(rounds=1, **common)
        two_rounds = PILRNSignal(rounds=2, **common)
        force_neighbour_gate(one_round)
        force_neighbour_gate(two_rounds)
        obs = normalized(torch.tensor([80.0, 0.0, 0.0]))
        visible = torch.tensor([1.0, 0.0, 0.0])
        regional_runoff = torch.tensor([10.0, 20.0, 30.0])
        args = (
            obs,
            visible,
            normalized(regional_runoff),
            torch.ones(3),
            torch.zeros(3, 1),
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.zeros(2, 8),
            torch.zeros(2, 1),
            3,
        )
        pred_one, _, _ = one_round(*args)
        pred_two, _, _ = two_rounds(*args)
        self.assertAlmostEqual(float(runoff(pred_one)[2]), 30.0, places=4)
        self.assertAlmostEqual(float(runoff(pred_two)[2]), 80.0, places=4)

    def test_gradients_remain_finite(self):
        torch.manual_seed(3)
        model = PILRNSignal(
            s_dim=2,
            y_mu=1.2,
            y_sd=0.8,
            rounds=2,
            edge_mode="full",
            area_correction=True,
        )
        prediction, _, _ = model(
            normalized(torch.tensor([20.0, 50.0, 5.0])),
            torch.tensor([1.0, 0.0, 0.0]),
            normalized(torch.tensor([18.0, 22.0, 30.0])),
            torch.ones(3),
            torch.zeros(3, 2),
            torch.tensor([0, 1, 2, 1]),
            torch.tensor([1, 2, 1, 0]),
            torch.randn(4, 8),
            torch.log(torch.tensor([[2.0], [0.5], [3.0], [0.7]])),
            3,
        )
        prediction.square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
