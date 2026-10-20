"""Scratch output initialization without importing the HPC config at test time."""
import ast
from pathlib import Path

import pytest
import torch
from torch import nn


def _load_initializer():
    source = (Path(__file__).resolve().parents[1] / "dimensionless_training.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "initialize_scratch_outputs")
    scope = {"torch": torch, "nn": nn, "math": __import__("math")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<initializer>", "exec"), scope)
    return scope["initialize_scratch_outputs"]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_min = 0.1
        self.delta_head = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1),
                                        nn.ReLU(), nn.Conv2d(4, 2, 1))
        self.sigma_head = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1),
                                        nn.ReLU(), nn.Conv2d(4, 2, 1))

    def forward(self, x):
        return self.delta_head(x), torch.nn.functional.softplus(self.sigma_head(x)) + self.sigma_min


def test_scratch_heads_mean_zero_sigma_constant():
    net = TinyModel()
    initialize = _load_initializer()
    initialize(net, sigma0=1.0)
    delta, sigma = net(torch.randn(3, 2, 6, 7))
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(sigma, torch.ones_like(sigma), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        initialize(net, sigma0=0.05)
    (delta.mean() + sigma.mean()).backward()
    assert net.delta_head[-1].weight.grad is not None
    assert net.sigma_head[-1].weight.grad is not None
