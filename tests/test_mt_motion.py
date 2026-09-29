"""Unit tests for Steerable MT motion processing in simplercolorvideovdp."""

import math
import pytest
import torch

from simplercolorvideovdp.mt import (
    SteerableMTIntegration,
    SteerableMTStage,
    compute_steerable_velocity_plane_weights,
)
from simplercolorvideovdp.metric import V1MTColorVideoVDP


def test_velocity_plane_weights_matching():
    scale_freqs = torch.tensor([1.5, 3.0])
    orientations = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
    temp_freqs = torch.tensor([0.0, 5.0])

    mt_dirs = torch.tensor([0.0])
    mt_speeds = torch.tensor([1.6667])

    w = compute_steerable_velocity_plane_weights(
        scale_freqs, orientations, temp_freqs, mt_dirs, mt_speeds, sigma_p=0.5
    )

    assert w.shape == (16, 1)
    # Matching channel: scale=3.0, ori=0, temp=5.0 -> idx 9
    assert w[9, 0].item() > 0.3
    # Off-plane channel: scale=3.0, ori=pi/2, temp=5.0 -> idx 13
    assert w[13, 0].item() < 1e-10


def test_v1_mt_metric_end_to_end():
    metric = V1MTColorVideoVDP(ppd=30.0, fps=30.0)

    ref_vid = torch.rand((1, 3, 5, 32, 32), dtype=torch.float32)
    test_vid = (ref_vid + 0.05 * torch.randn_like(ref_vid)).clamp(0, 1).requires_grad_(True)

    jod_identical = metric(ref_vid, ref_vid).item()
    assert pytest.approx(10.0, abs=1e-3) == jod_identical

    jod_dist = metric(test_vid, ref_vid).item()
    assert jod_dist < 10.0

    loss = metric.loss(test_vid, ref_vid)
    loss.backward()

    assert test_vid.grad is not None
    assert not torch.isnan(test_vid.grad).any()
