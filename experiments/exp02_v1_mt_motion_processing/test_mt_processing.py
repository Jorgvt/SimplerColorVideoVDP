"""Tests for Unified Steerable-driven MT Cortical Motion Processing Stage using the core library."""

import math
import pytest
import torch

from simplercolorvideovdp import (
    SteerableMTIntegration,
    SteerableMTStage,
    SteerableWeberPyramid,
    V1MTColorVideoVDP,
    apply_temporal_filtering,
    compute_steerable_velocity_plane_weights,
)


def test_steerable_velocity_plane_weights():
    """Verifies that velocity plane weights peak when omega = v * rho * cos(theta_k - theta_mt)."""
    scale_freqs = torch.tensor([1.5, 3.0])
    orientations = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
    temp_freqs = torch.tensor([0.0, 5.0])

    mt_dirs = torch.tensor([0.0])
    mt_speeds = torch.tensor([1.6667])

    w = compute_steerable_velocity_plane_weights(
        scale_freqs, orientations, temp_freqs, mt_dirs, mt_speeds, sigma_p=0.5
    )

    assert w.shape == (16, 1)
    # The exact matching channel: scale=3.0, ori=0, temp=5.0 -> flat idx 9
    assert w[9, 0].item() > 0.3
    # Off-plane channel: scale=3.0, ori=pi/2, temp=5.0 -> flat idx 13
    assert w[13, 0].item() < 1e-10


def test_steerable_mt_aperture_plaid():
    """Verifies that MT resolves the aperture problem on a Type I Plaid using Steerable subbands."""
    t_len, h, w = 15, 64, 64
    fps = 30.0
    ppd = 30.0
    sf = 1.5
    speed = 5.0 / sf

    ts = torch.linspace(0, (t_len - 1) / fps, t_len).view(1, 1, t_len, 1, 1)
    xs = torch.linspace(0, (w - 1) / ppd, w).view(1, 1, 1, 1, w)
    ys = torch.linspace(0, (h - 1) / ppd, h).view(1, 1, 1, h, 1)

    # Component 1: drifting at +45 deg (Down-Right)
    th1 = math.pi / 4.0
    comp1 = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(th1) + ys * math.sin(th1)) - 5.0 * ts))

    # Component 2: drifting at -45 deg (Up-Right)
    th2 = -math.pi / 4.0
    comp2 = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(th2) + ys * math.sin(th2)) - 5.0 * ts))

    plaid = 0.5 * (comp1 + comp2).repeat(1, 3, 1, 1, 1)

    # Temporal filtering
    dkl_dummy = plaid
    r = apply_temporal_filtering(dkl_dummy, dkl_dummy, fps=fps, temp_padding="replicate")

    # Steerable pyramid decomposition
    pyr = SteerableWeberPyramid(width=w, height=h, ppd=ppd, num_orientations=4)
    b_bands, _ = pyr.decompose(r)

    # Collect achromatic subbands across scales
    subbands = []
    for bb in range(pyr.num_levels):
        b_bb = b_bands[bb]
        sust_ach = b_bb[:, 0:1, ...]
        trans_ach = b_bb[:, 3:4, ...]
        scale_sub = torch.cat([sust_ach, trans_ach], dim=1).permute(0, 3, 1, 2, 4, 5).reshape(1, -1, t_len, h, w)
        subbands.append(scale_sub)

    stacked_v1 = torch.cat(subbands, dim=1)

    # MT Stage
    mt_dirs = [k * math.pi / 4.0 for k in range(8)]
    mt_stage = SteerableMTStage(
        scale_freqs=pyr.get_freqs()[:-1],
        orientations=pyr.orientations,
        temp_freqs=(0.0, 5.0),
        mt_directions=mt_dirs,
        mt_speeds=(speed * math.sqrt(2.0),),
        spatial_pool_size=(9, 9),
        sigma_p=1.0,
    )

    mt_norm = mt_stage(stacked_v1)
    mt_mean = torch.mean(mt_norm[:, :, 3:-3], dim=(-3, -2, -1))[0]

    # MT pattern response on horizontal motion axis is active
    assert mt_norm.shape == (1, 8, t_len, h, w)
    assert mt_mean[0].item() > 0.05
    assert mt_mean[4].item() > 0.05


def test_unified_metric_end_to_end_and_differentiability():
    """Verifies that the unified V1MTColorVideoVDP evaluates video pairs and is differentiable."""
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
    assert not torch.isinf(test_vid.grad).any()
