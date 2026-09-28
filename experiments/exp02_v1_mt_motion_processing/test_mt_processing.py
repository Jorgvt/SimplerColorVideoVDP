"""Pytest suite for V1-MT cortical motion processing stage."""

import math
import pytest
import torch

from experiments.exp02_v1_mt_motion_processing.metric_v1_mt import V1MTColorVideoVDP
from experiments.exp02_v1_mt_motion_processing.mt import (
    MTIntegration,
    MTNormalization,
    MTStage,
    compute_velocity_plane_weights,
)
from experiments.exp02_v1_mt_motion_processing.v1 import V1Stage
from experiments.exp02_v1_mt_motion_processing.v1_mt import V1MTModel


def test_velocity_plane_weights():
    """Verifies that velocity plane weights peak when tf = v * sf * cos(theta_v1 - theta_mt)."""
    v1_dirs = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
    v1_sf = torch.full((5,), 1.5)
    v1_tf = torch.full((5,), 2.4)

    mt_dirs = torch.tensor([0.0])  # Rightward MT channel
    mt_speed = torch.tensor([1.6])  # 1.6 deg/s -> dot_product = 1.6 * 1.5 * cos(0) = 2.4 (exact match for theta=0!)

    w = compute_velocity_plane_weights(v1_dirs, v1_sf, v1_tf, mt_dirs, mt_speed, sigma_p=0.5)

    assert w.shape == (5, 1)
    # Peak weight should be at V1 channel 0 (theta=0)
    assert torch.argmax(w[:, 0]).item() == 0
    # Orthogonal V1 channel (theta=pi/2, idx 2) has cos(pi/2)=0 -> tf - 0 = 2.4 -> very low weight
    assert w[0, 0].item() > 10.0 * w[2, 0].item()


def test_v1_direction_selectivity():
    """Verifies that a rightward drifting grating excites rightward V1 channel over leftward."""
    t_len, h, w = 9, 32, 32
    fps = 30.0
    ppd = 30.0
    sf = 1.5  # cpd
    speed = 1.6  # deg/sec -> tf = sf * speed = 2.4 Hz

    # Create rightward drifting grating: cos(2pi (sf * x - tf * t))
    ts = torch.linspace(0, (t_len - 1) / fps, t_len).view(1, t_len, 1, 1)
    xs = torch.linspace(0, (w - 1) / ppd, w).view(1, 1, 1, w)
    grating_right = torch.cos(2.0 * math.pi * (sf * xs - (sf * speed) * ts)).expand(1, t_len, h, w)

    v1 = V1Stage(
        directions=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
        sf=sf,
        tf=sf * speed,
        kernel_size=(7, 11, 11),
        pixels_per_degree=ppd,
        fps=fps,
    )

    v1_norm = v1(grating_right)  # (1, 4, T, H, W)
    mean_energy = torch.mean(v1_norm, dim=(-3, -2, -1))[0]  # Shape (4,) [Right, Up, Left, Down]

    # Rightward channel (idx 0) has peak response; orthogonal channels (idx 1, 3) near zero; leftward (idx 2) suppressed
    assert torch.argmax(mean_energy).item() == 0
    assert mean_energy[0].item() > mean_energy[2].item() * 3.0
    assert mean_energy[0].item() > mean_energy[1].item() * 20.0


def test_mt_pattern_motion_aperture_resolution():
    """Simoncelli-Heeger 1998 test: A plaid of two gratings moving at +/- 45 deg resolves to 0 deg global motion in MT."""
    t_len, h, w = 9, 32, 32
    fps = 30.0
    ppd = 30.0
    sf = 1.5
    speed = 1.6

    ts = torch.linspace(0, (t_len - 1) / fps, t_len).view(1, t_len, 1, 1)
    xs = torch.linspace(0, (w - 1) / ppd, w).view(1, 1, 1, w)
    ys = torch.linspace(0, (h - 1) / ppd, h).view(1, 1, h, 1)

    # Component 1: drifting at +45 deg (Up-Right)
    theta1 = math.pi / 4.0
    comp1 = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(theta1) + ys * math.sin(theta1)) - (sf * speed) * ts))

    # Component 2: drifting at -45 deg (Down-Right)
    theta2 = -math.pi / 4.0
    comp2 = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(theta2) + ys * math.sin(theta2)) - (sf * speed) * ts))

    # Combined Type I Plaid: True global motion vector is Rightward (0 deg) with speed = speed / cos(45 deg)
    plaid = 0.5 * (comp1 + comp2).expand(1, t_len, h, w)

    v1_dirs = [k * math.pi / 4.0 for k in range(8)]
    mt_dirs = [k * math.pi / 4.0 for k in range(8)]

    model = V1MTModel(
        v1_directions=v1_dirs,
        v1_sf=sf,
        v1_tf=sf * speed,
        v1_kernel_size=(7, 11, 11),
        mt_directions=mt_dirs,
        mt_speed=speed * math.sqrt(2.0),
        mt_spatial_pool_size=(9, 9),
        pixels_per_degree=ppd,
        fps=fps,
    )

    v1_norm, mt_norm, flow = model.forward_all(plaid)

    # MT response averaged over time and space: shape (8,)
    mt_mean = torch.mean(mt_norm, dim=(-3, -2, -1))[0]

    # Global pattern motion direction is 0 rad (idx 0)
    assert torch.argmax(mt_mean).item() == 0
    # Decoded flow should have positive vx (rightward) and near-zero vy
    mean_vx = flow[0, 0].mean().item()
    mean_vy = flow[0, 1].mean().item()
    assert mean_vx > 0.5
    assert abs(mean_vy) < 0.2


def test_dual_stream_metric_execution_and_differentiability():
    """Verifies end-to-end forward/loss pass on video pair with differentiability."""
    metric = V1MTColorVideoVDP(ppd=30.0, fps=30.0)

    # Video pair of 5 frames
    ref_vid = torch.rand((1, 3, 5, 32, 32), dtype=torch.float32)
    test_vid = (ref_vid + 0.05 * torch.randn_like(ref_vid)).clamp(0, 1).requires_grad_(True)

    # Identical video pair -> JOD == 10.0
    jod_identical = metric(ref_vid, ref_vid).item()
    assert pytest.approx(10.0, abs=1e-3) == jod_identical

    # Distorted video -> JOD < 10.0
    jod_dist = metric(test_vid, ref_vid).item()
    assert jod_dist < 10.0

    # Differentiability
    loss = metric.loss(test_vid, ref_vid)
    loss.backward()

    assert test_vid.grad is not None
    assert not torch.isnan(test_vid.grad).any()
    assert not torch.isinf(test_vid.grad).any()
