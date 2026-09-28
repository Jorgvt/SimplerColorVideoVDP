"""Tests for Orientation-Selective Steerable ColorVideoVDP."""

import math
import numpy as np
import pytest
import torch

from simplercolorvideovdp.metric import ColorVideoVDP
from experiments.exp01_orientation_selective_pyramid.masking_oriented import OrientedContrastMasking
from experiments.exp01_orientation_selective_pyramid.metric_oriented import OrientedColorVideoVDP
from experiments.exp01_orientation_selective_pyramid.steerable_pyramid import SteerableWeberPyramid


def test_steerable_pyramid_shapes():
    """Verifies output shapes of steerable pyramid decomposition."""
    b, c_all, t, h, w = 2, 6, 1, 128, 128
    ppd = 60.0
    pyr = SteerableWeberPyramid(width=w, height=h, ppd=ppd, num_orientations=4)

    dummy_input = torch.rand((b, c_all, t, h, w), dtype=torch.float32) + 0.1
    bands, log_bkg = pyr.decompose(dummy_input)

    assert len(bands) == pyr.band_count
    assert len(log_bkg) == pyr.band_count

    # Intermediate bands: K=4 orientations
    for level in range(pyr.num_levels):
        assert bands[level].shape == (b, c_all, t, 4, h, w)
        assert log_bkg[level].shape == (b, 2, t, h, w)

    # Baseband: K=1 orientation (isotropic)
    assert bands[-1].shape == (b, c_all, t, 1, h, w)
    assert log_bkg[-1].shape == (b, 2, t, 1, 1)


def test_orientation_selectivity_gratings():
    """Verifies that oriented sinusoidal gratings activate the corresponding orientation subbands."""
    h, w = 128, 128
    ppd = 60.0
    pyr = SteerableWeberPyramid(width=w, height=h, ppd=ppd, num_orientations=4)

    # 1. Vertical grating (spatial variation along x, orientation angle 0 rad)
    x = torch.linspace(0, 8 * 2 * math.pi, w).unsqueeze(0).repeat(h, 1)
    grating_vert = 0.5 + 0.3 * torch.sin(x)  # (H, W)

    # 2. Diagonal grating (spatial variation along 45 deg, orientation angle pi/4 rad)
    y = torch.linspace(0, 8 * 2 * math.pi, h).unsqueeze(1).repeat(1, w)
    grating_diag = 0.5 + 0.3 * torch.sin(x + y)

    # Decompose vertical grating
    r_vert = grating_vert.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1, 1)
    bands_vert, _ = pyr.decompose(r_vert)

    # Energy across 4 orientations at intermediate band 1: (B, C, T, K, H, W)
    energy_vert = torch.mean(bands_vert[1] ** 2, dim=(-2, -1))[0, 0, 0]  # Shape (4,)
    # Peak orientation for vertical stripes is theta=0
    assert torch.argmax(energy_vert).item() == 0

    # Decompose diagonal grating
    r_diag = grating_diag.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1, 1)
    bands_diag, _ = pyr.decompose(r_diag)
    energy_diag = torch.mean(bands_diag[1] ** 2, dim=(-2, -1))[0, 0, 0]
    # Peak orientation for diagonal stripes is theta=pi/4 (idx 1)
    assert torch.argmax(energy_diag).item() == 1


def test_oriented_masking_orthogonality():
    """Verifies that parallel masks suppress distortion more than orthogonal masks."""
    b, c, t, k, h, w = 1, 1, 1, 4, 64, 64
    masking = OrientedContrastMasking(cross_orientation_weight=0.2)

    # Distortion signal at orientation 0
    test_sig = torch.zeros((b, c, t, k, h, w))
    test_sig[0, 0, 0, 0, :, :] = 0.2
    ref_sig = torch.zeros((b, c, t, k, h, w))

    s = torch.ones((b, c, t, 1, h, w)) * 10.0

    # Case 1: Mask along parallel orientation 0
    ref_parallel = ref_sig.clone()
    test_parallel = test_sig.clone()
    test_parallel[0, 0, 0, 0, :, :] += 0.8  # Strong parallel background mask
    ref_parallel[0, 0, 0, 0, :, :] += 0.8
    d_parallel = masking(test_parallel, ref_parallel, s)

    # Case 2: Mask along orthogonal orientation 2 (90 deg)
    ref_ortho = ref_sig.clone()
    test_ortho = test_sig.clone()
    test_ortho[0, 0, 0, 2, :, :] += 0.8  # Strong orthogonal background mask
    ref_ortho[0, 0, 0, 2, :, :] += 0.8
    d_ortho = masking(test_ortho, ref_ortho, s)

    # Masked difference on orientation 0: parallel mask should suppress difference more (lower d)
    diff_parallel = d_parallel[0, 0, 0, 0].mean().item()
    diff_ortho = d_ortho[0, 0, 0, 0].mean().item()

    assert diff_parallel < diff_ortho


def test_oriented_metric_end_to_end():
    """Verifies end-to-end forward pass, identical pair JOD=10, and distortion monotonicity."""
    metric = OrientedColorVideoVDP(ppd=40.0)

    # Identical image -> JOD == 10.0
    img = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    jod_identical = metric(img, img)
    assert pytest.approx(10.0, abs=1e-3) == jod_identical.item()

    # Distorted image -> JOD < 10.0
    noise_mild = img + 0.05 * torch.randn_like(img)
    noise_severe = img + 0.3 * torch.randn_like(img)

    jod_mild = metric(noise_mild.clamp(0, 1), img).item()
    jod_severe = metric(noise_severe.clamp(0, 1), img).item()

    assert jod_mild < 10.0
    assert jod_severe < jod_mild


def test_oriented_metric_differentiability():
    """Verifies that OrientedColorVideoVDP is fully differentiable."""
    metric = OrientedColorVideoVDP(ppd=40.0)

    test_img = torch.rand((1, 3, 64, 64), dtype=torch.float32, requires_grad=True)
    ref_img = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    loss = metric.loss(test_img, ref_img)
    loss.backward()

    assert test_img.grad is not None
    assert not torch.isnan(test_img.grad).any()
    assert not torch.isinf(test_img.grad).any()


def test_oriented_video_evaluation():
    """Verifies video processing (T > 1) with 4 spatiotemporal channels."""
    metric = OrientedColorVideoVDP(ppd=40.0)

    test_vid = torch.rand((1, 3, 4, 64, 64), dtype=torch.float32)
    ref_vid = torch.rand((1, 3, 4, 64, 64), dtype=torch.float32)

    jod_vid, stats = metric(test_vid, ref_vid, fps=30.0, return_stats=True)
    assert 0.0 <= jod_vid.item() <= 10.0
    assert stats["num_frames"] == 4
    assert len(stats["orientations_deg"]) == 4
