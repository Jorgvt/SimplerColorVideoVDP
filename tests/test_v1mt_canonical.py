"""Tests for Canonical V1-MT Cortical Model and Canonical Hybrid Metric."""

import math
import torch
import pytest
from simplercolorvideovdp.v1mt_canonical import (
    CanonicalHybridColorVideoVDP,
    CanonicalV1MTModel,
    MTIntegrationTorch,
    MTNormalizationTorch,
    SustainedFormModelTorch,
    V1FilterLayerTorch,
    V1NormalizationTorch,
    compute_velocity_plane_weights_torch,
    pad_symmetric_5d,
    rgb_to_dkl_torch,
    synthesize_2d_gabor_kernels_torch,
    synthesize_3d_gabor_kernels_torch,
)


def test_rgb_to_dkl_torch():
    rgb = torch.rand(2, 5, 16, 16, 3)
    dkl = rgb_to_dkl_torch(rgb)
    assert dkl.shape == (2, 5, 16, 16, 3)
    assert not torch.isnan(dkl).any()


def test_pad_symmetric_5d():
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 2, 2)
    padded = pad_symmetric_5d(x, pad_t=(1, 1), pad_h=(1, 1), pad_w=(1, 1))
    assert padded.shape == (1, 1, 5, 4, 4)
    assert not torch.isnan(padded).any()


def test_synthesize_3d_gabor_kernels():
    theta = torch.tensor([0.0, math.pi / 2])
    sf = torch.tensor([1.0])
    tf = torch.tensor([4.0])
    sigma_x = torch.tensor([0.2])
    sigma_y = torch.tensor([0.2])
    sigma_t = torch.tensor([0.05])
    kernels = synthesize_3d_gabor_kernels_torch(
        theta, sf, tf, sigma_x, sigma_y, sigma_t,
        kernel_size=(5, 5, 5), pixels_per_degree=30.0, fps=240.0
    )
    # 2 directions * 2 phases (even, odd) = 4
    assert kernels.shape == (4, 1, 5, 5, 5)
    assert not torch.isnan(kernels).any()


def test_synthesize_2d_gabor_kernels():
    theta = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
    sf = torch.tensor([1.5])
    sigma_x = torch.tensor([0.133])
    sigma_y = torch.tensor([0.133])
    kernels = synthesize_2d_gabor_kernels_torch(
        theta, sf, sigma_x, sigma_y,
        kernel_size=(7, 7), pixels_per_degree=30.0
    )
    # 4 directions * 2 phases = 8
    assert kernels.shape == (8, 1, 7, 7)
    assert not torch.isnan(kernels).any()


def test_v1_filter_and_normalization():
    v1_filter = V1FilterLayerTorch(
        directions=(0.0, math.pi / 2),
        sf=1.0,
        tf=4.0,
        kernel_size=(5, 5, 5),
        pixels_per_degree=30.0,
        fps=240.0
    )
    v1_norm = V1NormalizationTorch(spatial_pool_size=(3, 3))

    lum = torch.rand(1, 1, 8, 16, 16)
    energy = v1_filter(lum)
    assert energy.shape == (1, 2, 8, 16, 16)

    norm_out = v1_norm(energy)
    assert norm_out.shape == (1, 2, 8, 16, 16)
    assert (norm_out >= 0).all()


def test_mt_integration_and_normalization():
    v1_filter = V1FilterLayerTorch(
        directions=(0.0, math.pi / 2),
        kernel_size=(5, 5, 5)
    )
    mt_integ = MTIntegrationTorch(
        v1_filter=v1_filter,
        mt_directions=(0.0, math.pi / 2),
        mt_speeds=(1.0, 4.0, 16.0),
        spatial_pool_size=(3, 3)
    )
    mt_norm = MTNormalizationTorch()

    v1_norm = torch.rand(1, 2, 8, 16, 16)
    mt_lin = mt_integ(v1_norm)
    # 3 speeds * 2 directions = 6 channels
    assert mt_lin.shape == (1, 6, 8, 16, 16)

    mt_out = mt_norm(mt_lin)
    assert mt_out.shape == (1, 6, 8, 16, 16)
    assert (mt_out >= 0).all()


def test_sustained_form_model():
    form_model = SustainedFormModelTorch(
        directions=(0.0, math.pi / 2),
        kernel_size=(5, 5),
        spatial_pool_size=(3, 3)
    )
    dkl_video = torch.rand(1, 6, 16, 16, 3)
    form_out = form_model(dkl_video, batch_t=4)
    # Shape: (B, C=3, num_dirs=2, T=6, H=16, W=16)
    assert form_out.shape == (1, 3, 2, 6, 16, 16)
    assert (form_out >= 0).all()


def test_canonical_v1mt_model_difference():
    model = CanonicalV1MTModel(
        fps=240.0,
        pixels_per_degree=30.0,
        initial_speeds=(1.0, 4.0, 16.0),
        v1_kernel_size=(5, 5, 5)
    )
    # Small test video: 8 frames, 24x24
    ref = torch.rand(8, 24, 24, 3)
    dist = ref + 0.05 * torch.randn(8, 24, 24, 3)
    dist = torch.clamp(dist, 0.0, 1.0)

    diff, stats = model.compute_difference(ref, dist, chunk_size=4)
    assert isinstance(diff, torch.Tensor)
    assert diff.ndim == 0  # scalar
    assert diff.item() >= 0.0
    assert "mae_slow" in stats
    assert "mae_med" in stats
    assert "mae_fast" in stats
    assert "mae_fa" in stats
    assert "mae_frg" in stats
    assert "mae_fby" in stats


def test_canonical_hybrid_model():
    model = CanonicalHybridColorVideoVDP(
        fps=240.0,
        v1_kernel_size=(5, 5, 5),
        init_w_cvvdp=0.6,
        init_w_v1mt=0.4,
        jod_intercept=10.0,
        jod_slope=2.0
    )
    d_c = torch.tensor([0.2, 0.5, 1.0])
    d_v = torch.tensor([0.1, 0.3, 0.8])

    pred_jod, d_hybrid = model.forward_from_features(d_c, d_v)
    assert pred_jod.shape == (3,)
    assert d_hybrid.shape == (3,)
    # Higher difference should result in lower JOD
    assert pred_jod[0] > pred_jod[1] > pred_jod[2]
