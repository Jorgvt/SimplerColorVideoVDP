"""Unit tests for HybridColorVideoVDP (Calibrated Isotropic Backbone + Parallel Steerable MT Stream)."""

import pytest
import torch

from simplercolorvideovdp.metric import HybridColorVideoVDP


@pytest.fixture
def device():
    return torch.device("cpu")


def test_hybrid_colorvideovdp_image_forward(device):
    """Verifies that HybridColorVideoVDP reduces to isotropic ColorVideoVDP on single images."""
    metric = HybridColorVideoVDP(motion_weight=500.0).to(device)

    # 1 image pair: (B=1, C=3, T=1, H=64, W=64)
    test_img = torch.rand((1, 3, 1, 64, 64), device=device)
    ref_img = torch.rand((1, 3, 1, 64, 64), device=device)

    jod, stats = metric(test_img, ref_img, return_stats=True)
    assert isinstance(jod, torch.Tensor)
    assert 0.0 <= float(jod.item()) <= 10.0
    assert stats["raw_q_motion"] == 0.0
    assert stats["raw_q_total"] == stats["raw_q_iso"]


def test_hybrid_colorvideovdp_video_forward(device):
    """Verifies that HybridColorVideoVDP activates both isotropic and MT streams on videos."""
    metric = HybridColorVideoVDP(motion_weight=500.0, fps=30.0).to(device)

    # Short video pair: (B=1, C=3, T=8, H=64, W=64)
    test_vid = torch.rand((1, 3, 8, 64, 64), device=device)
    ref_vid = torch.rand((1, 3, 8, 64, 64), device=device)

    jod, stats = metric(test_vid, ref_vid, fps=30.0, return_stats=True)
    assert isinstance(jod, torch.Tensor)
    assert 0.0 <= float(jod.item()) <= 10.0
    assert stats["raw_q_iso"] > 0.0
    assert stats["raw_q_motion"] >= 0.0
    assert stats["raw_q_total"] >= stats["raw_q_iso"]
