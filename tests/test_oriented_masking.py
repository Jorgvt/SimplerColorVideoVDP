"""Unit tests for OrientedContrastMasking in simplercolorvideovdp."""

import pytest
import torch

from simplercolorvideovdp.masking import OrientedContrastMasking
from simplercolorvideovdp.metric import OrientedColorVideoVDP


def test_oriented_masking_cross_orientation():
    b, c, t, k, h, w = 1, 1, 1, 4, 64, 64
    masking = OrientedContrastMasking(cross_orientation_weight=0.2)

    test_sig = torch.zeros((b, c, t, k, h, w))
    test_sig[0, 0, 0, 0, :, :] = 0.2
    ref_sig = torch.zeros((b, c, t, k, h, w))
    s = torch.ones((b, c, t, 1, h, w)) * 10.0

    # Parallel mask
    test_parallel = test_sig.clone()
    ref_parallel = ref_sig.clone()
    test_parallel[0, 0, 0, 0, :, :] += 0.8
    ref_parallel[0, 0, 0, 0, :, :] += 0.8
    d_parallel = masking(test_parallel, ref_parallel, s)

    # Orthogonal mask
    test_ortho = test_sig.clone()
    ref_ortho = ref_sig.clone()
    test_ortho[0, 0, 0, 2, :, :] += 0.8
    ref_ortho[0, 0, 0, 2, :, :] += 0.8
    d_ortho = masking(test_ortho, ref_ortho, s)

    # Parallel mask suppresses distortion more -> lower difference d
    assert d_parallel[0, 0, 0, 0].mean().item() < d_ortho[0, 0, 0, 0].mean().item()


def test_oriented_metric_end_to_end_and_gradient():
    metric = OrientedColorVideoVDP(ppd=40.0)

    img = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    jod_identical = metric(img, img)
    assert pytest.approx(10.0, abs=1e-3) == jod_identical.item()

    test_img = (img + 0.1 * torch.randn_like(img)).clamp(0, 1).requires_grad_(True)
    loss = metric.loss(test_img, img)
    loss.backward()

    assert test_img.grad is not None
    assert not torch.isnan(test_img.grad).any()
