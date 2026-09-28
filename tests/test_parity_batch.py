"""Batched inference differential tests comparing SimplerColorVideoVDP against ColorVideoVDP."""

import pytest
import torch
import pycvvdp
from simplercolorvideovdp import ColorVideoVDP


@pytest.fixture
def device():
    return torch.device("cpu")


def test_batch_of_images_parity(device):
    """Test batched image pairs."""
    orig_cvvdp = pycvvdp.cvvdp(device=device, quiet=True)
    simpler_cvvdp = ColorVideoVDP().to(device)

    torch.manual_seed(101)
    b = 4
    ref = torch.rand((b, 3, 64, 64), dtype=torch.float32, device=device) * 0.8 + 0.1
    test = (ref + torch.randn_like(ref) * 0.04).clamp(0.0, 1.0)

    orig_q, _ = orig_cvvdp.predict(test, ref, dim_order="BCHW")
    simpler_q = simpler_cvvdp(test, ref)

    # Convert to tensors
    if not isinstance(orig_q, torch.Tensor):
        orig_q = torch.tensor(orig_q, device=device)
    if not isinstance(simpler_q, torch.Tensor):
        simpler_q = torch.tensor(simpler_q, device=device)

    max_diff = (orig_q - simpler_q).abs().max().item()
    assert max_diff < 1e-5, f"Batch image diff={max_diff}"

    # Verify each item individually matches the batch slice
    for i in range(b):
        single_simpler = simpler_cvvdp(test[i : i + 1], ref[i : i + 1])
        assert abs(single_simpler.item() - simpler_q[i].item()) < 1e-6


def test_loss_function_gradient(device):
    """Verify that SimplerColorVideoVDP.loss produces meaningful gradients for optimization."""
    simpler_cvvdp = ColorVideoVDP().to(device)

    ref = torch.rand((1, 3, 64, 64), dtype=torch.float32, device=device)
    test = ref.clone().detach().requires_grad_(True)

    loss = simpler_cvvdp.loss(test, ref)
    loss.backward()

    assert test.grad is not None
    assert not torch.isnan(test.grad).any()
    assert not torch.isinf(test.grad).any()
