"""End-to-end video differential tests comparing SimplerColorVideoVDP against ColorVideoVDP."""

import pytest
import torch
import pycvvdp
from simplercolorvideovdp import ColorVideoVDP


@pytest.fixture
def device():
    return torch.device("cpu")


def test_video_parity_short_and_long(device):
    """Test video sequences across various frame counts and frame rates."""
    orig_cvvdp = pycvvdp.cvvdp(device=device, quiet=True)
    simpler_cvvdp = ColorVideoVDP().to(device)

    torch.manual_seed(42)

    configs = [
        {"fps": 30.0, "frames": 5, "h": 64, "w": 64},
        {"fps": 24.0, "frames": 10, "h": 64, "w": 64},
        {"fps": 60.0, "frames": 15, "h": 96, "w": 96},
    ]

    for cfg in configs:
        t = cfg["frames"]
        h = cfg["h"]
        w = cfg["w"]
        fps = cfg["fps"]

        # Base synthetic video: (B, C, T, H, W)
        ref = torch.rand((1, 3, t, h, w), dtype=torch.float32, device=device) * 0.8 + 0.1

        # Test video with temporal distortion (moving noise)
        test = ref.clone()
        for f in range(t):
            test[:, :, f, :, :] = (ref[:, :, f, :, :] + torch.randn((1, 3, h, w), device=device) * (0.02 * (f + 1) / t)).clamp(0.0, 1.0)

        # Original ColorVideoVDP predict with BCFHW order
        orig_q, _ = orig_cvvdp.predict(test, ref, dim_order="BCFHW", frames_per_second=fps)
        simpler_q = simpler_cvvdp(test, ref, fps=fps)

        diff = abs(orig_q.item() - simpler_q.item())
        assert diff < 1e-5, f"Video at fps={fps}, T={t} diff={diff} (orig={orig_q.item()}, simpler={simpler_q.item()})"


def test_video_padding_modes(device):
    """Verify both replicate and symmetric temporal padding modes on video sequences."""
    for pad_mode in ["replicate", "symmetric"]:
        orig_cvvdp = pycvvdp.cvvdp(temp_padding=pad_mode, device=device, quiet=True)
        simpler_cvvdp = ColorVideoVDP(temp_padding=pad_mode).to(device)

        torch.manual_seed(99)
        ref = torch.rand((1, 3, 8, 64, 64), dtype=torch.float32, device=device) * 0.7 + 0.15
        test = (ref + torch.randn_like(ref) * 0.03).clamp(0.0, 1.0)

        orig_q, _ = orig_cvvdp.predict(test, ref, dim_order="BCFHW", frames_per_second=30.0)
        simpler_q = simpler_cvvdp(test, ref, fps=30.0)

        diff = abs(orig_q.item() - simpler_q.item())
        assert diff < 1e-5, f"Padding {pad_mode} diff={diff} (orig={orig_q.item()}, simpler={simpler_q.item()})"
