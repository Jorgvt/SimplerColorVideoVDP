"""End-to-end image differential tests comparing SimplerColorVideoVDP against ColorVideoVDP."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import pycvvdp
from simplercolorvideovdp import ColorVideoVDP


@pytest.fixture
def device():
    return torch.device("cpu")


def test_identical_image_pairs(device):
    """Test identical reference and test images return exactly 10.0 JOD."""
    orig_cvvdp = pycvvdp.cvvdp(device=device, quiet=True)
    simpler_cvvdp = ColorVideoVDP().to(device)

    img = torch.rand((1, 3, 128, 128), dtype=torch.float32, device=device)

    orig_q, _ = orig_cvvdp.predict(img, img, dim_order="BCHW")
    simpler_q = simpler_cvvdp(img, img)

    orig_val = orig_q.item() if isinstance(orig_q, torch.Tensor) else float(orig_q)
    simpler_val = simpler_q.item() if isinstance(simpler_q, torch.Tensor) else float(simpler_q)

    assert abs(orig_val - 10.0) < 1e-4
    assert abs(simpler_val - 10.0) < 1e-4
    assert abs(orig_val - simpler_val) < 1e-5


def test_synthetic_distortions_parity(device):
    """Test image pairs with noise, blur, and color distortions across different resolutions."""
    orig_cvvdp = pycvvdp.cvvdp(device=device, quiet=True)
    simpler_cvvdp = ColorVideoVDP().to(device)

    torch.manual_seed(42)

    for shape in [(1, 3, 64, 64), (1, 3, 128, 192), (1, 3, 200, 200)]:
        ref = torch.rand(shape, dtype=torch.float32, device=device) * 0.8 + 0.1

        # Test 1: Gaussian noise
        test_noise = (ref + torch.randn_like(ref) * 0.05).clamp(0.0, 1.0)
        orig_q_noise, _ = orig_cvvdp.predict(test_noise, ref, dim_order="BCHW")
        simpler_q_noise = simpler_cvvdp(test_noise, ref)

        diff_noise = abs(orig_q_noise.item() - simpler_q_noise.item())
        assert diff_noise < 1e-5, f"Noise test on shape {shape} diff={diff_noise} (orig={orig_q_noise.item()}, simpler={simpler_q_noise.item()})"

        # Test 2: Contrast shift
        test_contrast = (ref * 0.85).clamp(0.0, 1.0)
        orig_q_contrast, _ = orig_cvvdp.predict(test_contrast, ref, dim_order="BCHW")
        simpler_q_contrast = simpler_cvvdp(test_contrast, ref)

        diff_contrast = abs(orig_q_contrast.item() - simpler_q_contrast.item())
        assert diff_contrast < 1e-5, f"Contrast test on shape {shape} diff={diff_contrast}"

        # Test 3: Chromatic shift
        test_color = ref.clone()
        test_color[:, 0, :, :] = (test_color[:, 0, :, :] * 1.1).clamp(0.0, 1.0)
        orig_q_color, _ = orig_cvvdp.predict(test_color, ref, dim_order="BCHW")
        simpler_q_color = simpler_cvvdp(test_color, ref)

        diff_color = abs(orig_q_color.item() - simpler_q_color.item())
        assert diff_color < 1e-5, f"Color shift test on shape {shape} diff={diff_color}"


def test_real_media_parity(device):
    """Test on example images from original repository."""
    media_dir = Path(__file__).resolve().parent.parent.parent / "ColorVideoVDP" / "example_media"
    
    img_files = ["tree.jpg", "SIGGRAPH_wordcloud.png", "wavy_facade.png"]
    
    orig_cvvdp = pycvvdp.cvvdp(device=device, quiet=True)
    simpler_cvvdp = ColorVideoVDP().to(device)

    for fname in img_files:
        fpath = media_dir / fname
        if not fpath.exists():
            continue

        pil_img = Image.open(fpath).convert("RGB")
        # Resize to moderate size for quick testing
        pil_img = pil_img.resize((256, 256))
        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        ref_tensor = torch.tensor(np_img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Distort with additive noise
        test_tensor = (ref_tensor + torch.randn_like(ref_tensor) * 0.03).clamp(0.0, 1.0)

        orig_q, _ = orig_cvvdp.predict(test_tensor, ref_tensor, dim_order="BCHW")
        simpler_q = simpler_cvvdp(test_tensor, ref_tensor)

        diff = abs(orig_q.item() - simpler_q.item())
        assert diff < 1e-5, f"Real image {fname} diff={diff} (orig={orig_q.item()}, simpler={simpler_q.item()})"


def test_different_display_models(device):
    """Test parity across different display models."""
    for display in ["standard_4k", "standard_fhd", "standard_phone"]:
        orig_cvvdp = pycvvdp.cvvdp(display_name=display, device=device, quiet=True)
        simpler_cvvdp = ColorVideoVDP(display_name=display).to(device)

        torch.manual_seed(123)
        ref = torch.rand((1, 3, 100, 100), dtype=torch.float32, device=device)
        test = (ref + torch.randn_like(ref) * 0.04).clamp(0.0, 1.0)

        orig_q, _ = orig_cvvdp.predict(test, ref, dim_order="BCHW")
        simpler_q = simpler_cvvdp(test, ref)

        diff = abs(orig_q.item() - simpler_q.item())
        assert diff < 1e-5, f"Display {display} diff={diff} (orig={orig_q.item()}, simpler={simpler_q.item()})"
