"""Component-level differential tests comparing SimplerColorVideoVDP against ColorVideoVDP."""

import pytest
import torch
import numpy as np

# Original modules
import pycvvdp
from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.lpyr_dec import weber_contrast_pyr
from pycvvdp.csf import castleCSF

# Simpler modules
from simplercolorvideovdp.display import load_display_model, DisplayPhotometry, DisplayGeometry
from simplercolorvideovdp.colorspace import get_rgb_to_dkl_matrix, linear_rgb_to_dkl
from simplercolorvideovdp.pyramid import WeberLaplacianPyramid
from simplercolorvideovdp.csf import CastleCSF
from simplercolorvideovdp.temporal import compute_temporal_filters


def test_color_matrix_parity():
    """Verify linear RGB to DKL matrix equivalence."""
    device = torch.device("cpu")
    orig_dm = vvdp_display_photometry.load("standard_4k", [])
    
    rgb_test = torch.rand((1, 3, 1, 64, 64), dtype=torch.float32, device=device)
    
    # Original conversion
    orig_dkl = orig_dm.linear_2_target_colorspace(rgb_test, "DKLd65")
    
    # Simpler conversion
    simpler_dkl = linear_rgb_to_dkl(rgb_test, "sRGB")
    
    max_diff = (orig_dkl - simpler_dkl).abs().max().item()
    assert max_diff < 1e-6, f"DKL conversion difference {max_diff} exceeds tolerance"


def test_pyramid_decomposition_parity():
    """Verify Weber Laplacian pyramid band-by-band decomposition."""
    device = torch.device("cpu")
    w, h = 128, 128
    ppd = 60.0
    
    orig_pyr = weber_contrast_pyr(w, h, ppd, device, contrast="weber_g1")
    simpler_pyr = WeberLaplacianPyramid(w, h, ppd).to(device)
    
    assert orig_pyr.get_band_count() == simpler_pyr.band_count
    np.testing.assert_allclose(orig_pyr.get_freqs(), simpler_pyr.get_freqs(), rtol=1e-5)
    
    torch.manual_seed(42)
    # Interleaved tensor (B, 6, 1, H, W)
    r_tensor = torch.rand((1, 6, 1, h, w), dtype=torch.float32, device=device) + 0.1
    
    orig_lpyr, orig_lbkg = orig_pyr.decompose(r_tensor)
    simpler_lpyr, simpler_lbkg = simpler_pyr.decompose(r_tensor)
    
    assert len(orig_lpyr) == len(simpler_lpyr)
    
    for b in range(len(orig_lpyr)):
        # Note: orig_pyr.get_band applies the band multiplier (1 or 2)
        orig_band = orig_pyr.get_band(orig_lpyr, b)
        simpler_band = simpler_lpyr[b]
        
        diff_band = (orig_band - simpler_band).abs().max().item()
        assert diff_band < 1e-5, f"Pyramid band {b} difference {diff_band} exceeds tolerance"
        
        diff_lbkg = (orig_lbkg[b] - simpler_lbkg[b]).abs().max().item()
        assert diff_lbkg < 1e-5, f"Pyramid background {b} difference {diff_lbkg} exceeds tolerance"


def test_csf_lookup_parity():
    """Verify CastleCSF sensitivity evaluation across various frequencies and background luminances."""
    device = torch.device("cpu")
    orig_csf = castleCSF(csf_version="weber_fixed_size", device=device, config_paths=[])
    simpler_csf = CastleCSF().to(device)
    
    log_l_bkg = torch.linspace(-1.0, 3.0, 100, device=device).reshape(1, 1, 1, 10, 10)
    
    for rho in [0.5, 2.0, 8.0, 16.0, 30.0]:
        for omega, tch in [(0, 0), (5, 1)]:
            for cch in range(3 if tch == 0 else 1):
                orig_s = orig_csf.sensitivity(rho, omega, log_l_bkg, cch, sigma=-1.5)
                simpler_s = simpler_csf.sensitivity(rho, omega, log_l_bkg, cch, sensitivity_correction_db=0.0)
                
                max_diff = (orig_s - simpler_s).abs().max().item()
                rel_diff = ((orig_s - simpler_s).abs() / orig_s).max().item()
                assert rel_diff < 1e-5, f"CSF diff at rho={rho}, omega={omega}, ch={cch}: rel_diff={rel_diff}"


def test_temporal_filter_parity():
    """Verify temporal filter impulse responses."""
    device = torch.device("cpu")
    orig_metric = pycvvdp.cvvdp(device=device, quiet=True)
    
    for fps in [24.0, 30.0, 60.0, 120.0]:
        orig_f, orig_omega = orig_metric.get_temporal_filters(fps)
        simpler_f, n = compute_temporal_filters(fps, device=device)
        
        assert len(orig_f) == len(simpler_f)
        assert len(orig_f[0]) == n
        
        for k in range(4):
            max_diff = (orig_f[k] - simpler_f[k]).abs().max().item()
            assert max_diff < 1e-5, f"Temporal filter {k} at fps={fps} difference {max_diff}"
