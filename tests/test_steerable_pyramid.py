"""Unit tests for SteerableWeberPyramid in simplercolorvideovdp."""

import math
import torch
import pytest

from simplercolorvideovdp.pyramid import SteerableWeberPyramid


def test_steerable_pyramid_decomposition_shapes():
    b, c, t, h, w = 2, 6, 1, 128, 128
    ppd = 60.0
    pyr = SteerableWeberPyramid(width=w, height=h, ppd=ppd, num_orientations=4)

    dummy_input = torch.rand((b, c, t, h, w), dtype=torch.float32) + 0.1
    bands, log_bkg = pyr.decompose(dummy_input)

    assert len(bands) == pyr.band_count
    assert len(log_bkg) == pyr.band_count

    # Intermediate bands: K=4 orientations with octave downsampling
    for level in range(pyr.num_levels):
        h_l, w_l = pyr.level_sizes[level]
        assert bands[level].shape == (b, c, t, 4, h_l, w_l)
        assert log_bkg[level].shape == (b, 2, t, h_l, w_l)

    # Baseband: K=1 isotropic orientation at base resolution
    h_base, w_base = pyr.base_size
    assert bands[-1].shape == (b, c, t, 1, h_base, w_base)
    assert log_bkg[-1].shape == (b, 2, t, 1, 1)


def test_steerable_pyramid_orientation_tuning():
    h, w = 128, 128
    ppd = 60.0
    pyr = SteerableWeberPyramid(width=w, height=h, ppd=ppd, num_orientations=4)

    # Vertical stripes (theta=0, horizontal frequencies)
    x = torch.linspace(0, 8 * 2 * math.pi, w).unsqueeze(0).repeat(h, 1)
    grating_vert = (0.5 + 0.3 * torch.sin(x)).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1, 1)

    bands_vert, _ = pyr.decompose(grating_vert)
    energy_vert = torch.mean(bands_vert[1] ** 2, dim=(-2, -1))[0, 0, 0]
    assert torch.argmax(energy_vert).item() == 0

    # Diagonal stripes (theta=pi/4)
    y = torch.linspace(0, 8 * 2 * math.pi, h).unsqueeze(1).repeat(1, w)
    grating_diag = (0.5 + 0.3 * torch.sin(x + y)).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1, 1)

    bands_diag, _ = pyr.decompose(grating_diag)
    energy_diag = torch.mean(bands_diag[1] ** 2, dim=(-2, -1))[0, 0, 0]
    assert torch.argmax(energy_diag).item() == 1
