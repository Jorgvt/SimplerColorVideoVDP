"""Steerable Weber Contrast Pyramid for SimplerColorVideoVDP.

Implements a multiscale, multi-orientation steerable pyramid (Simoncelli & Freeman)
with Weber contrast normalization and tight-frame properties.
"""

from typing import List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_polar_grid(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates normalized radial frequency (r in [0, pi]) and angular (theta in [-pi, pi]) grids."""
    # Frequencies in [-pi, pi]
    y_freq = torch.fft.fftfreq(h, d=1.0, device=device) * (2.0 * math.pi)
    x_freq = torch.fft.fftfreq(w, d=1.0, device=device) * (2.0 * math.pi)

    # 2D coordinate meshgrid
    y_grid, x_grid = torch.meshgrid(y_freq, x_freq, indexing="ij")

    # Radial frequency r = sqrt(wx^2 + wy^2), bounded at max frequency pi
    r = torch.sqrt(x_grid**2 + y_grid**2)

    # Angular coordinate theta in [-pi, pi]
    # theta = 0 corresponds to positive x-axis (horizontal frequencies / vertical spatial features)
    theta = torch.atan2(y_grid, x_grid)

    return r, theta


def _steerable_radial_filters(r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes highpass, bandpass, and lowpass radial filter profiles.

    Satisfies:
        H0(r)^2 + L0(r)^2 = 1
        B(r)^2 + L(r)^2 = 1 (on the transition band)
    """
    pi = math.pi
    r_safe = torch.clamp(r, min=1e-8)

    # Highpass H0 and Lowpass L0 for initial stage
    # Transition band between pi/4 and pi/2
    l0 = torch.zeros_like(r)
    mask_low = r <= (pi / 4.0)
    mask_trans = (r > (pi / 4.0)) & (r < (pi / 2.0))
    l0[mask_low] = 1.0
    l0[mask_trans] = torch.cos((pi / 2.0) * torch.log2((4.0 * r_safe[mask_trans]) / pi))

    h0 = torch.zeros_like(r)
    mask_high = r >= (pi / 2.0)
    h0[mask_high] = 1.0
    h0[mask_trans] = torch.cos((pi / 2.0) * torch.log2((2.0 * r_safe[mask_trans]) / pi))

    # Bandpass transition filter B(r) and recursive lowpass L(r)
    # Scaled octave transition between pi/4 and pi/2
    b = torch.zeros_like(r)
    b[mask_trans] = torch.cos((pi / 2.0) * torch.log2((2.0 * r_safe[mask_trans]) / pi))

    return h0, b, l0


def _orientation_filter(theta: torch.Tensor, theta_k: float, num_orientations: int) -> torch.Tensor:
    """Computes angular orientation filter G_k(theta) for k-th orientation.

    Uses cos^(K-1)(theta - theta_k) with exact tight-frame normalization.
    """
    k = num_orientations
    # Alpha normalization factor ensuring sum_k |G_k(theta)|^2 = 1
    # For K=4, alpha_4 = 2 / sqrt(5)
    # In general: alpha_K = 2^(K-1) * (K-1)! / sqrt(K * (2K - 2)!)
    fac_k_minus_1 = math.factorial(k - 1)
    fac_2k_minus_2 = math.factorial(2 * k - 2)
    alpha_k = (2.0 ** (k - 1)) * fac_k_minus_1 / math.sqrt(k * fac_2k_minus_2)

    # Angular difference wrapped in [-pi, pi]
    d_theta = torch.remainder(theta - theta_k + math.pi, 2.0 * math.pi) - math.pi

    # Filter is non-zero only where |d_theta| < pi/2
    g = torch.zeros_like(theta)
    mask = torch.abs(d_theta) < (math.pi / 2.0)
    g[mask] = alpha_k * (torch.cos(d_theta[mask]) ** (k - 1))

    return g


class SteerableWeberPyramid(nn.Module):
    """Multi-scale, multi-orientation steerable Weber contrast pyramid.

    Decomposes an input tensor into spatial frequency bands, each containing
    K orientation subbands (e.g., K=4 for 0°, 45°, 90°, 135°), plus a low-frequency
    baseband, normalized by background luminance to form Weber contrast.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        ppd: Pixels per degree of visual angle.
        num_orientations: Number of orientation subbands (default: 4).
        min_freq: Minimum spatial frequency in cycles per degree (cpd).
    """

    def __init__(
        self,
        width: int,
        height: int,
        ppd: float,
        num_orientations: int = 4,
        min_freq: float = 0.2,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.ppd = ppd
        self.num_orientations = num_orientations
        self.min_freq = min_freq

        # Calculate number of scale levels based on dimensions and min_freq
        max_levels = int(np.floor(np.log2(min(height, width)))) - 1
        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0, 14.0)) * 0.3228], 0) * (self.ppd / 2.0)
        invalid_bands = np.array(np.nonzero(bands <= self.min_freq))

        if invalid_bands.shape[-1] == 0 or invalid_bands.shape[0] == 0 or len(invalid_bands[0]) == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        self.num_levels = int(np.clip(max_band + 1, 1, max_levels))
        self.band_freqs = np.array([1.0] + [0.3228 * (2.0 ** (-f)) for f in range(self.num_levels)]) * (self.ppd / 2.0)

        # Orientation angles (e.g. 0, pi/4, pi/2, 3pi/4 for K=4)
        self.orientations = [k * math.pi / self.num_orientations for k in range(self.num_orientations)]

        # Precompute frequency grids and filter masks
        self._build_filters()

    def _build_filters(self):
        """Constructs and registers frequency-domain filter buffers."""
        r, theta = _get_polar_grid(self.height, self.width, device=torch.device("cpu"))
        h0, b_rad, l0 = _steerable_radial_filters(r)

        self.register_buffer("filter_h0", h0)
        self.register_buffer("filter_l0", l0)

        # Precompute oriented filters for each scale
        # For multiscale FFT pyramid, scale filters scale with octave r_scale = r * (2^scale)
        scale_filters = []
        lowpass_filters = []

        for level in range(self.num_levels):
            # Radial scaling by 2^level
            r_level = r * (2.0**level)
            _, b_level, l_level = _steerable_radial_filters(r_level)

            ori_filters = []
            for ori_idx, theta_k in enumerate(self.orientations):
                g_k = _orientation_filter(theta, theta_k, self.num_orientations)
                # Subband filter = B(2^level * r) * G_k(theta)
                band_k = b_level * g_k
                self.register_buffer(f"filter_band_s{level}_o{ori_idx}", band_k)
                ori_filters.append(band_k)

            self.register_buffer(f"filter_lowpass_s{level}", l_level)
            scale_filters.append(ori_filters)
            lowpass_filters.append(l_level)

    @property
    def band_count(self) -> int:
        """Total number of spatial frequency levels including baseband."""
        return self.num_levels + 1

    def get_freqs(self) -> np.ndarray:
        """Returns spatial center frequencies per band in cycles per degree (cpd)."""
        return self.band_freqs.copy()

    def decompose(
        self, r: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Decomposes interleaved tensor R into Weber contrast oriented subbands.

        Args:
            r: Tensor of shape (B, 2*C, T, H, W) containing interleaved test/ref channels.

        Returns:
            lpyr: List of contrast subband tensors for each scale:
                  - For intermediate scales (i = 0..num_levels-1):
                    Shape (B, 2*C, T, K, H, W) where K is num_orientations.
                  - For baseband (i = num_levels):
                    Shape (B, 2*C, T, 1, H, W) isotropic low-frequency residual.
            log_l_bkg_pyr: List of log10 background luminance tensors for each scale.
        """
        b, c_all, t, h, w = r.shape
        device = r.device

        # Reshape to 2D for batched FFT: (B * C_all * T, H, W)
        x_flat = r.reshape(-1, h, w)
        x_fft = torch.fft.fft2(x_flat)

        lpyr = []
        log_l_bkg_pyr = []

        # 1. Initial Highpass / Lowpass split
        l0_filter = getattr(self, "filter_l0").to(device)
        curr_l_fft = x_fft * l0_filter

        # 2. Decompose into oriented scales
        for level in range(self.num_levels):
            # Compute oriented subbands at this scale
            oriented_bands = []
            for ori_idx in range(self.num_orientations):
                band_filter = getattr(self, f"filter_band_s{level}_o{ori_idx}").to(device)
                subband_fft = curr_l_fft * band_filter
                subband_spatial = torch.fft.ifft2(subband_fft).real.view(b, c_all, t, h, w)
                oriented_bands.append(subband_spatial)

            # Stack along orientation dimension: (B, C_all, T, K, H, W)
            scale_oriented = torch.stack(oriented_bands, dim=3)

            # Next lowpass signal
            l_filter = getattr(self, f"filter_lowpass_s{level}").to(device)
            next_l_fft = curr_l_fft * l_filter
            lowpass_spatial = torch.fft.ifft2(next_l_fft).real.view(b, c_all, t, h, w)

            # Background luminance for Weber normalization
            l_bkg = torch.clamp(lowpass_spatial[..., 0:2, :, :, :], min=0.01)  # Test and Ref luminance
            l_bkg_expanded = l_bkg.unsqueeze(3)  # Shape: (B, 2, T, 1, H, W)

            # Weber contrast computation: subband / l_bkg
            contrast = torch.empty_like(scale_oriented)
            contrast[..., 0::2, :, :, :, :] = torch.clamp(
                scale_oriented[..., 0::2, :, :, :, :] / l_bkg_expanded[..., 0:1, :, :, :, :],
                max=1000.0,
            )
            contrast[..., 1::2, :, :, :, :] = torch.clamp(
                scale_oriented[..., 1::2, :, :, :, :] / l_bkg_expanded[..., 1:2, :, :, :, :],
                max=1000.0,
            )

            # Scale multiplier (2.0 for intermediate bands, consistent with ColorVideoVDP)
            band_mul = 1.0 if level == 0 else 2.0
            contrast = contrast * band_mul

            lpyr.append(contrast)
            log_l_bkg_pyr.append(torch.log10(l_bkg))

            curr_l_fft = next_l_fft

        # 3. Baseband (isotropic residual)
        baseband_spatial = torch.fft.ifft2(curr_l_fft).real.view(b, c_all, t, h, w)
        l_bkg_base = torch.clamp(baseband_spatial[..., 0:2, :, :, :], min=0.01)
        l_bkg_base_mean = torch.mean(l_bkg_base, dim=[-1, -2], keepdim=True)

        contrast_base = torch.empty_like(baseband_spatial)
        contrast_base[..., 0::2, :, :, :] = torch.clamp(
            baseband_spatial[..., 0::2, :, :, :] / l_bkg_base_mean[..., 0:1, :, :, :],
            max=1000.0,
        )
        contrast_base[..., 1::2, :, :, :] = torch.clamp(
            baseband_spatial[..., 1::2, :, :, :] / l_bkg_base_mean[..., 1:2, :, :, :],
            max=1000.0,
        )

        # Baseband has K=1 orientation (isotropic) -> shape (B, C_all, T, 1, H, W)
        contrast_base = contrast_base.unsqueeze(3)
        lpyr.append(contrast_base)
        log_l_bkg_pyr.append(torch.log10(l_bkg_base_mean))

        return lpyr, log_l_bkg_pyr
