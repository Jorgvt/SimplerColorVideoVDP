"""Weber Contrast Laplacian and Steerable Pyramids for SimplerColorVideoVDP."""

import math
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _interleave_zeros_and_pad(x: torch.Tensor, exp_size: Tuple[int, int], dim: int) -> torch.Tensor:
    """Interleaves zeros for expand operation and applies symmetric padding."""
    new_shape = list(x.shape)
    new_shape[dim] = exp_size[dim] + 4
    z = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    odd_no = exp_size[dim] % 2

    if dim == -2:
        z[..., 2:-2:2, :] = x
        z[..., 0, :] = x[..., 0, :]
        z[..., -2 + odd_no, :] = x[..., -1, :]
    elif dim == -1:
        z[..., :, 2:-2:2] = x
        z[..., :, 0] = x[..., :, 0]
        z[..., :, -2 + odd_no] = x[..., :, -1]
    else:
        raise ValueError("Invalid dimension for padding")

    return z


class WeberLaplacianPyramid(nn.Module):
    """Computes Weber contrast Laplacian pyramid decomposition for interleaved test/reference channels."""

    def __init__(self, width: int, height: int, ppd: float, kernel_a: float = 0.4, min_freq: float = 0.2):
        super().__init__()
        self.width = width
        self.height = height
        self.ppd = ppd
        self.kernel_a = kernel_a
        self.min_freq = min_freq

        max_levels = int(np.floor(np.log2(min(height, width)))) - 1
        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0, 14.0)) * 0.3228], 0) * (self.ppd / 2.0)
        invalid_bands = np.array(np.nonzero(bands <= self.min_freq))

        if invalid_bands.shape[-1] == 0 or invalid_bands.shape[0] == 0 or len(invalid_bands[0]) == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        self.num_levels = int(np.clip(max_band + 1, 0, max_levels))
        self.band_freqs = np.array([1.0] + [0.3228 * (2.0 ** (-f)) for f in range(self.num_levels)]) * (self.ppd / 2.0)

        # 5-tap filter kernels
        k = torch.tensor(
            [0.25 - kernel_a / 2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a / 2.0],
            dtype=torch.float32,
        )
        self.register_buffer("k_vert", k.view(1, 1, 5, 1))
        self.register_buffer("k_horiz", k.view(1, 1, 1, 5))

    @property
    def band_count(self) -> int:
        return self.num_levels + 1

    def get_freqs(self) -> np.ndarray:
        """Returns spatial frequencies per band in cpd."""
        freqs = self.band_freqs.copy()
        return freqs

    def gausspyr_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 2D separable Gaussian reduction (subsampling by 2)."""
        h, w = x.shape[-2], x.shape[-1]
        x_flat = x.reshape(-1, 1, h, w)

        # Vertical reduction
        y_a = F.conv2d(x_flat, self.k_vert, stride=(2, 1), padding=(2, 0)).view(x.shape[:-2] + (-1, w))
        y_a[..., 0, :] += x[..., 0, :] * self.k_vert[0, 0, 1, 0] + x[..., 1, :] * self.k_vert[0, 0, 0, 0]
        if x.shape[-2] % 2 == 1:
            y_a[..., -1, :] += x[..., -1, :] * self.k_vert[0, 0, 3, 0] + x[..., -2, :] * self.k_vert[0, 0, 4, 0]
        else:
            y_a[..., -1, :] += x[..., -1, :] * self.k_vert[0, 0, 4, 0]

        # Horizontal reduction
        h_mid = y_a.shape[-2]
        y_a_flat = y_a.reshape(-1, 1, h_mid, w)
        y = F.conv2d(y_a_flat, self.k_horiz, stride=(1, 2), padding=(0, 2)).view(x.shape[:-2] + (h_mid, -1))
        y[..., :, 0] += y_a[..., :, 0] * self.k_horiz[0, 0, 0, 1] + y_a[..., :, 1] * self.k_horiz[0, 0, 0, 0]
        if x.shape[-1] % 2 == 1:
            y[..., :, -1] += y_a[..., :, -1] * self.k_horiz[0, 0, 0, 3] + y_a[..., :, -2] * self.k_horiz[0, 0, 0, 4]
        else:
            y[..., :, -1] += y_a[..., :, -1] * self.k_horiz[0, 0, 0, 4]

        return y

    def gausspyr_expand(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Expands tensor to target_size with 2D separable convolution."""
        y_a = _interleave_zeros_and_pad(x, exp_size=target_size, dim=-2)
        h, w = y_a.shape[-2], y_a.shape[-1]
        y_a = F.conv2d(y_a.reshape(-1, 1, h, w), self.k_vert * 2.0).view(x.shape[:-2] + (-1, w))

        y = _interleave_zeros_and_pad(y_a, exp_size=target_size, dim=-1)
        h, w = y.shape[-2], y.shape[-1]
        y = F.conv2d(y.reshape(-1, 1, h, w), self.k_horiz * 2.0).view(x.shape[:-2] + (target_size[0], target_size[1]))

        return y

    def decompose(self, r: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Decomposes interleaved tensor R (B, 2*C, T, H, W) into Weber contrast pyramid.

        Returns:
            lpyr: List of contrast tensors for each band (scaled by band multiplier).
            log_l_bkg_pyr: List of log10 background luminance tensors for each band.
        """
        levels = self.num_levels + 1

        # 1. Build Gaussian pyramid
        gpyr = [r]
        for _ in range(1, levels):
            gpyr.append(self.gausspyr_reduce(gpyr[-1]))

        lpyr = []
        log_l_bkg_pyr = []

        for i in range(levels):
            is_baseband = i == (levels - 1)

            if is_baseband:
                layer = gpyr[i]
                l_bkg = torch.clamp(gpyr[i][..., 0:2, :, :, :], min=0.01)
                l_bkg = torch.mean(l_bkg, dim=[-1, -2], keepdim=True)
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i + 1], (gpyr[i].shape[-2], gpyr[i].shape[-1]))
                layer = gpyr[i] - glayer_ex
                l_bkg = torch.clamp(glayer_ex[..., 0:2, :, :, :], min=0.01)

            contrast = torch.empty_like(layer)
            contrast[..., 0::2, :, :, :] = torch.clamp(layer[..., 0::2, :, :, :] / l_bkg[..., 0:1, :, :, :], max=1000.0)
            contrast[..., 1::2, :, :, :] = torch.clamp(layer[..., 1::2, :, :, :] / l_bkg[..., 1:2, :, :, :], max=1000.0)

            # Apply band multiplier (2.0 for intermediate bands, 1.0 for band 0 and baseband)
            band_mul = 1.0 if (i == 0 or is_baseband) else 2.0
            contrast = contrast * band_mul

            lpyr.append(contrast)
            log_l_bkg_pyr.append(torch.log10(l_bkg))

        return lpyr, log_l_bkg_pyr


# =========================================================================
# Steerable Multi-Orientation Weber Pyramid
# =========================================================================

def _get_polar_grid(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates normalized radial frequency (r in [0, pi]) and angular (theta in [-pi, pi]) grids for rfft2."""
    y_freq = torch.fft.fftfreq(h, d=1.0, device=device) * (2.0 * math.pi)
    x_freq = torch.fft.rfftfreq(w, d=1.0, device=device) * (2.0 * math.pi)

    y_grid, x_grid = torch.meshgrid(y_freq, x_freq, indexing="ij")
    r = torch.sqrt(x_grid**2 + y_grid**2)
    theta = torch.atan2(y_grid, x_grid)

    return r, theta


def _steerable_radial_filters(r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes highpass, bandpass, and lowpass radial filter profiles."""
    pi = math.pi
    r_safe = torch.clamp(r, min=1e-8)

    l0 = torch.zeros_like(r)
    mask_low = r <= (pi / 4.0)
    mask_trans = (r > (pi / 4.0)) & (r < (pi / 2.0))
    l0[mask_low] = 1.0
    l0[mask_trans] = torch.cos((pi / 2.0) * torch.log2((4.0 * r_safe[mask_trans]) / pi))

    h0 = torch.zeros_like(r)
    mask_high = r >= (pi / 2.0)
    h0[mask_high] = 1.0
    h0[mask_trans] = torch.cos((pi / 2.0) * torch.log2((2.0 * r_safe[mask_trans]) / pi))

    b = torch.zeros_like(r)
    b[mask_trans] = torch.cos((pi / 2.0) * torch.log2((2.0 * r_safe[mask_trans]) / pi))

    return h0, b, l0


def _orientation_filter(theta: torch.Tensor, theta_k: float, num_orientations: int) -> torch.Tensor:
    """Computes angular orientation filter G_k(theta) with tight-frame normalization."""
    k = num_orientations
    fac_k_minus_1 = math.factorial(k - 1)
    fac_2k_minus_2 = math.factorial(2 * k - 2)
    alpha_k = (2.0 ** (k - 1)) * fac_k_minus_1 / math.sqrt(k * fac_2k_minus_2)

    d_theta = torch.remainder(theta - theta_k + math.pi, 2.0 * math.pi) - math.pi
    g = torch.zeros_like(theta)
    mask = torch.abs(d_theta) < (math.pi / 2.0)
    g[mask] = alpha_k * (torch.cos(d_theta[mask]) ** (k - 1))

    return g


class SteerableWeberPyramid(nn.Module):
    """Multi-scale, multi-orientation steerable Weber contrast pyramid with octave downsampling.

    Decomposes an input tensor into spatial frequency bands, each downsampled by 2x
    at each octave, containing K orientation subbands (e.g., K=4 for 0°, 45°, 90°, 135°),
    plus a low-frequency baseband, normalized by background luminance to form Weber contrast.
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

        max_levels = int(np.floor(np.log2(min(height, width)))) - 1
        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0, 14.0)) * 0.3228], 0) * (self.ppd / 2.0)
        invalid_bands = np.array(np.nonzero(bands <= self.min_freq))

        if invalid_bands.shape[-1] == 0 or invalid_bands.shape[0] == 0 or len(invalid_bands[0]) == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        self.num_levels = int(np.clip(max_band + 1, 1, max_levels))
        self.band_freqs = np.array([1.0] + [0.3228 * (2.0 ** (-f)) for f in range(self.num_levels)]) * (self.ppd / 2.0)
        self.orientations = [k * math.pi / self.num_orientations for k in range(self.num_orientations)]

        # Spatial sizes for all levels
        self.level_sizes = []
        cur_h, cur_w = height, width
        for _ in range(self.num_levels):
            self.level_sizes.append((cur_h, cur_w))
            cur_h = max(1, cur_h // 2)
            cur_w = max(1, cur_w // 2)
        self.base_size = (cur_h, cur_w)

        self._build_filters()

    def _build_filters(self):
        """Constructs and registers frequency-domain filter buffers for each scale."""
        for level, (h_l, w_l) in enumerate(self.level_sizes):
            r, theta = _get_polar_grid(h_l, w_l, device=torch.device("cpu"))
            h0, b_rad, l0 = _steerable_radial_filters(r)

            if level == 0:
                self.register_buffer("filter_h0", h0)
                self.register_buffer("filter_l0", l0)

            for ori_idx, theta_k in enumerate(self.orientations):
                g_k = _orientation_filter(theta, theta_k, self.num_orientations)
                band_k = b_rad * g_k
                self.register_buffer(f"filter_band_s{level}_o{ori_idx}", band_k)

            self.register_buffer(f"filter_lowpass_s{level}", l0)

    @property
    def band_count(self) -> int:
        return self.num_levels + 1

    def get_freqs(self) -> np.ndarray:
        return self.band_freqs.copy()

    def decompose(
        self, r: torch.Tensor, slice_size: int = 15
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Decomposes interleaved tensor R into Weber contrast oriented subbands with octave downsampling.

        Args:
            r: Tensor of shape (B, 2*C, T, H, W) containing interleaved test/ref channels.
            slice_size: Number of time frames to process per spatial FFT batch to minimize VRAM.

        Returns:
            lpyr: List of contrast subband tensors:
                  - Intermediate scales (i = 0..num_levels-1): Shape (B, 2*C, T, K, H_i, W_i)
                  - Baseband (i = num_levels): Shape (B, 2*C, T, 1, H_base, W_base)
            log_l_bkg_pyr: List of log10 background luminance tensors.
        """
        b, c_all, t, h, w = r.shape
        device = r.device

        lpyr = []
        log_l_bkg_pyr = []

        curr_spatial = r

        for level in range(self.num_levels):
            h_l, w_l = self.level_sizes[level]

            l_filter = getattr(self, f"filter_lowpass_s{level}").to(device)
            if level == 0:
                l0_filter = getattr(self, "filter_l0").to(device)

            contrast = torch.empty((b, c_all, t, self.num_orientations, h_l, w_l), dtype=r.dtype, device=device)
            lowpass_spatial = torch.empty((b, c_all, t, h_l, w_l), dtype=r.dtype, device=device)

            # Process in temporal slices to keep peak VRAM bounded
            eff_slice = slice_size if (t > slice_size and h_l * w_l >= 128 * 128) else t
            for t_start in range(0, t, eff_slice):
                t_end = min(t_start + eff_slice, t)
                sub_spatial = curr_spatial[:, :, t_start:t_end, :, :]
                curr_flat = sub_spatial.reshape(-1, h_l, w_l)
                curr_fft = torch.fft.rfft2(curr_flat)

                if level == 0:
                    curr_fft = curr_fft * l0_filter

                next_l_fft = curr_fft * l_filter
                lp_sp = torch.fft.irfft2(next_l_fft, s=(h_l, w_l)).view(b, c_all, t_end - t_start, h_l, w_l)
                lowpass_spatial[:, :, t_start:t_end, :, :] = lp_sp

                l_bkg = torch.clamp(lp_sp[..., 0:2, :, :, :], min=0.01)
                l_bkg_0 = l_bkg[..., 0:1, :, :, :]
                l_bkg_1 = l_bkg[..., 1:2, :, :, :]

                for ori_idx in range(self.num_orientations):
                    band_filter = getattr(self, f"filter_band_s{level}_o{ori_idx}").to(device)
                    subband_fft = curr_fft * band_filter
                    subband_spatial = torch.fft.irfft2(subband_fft, s=(h_l, w_l)).view(b, c_all, t_end - t_start, h_l, w_l)

                    contrast[:, 0::2, t_start:t_end, ori_idx, :, :] = torch.clamp(
                        subband_spatial[:, 0::2] / l_bkg_0, min=-1000.0, max=1000.0
                    )
                    contrast[:, 1::2, t_start:t_end, ori_idx, :, :] = torch.clamp(
                        subband_spatial[:, 1::2] / l_bkg_1, min=-1000.0, max=1000.0
                    )

            band_mul = 1.0 if level == 0 else 2.0
            contrast.mul_(band_mul)

            lpyr.append(contrast)
            log_l_bkg_pyr.append(torch.log10(torch.clamp(lowpass_spatial[..., 0:2, :, :, :], min=0.01)))

            # Downsample lowpass for next level
            next_h, next_w = (
                self.level_sizes[level + 1]
                if level < self.num_levels - 1
                else self.base_size
            )
            curr_flat_lp = lowpass_spatial.reshape(-1, 1, h_l, w_l)
            downsampled = F.interpolate(curr_flat_lp, size=(next_h, next_w), mode="area")
            curr_spatial = downsampled.view(b, c_all, t, next_h, next_w)

        # Baseband (isotropic residual)
        h_base, w_base = self.base_size
        baseband_spatial = curr_spatial
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

        contrast_base = contrast_base.unsqueeze(3)
        lpyr.append(contrast_base)
        log_l_bkg_pyr.append(torch.log10(l_bkg_base_mean))

        return lpyr, log_l_bkg_pyr
