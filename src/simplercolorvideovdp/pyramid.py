"""Weber Contrast Laplacian Pyramid Decomposition for SimplerColorVideoVDP."""

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
