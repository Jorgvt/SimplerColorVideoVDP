"""Canonical Direct-Stream V1-MT Cortical Motion & Form Model.

Ported from canonical 6-channel V1-MT architecture with:
- 3D spatio-temporal quadrature Gabor filter bank in V1 (8 directions, sf=1.0 cpd, tf=4.0 Hz)
- Divisive cross-orientation and spatial normalization in V1
- MT velocity plane integration for 3 speeds (1.0, 4.0, 16.0 deg/s) and 8 directions
- MT half-squaring and population divisive normalization
- Sustained 2D spatial Gabor Form pathway on DKL color channels (Luminance, RG, BY)
- Divisive form normalization and MAE (L1) feature distance
- High-efficiency temporal sub-chunk streaming for high-resolution video processing
- Direct linear hybrid fusion with ColorVideoVDP / isotropic backbone
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Color Conversion (RGB -> DKL Opponent Space)
# ==============================================================================

def rgb_to_dkl_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Converts normalized RGB video tensor to DKL opponent color space.

    Args:
        rgb: Tensor of shape (..., 3) with values in [0, 1].

    Returns:
        Tensor of shape (..., 3) with [Luminance (L+M), Red-Green (L-M), Blue-Yellow (S-(L+M))].
    """
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    rg = 0.7071 * rgb[..., 0] - 0.7071 * rgb[..., 1]
    by = -0.4082 * rgb[..., 0] - 0.4082 * rgb[..., 1] + 0.8165 * rgb[..., 2]
    return torch.stack([lum, rg, by], dim=-1)


def pad_symmetric_5d(x: torch.Tensor, pad_t: Tuple[int, int], pad_h: Tuple[int, int], pad_w: Tuple[int, int]) -> torch.Tensor:
    """Applies exact symmetric padding (mirroring with edge duplication) to a 5D tensor (B, C, T, H, W)."""
    pt_l, pt_r = pad_t
    ph_l, ph_r = pad_h
    pw_l, pw_r = pad_w

    if pw_l > 0 or pw_r > 0:
        w_left = torch.flip(x[..., :pw_l], dims=[-1]) if pw_l > 0 else x[..., :0]
        w_right = torch.flip(x[..., -pw_r:], dims=[-1]) if pw_r > 0 else x[..., :0]
        x = torch.cat([w_left, x, w_right], dim=-1)

    if ph_l > 0 or ph_r > 0:
        h_left = torch.flip(x[..., :ph_l, :], dims=[-2]) if ph_l > 0 else x[..., :0, :]
        h_right = torch.flip(x[..., -ph_r:, :], dims=[-2]) if ph_r > 0 else x[..., :0, :]
        x = torch.cat([h_left, x, h_right], dim=-2)

    if pt_l > 0 or pt_r > 0:
        t_left = torch.flip(x[..., :pt_l, :, :], dims=[-3]) if pt_l > 0 else x[..., :0, :, :]
        t_right = torch.flip(x[..., -pt_r:, :, :], dims=[-3]) if pt_r > 0 else x[..., :0, :, :]
        x = torch.cat([t_left, x, t_right], dim=-3)

    return x


# ==============================================================================
# 2. 3D Gabor Kernel Synthesis (Motion Pathway)
# ==============================================================================

def synthesize_3d_gabor_kernels_torch(
    theta: torch.Tensor,
    sf: torch.Tensor,
    tf: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_t: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    pixels_per_degree: float,
    fps: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generates 3D Gabor filter bank kernels in Cartesian coordinates using PyTorch."""
    num_directions = theta.shape[0]
    kt, kh, kw = kernel_size
    dev = device if device is not None else theta.device

    ts = torch.linspace(-kt / 2.0, kt / 2.0, kt, device=dev, dtype=dtype) / fps
    ys = torch.linspace(-kh / 2.0, kh / 2.0, kh, device=dev, dtype=dtype) / pixels_per_degree
    xs = torch.linspace(-kw / 2.0, kw / 2.0, kw, device=dev, dtype=dtype) / pixels_per_degree

    T, Y, X = torch.meshgrid(ts, ys, xs, indexing='ij')
    Y_cartesian = -Y

    X_exp = X.unsqueeze(0).expand(num_directions, -1, -1, -1)
    Y_exp = Y_cartesian.unsqueeze(0).expand(num_directions, -1, -1, -1)
    T_exp = T.unsqueeze(0).expand(num_directions, -1, -1, -1)

    theta_4d = theta.view(num_directions, 1, 1, 1)
    sf_4d = sf.view(-1, 1, 1, 1)
    tf_4d = tf.view(-1, 1, 1, 1)
    sx_4d = sigma_x.view(-1, 1, 1, 1)
    sy_4d = sigma_y.view(-1, 1, 1, 1)
    st_4d = sigma_t.view(-1, 1, 1, 1)

    cos_t = torch.cos(theta_4d)
    sin_t = torch.sin(theta_4d)

    x_prime = X_exp * cos_t + Y_exp * sin_t
    y_prime = -X_exp * sin_t + Y_exp * cos_t

    spatial_envelope = torch.exp(-0.5 * ((x_prime / sx_4d)**2 + (y_prime / sy_4d)**2))
    temporal_envelope = torch.exp(-0.5 * (T_exp / st_4d)**2)
    envelope = spatial_envelope * temporal_envelope

    carrier_arg = 2.0 * math.pi * (sf_4d * x_prime + tf_4d * T_exp)
    even_raw = envelope * torch.cos(carrier_arg)
    odd_raw = envelope * torch.sin(carrier_arg)

    even_norm = even_raw - torch.mean(even_raw, dim=(1, 2, 3), keepdim=True)
    odd_norm = odd_raw - torch.mean(odd_raw, dim=(1, 2, 3), keepdim=True)

    even_std = torch.std(even_raw, dim=(1, 2, 3), keepdim=True) + 1e-8
    odd_std = torch.std(odd_raw, dim=(1, 2, 3), keepdim=True) + 1e-8

    even_final = even_norm / even_std
    odd_final = odd_norm / odd_std

    kernels = torch.empty((2 * num_directions, 1, kt, kh, kw), device=dev, dtype=dtype)
    kernels[0::2, 0] = even_final
    kernels[1::2, 0] = odd_final

    return kernels


# ==============================================================================
# 3. 2D Spatial Gabor Kernel Synthesis (Form Pathway)
# ==============================================================================

def synthesize_2d_gabor_kernels_torch(
    theta: torch.Tensor,
    sf: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    kernel_size: Tuple[int, int],
    pixels_per_degree: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generates 2D static Gabor filter kernels for the Sustained Form Pathway."""
    num_directions = theta.shape[0]
    kh, kw = kernel_size
    dev = device if device is not None else theta.device

    ys = torch.linspace(-kh / 2.0, kh / 2.0, kh, device=dev, dtype=dtype) / pixels_per_degree
    xs = torch.linspace(-kw / 2.0, kw / 2.0, kw, device=dev, dtype=dtype) / pixels_per_degree

    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Y_cartesian = -Y

    X_exp = X.unsqueeze(0).expand(num_directions, -1, -1)
    Y_exp = Y_cartesian.unsqueeze(0).expand(num_directions, -1, -1)

    theta_3d = theta.view(num_directions, 1, 1)
    sf_3d = sf.view(-1, 1, 1)
    sx_3d = sigma_x.view(-1, 1, 1)
    sy_3d = sigma_y.view(-1, 1, 1)

    cos_t = torch.cos(theta_3d)
    sin_t = torch.sin(theta_3d)

    x_prime = X_exp * cos_t + Y_exp * sin_t
    y_prime = -X_exp * sin_t + Y_exp * cos_t

    envelope = torch.exp(-0.5 * ((x_prime / sx_3d)**2 + (y_prime / sy_3d)**2))
    carrier_arg = 2.0 * math.pi * sf_3d * x_prime

    even_raw = envelope * torch.cos(carrier_arg)
    odd_raw = envelope * torch.sin(carrier_arg)

    even_norm = even_raw - torch.mean(even_raw, dim=(1, 2), keepdim=True)
    odd_norm = odd_raw - torch.mean(odd_raw, dim=(1, 2), keepdim=True)

    even_std = torch.std(even_raw, dim=(1, 2), keepdim=True) + 1e-8
    odd_std = torch.std(odd_raw, dim=(1, 2), keepdim=True) + 1e-8

    even_final = even_norm / even_std
    odd_final = odd_norm / odd_std

    kernels = torch.empty((2 * num_directions, 1, kh, kw), device=dev, dtype=dtype)
    kernels[0::2, 0] = even_final
    kernels[1::2, 0] = odd_final

    return kernels


def compute_velocity_plane_weights_torch(
    theta_v1: torch.Tensor,
    sf_v1: torch.Tensor,
    tf_v1: torch.Tensor,
    theta_mt: torch.Tensor,
    speed_mt: torch.Tensor,
    sigma_p: float = 0.5
) -> torch.Tensor:
    """Computes Gaussian velocity plane integration weights W (V1 -> MT)."""
    th_v1 = theta_v1.unsqueeze(0)
    sf = sf_v1.unsqueeze(0)
    tf = tf_v1.unsqueeze(0)

    th_mt = theta_mt.unsqueeze(1)
    spd_mt = speed_mt.unsqueeze(1)

    predicted_tf = sf * spd_mt * torch.cos(th_v1 - th_mt)
    diff = tf - predicted_tf
    weights = torch.exp(-0.5 * (diff / sigma_p)**2)

    denom = torch.sum(weights, dim=1, keepdim=True) + 1e-8
    return weights / denom


# ==============================================================================
# 4. Neural Layers
# ==============================================================================

class V1FilterLayerTorch(nn.Module):
    """V1 3D Spatio-Temporal Quadrature Filter Bank."""

    def __init__(
        self,
        directions: Sequence[float] = (0.0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4),
        sf: float = 1.0,
        tf: float = 4.0,
        sigma_x: float = 0.2,
        sigma_y: float = 0.2,
        sigma_t: float = 0.05,
        kernel_size: Tuple[int, int, int] = (15, 15, 15),
        pixels_per_degree: float = 30.0,
        fps: float = 240.0,
        freeze_gabor: bool = True
    ):
        super().__init__()
        self.directions = list(directions)
        self.kernel_size = kernel_size
        self.pixels_per_degree = pixels_per_degree
        self.fps = fps
        self.freeze_gabor = freeze_gabor

        theta_arr = torch.tensor(self.directions, dtype=torch.float32)
        sf_arr = torch.tensor([sf], dtype=torch.float32)
        tf_arr = torch.tensor([tf], dtype=torch.float32)
        sigma_x_arr = torch.tensor([sigma_x], dtype=torch.float32)
        sigma_y_arr = torch.tensor([sigma_y], dtype=torch.float32)
        sigma_t_arr = torch.tensor([sigma_t], dtype=torch.float32)

        if freeze_gabor:
            self.register_buffer('theta', theta_arr)
            self.register_buffer('sf', sf_arr)
            self.register_buffer('tf', tf_arr)
            self.register_buffer('sigma_x', sigma_x_arr)
            self.register_buffer('sigma_y', sigma_y_arr)
            self.register_buffer('sigma_t', sigma_t_arr)
        else:
            self.theta = nn.Parameter(theta_arr)
            self.sf = nn.Parameter(sf_arr)
            self.tf = nn.Parameter(tf_arr)
            self.sigma_x = nn.Parameter(sigma_x_arr)
            self.sigma_y = nn.Parameter(sigma_y_arr)
            self.sigma_t = nn.Parameter(sigma_t_arr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, 1, T, H, W). Returns energy of shape (B, num_directions, T, H, W)."""
        kernels = synthesize_3d_gabor_kernels_torch(
            self.theta, self.sf, self.tf, self.sigma_x, self.sigma_y, self.sigma_t,
            self.kernel_size, self.pixels_per_degree, self.fps, device=x.device, dtype=x.dtype
        )

        kt, kh, kw = self.kernel_size
        pt = (kt - 1) // 2
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

        padded_x = pad_symmetric_5d(x, (pt, pt), (ph, ph), (pw, pw))
        conv_out = F.conv3d(padded_x, kernels, stride=1, padding=0)

        even = conv_out[:, 0::2, :, :, :]
        odd = conv_out[:, 1::2, :, :, :]
        energy = even**2 + odd**2
        return energy


class V1NormalizationTorch(nn.Module):
    """V1 divisive cross-orientation and spatial normalization."""

    def __init__(self, spatial_pool_size: Tuple[int, int] = (5, 5), trainable_sigma: bool = True):
        super().__init__()
        self.spatial_pool_size = spatial_pool_size
        self.sigma = nn.Parameter(torch.tensor([0.1], dtype=torch.float32), requires_grad=trainable_sigma)

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        cross_channel_pool = torch.mean(energy, dim=1, keepdim=True)
        kh, kw = self.spatial_pool_size
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

        padded_pool = pad_symmetric_5d(cross_channel_pool, (0, 0), (ph, ph), (pw, pw))
        pool_denominator = F.avg_pool3d(padded_pool, kernel_size=(1, kh, kw), stride=1, padding=0)
        return energy / (self.sigma**2 + pool_denominator)


class MTIntegrationTorch(nn.Module):
    """Middle Temporal (MT) linear integration layer across 2D velocity space."""

    def __init__(
        self,
        v1_filter: V1FilterLayerTorch,
        mt_directions: Sequence[float],
        mt_speeds: Sequence[float] = (1.0, 4.0, 16.0),
        sigma_p: float = 0.5,
        spatial_pool_size: Tuple[int, int] = (15, 15),
        trainable_speeds: bool = False
    ):
        super().__init__()
        self.v1_filter = v1_filter
        self.mt_directions = list(mt_directions)
        self.mt_speeds = list(mt_speeds)
        self.sigma_p = sigma_p
        self.spatial_pool_size = spatial_pool_size
        self.trainable_speeds = trainable_speeds

        speeds_grid, dirs_grid = [], []
        for s in self.mt_speeds:
            for d in self.mt_directions:
                speeds_grid.append(float(s))
                dirs_grid.append(float(d))
        theta_mt = torch.tensor(dirs_grid, dtype=torch.float32)
        speed_mt = torch.tensor(speeds_grid, dtype=torch.float32)

        self.register_buffer('theta_mt', theta_mt)
        if trainable_speeds:
            self.speed_mt = nn.Parameter(speed_mt)
        else:
            self.register_buffer('speed_mt', speed_mt)

    def forward(self, v1_norm: torch.Tensor) -> torch.Tensor:
        kh, kw = self.spatial_pool_size
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

        padded_v1 = pad_symmetric_5d(v1_norm, (0, 0), (ph, ph), (pw, pw))
        pooled_v1 = F.avg_pool3d(padded_v1, kernel_size=(1, kh, kw), stride=1, padding=0)

        weights = compute_velocity_plane_weights_torch(
            self.v1_filter.theta, self.v1_filter.sf, self.v1_filter.tf,
            self.theta_mt, self.speed_mt, self.sigma_p
        )
        w_conv = weights.t().reshape(-1, pooled_v1.shape[1], 1, 1, 1)
        return F.conv3d(pooled_v1, w_conv)


class MTNormalizationTorch(nn.Module):
    """MT half-squaring and population divisive normalization."""

    def __init__(self, trainable_sigma: bool = True):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor([0.1], dtype=torch.float32), requires_grad=trainable_sigma)

    def forward(self, mt_linear: torch.Tensor) -> torch.Tensor:
        rectified = torch.square(F.relu(mt_linear))
        pool_denominator = torch.mean(rectified, dim=1, keepdim=True)
        return rectified / (self.sigma**2 + pool_denominator)


class SustainedFormModelTorch(nn.Module):
    """2D Spatial Gabor form pathway module on DKL opponent color channels."""

    def __init__(
        self,
        directions: Sequence[float] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
        sf: float = 1.5,
        sigma_x: float = 0.133,
        sigma_y: float = 0.133,
        kernel_size: Tuple[int, int] = (9, 9),
        spatial_pool_size: Tuple[int, int] = (5, 5),
        pixels_per_degree: float = 30.0,
        freeze_gabor: bool = True
    ):
        super().__init__()
        self.directions = list(directions)
        self.kernel_size = kernel_size
        self.spatial_pool_size = spatial_pool_size
        self.pixels_per_degree = pixels_per_degree
        self.freeze_gabor = freeze_gabor

        theta_arr = torch.tensor(self.directions, dtype=torch.float32)
        if freeze_gabor:
            self.register_buffer('theta', theta_arr)
            self.register_buffer('sf', torch.tensor([sf], dtype=torch.float32))
            self.register_buffer('sigma_x', torch.tensor([sigma_x], dtype=torch.float32))
            self.register_buffer('sigma_y', torch.tensor([sigma_y], dtype=torch.float32))
        else:
            self.theta = nn.Parameter(theta_arr)
            self.sf = nn.Parameter(torch.tensor([sf], dtype=torch.float32))
            self.sigma_x = nn.Parameter(torch.tensor([sigma_x], dtype=torch.float32))
            self.sigma_y = nn.Parameter(torch.tensor([sigma_y], dtype=torch.float32))

        self.sigma_norm = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, dkl_video: torch.Tensor, batch_t: int = 20) -> torch.Tensor:
        """Args: dkl_video of shape (B, T, H, W, 3). Returns: (B, 3, 4, T, H, W)."""
        B, T, H, W, C = dkl_video.shape
        kernels = synthesize_2d_gabor_kernels_torch(
            self.theta, self.sf, self.sigma_x, self.sigma_y,
            self.kernel_size, self.pixels_per_degree, device=dkl_video.device, dtype=dkl_video.dtype
        )

        kh, kw = self.kernel_size
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
        num_dirs = len(self.directions)

        # Process in temporal sub-slices to strictly bound VRAM peak
        norm_energies = []
        for t_idx in range(0, T, batch_t):
            t_slice = dkl_video[:, t_idx:min(t_idx + batch_t, T)]  # (B, T_sub, H, W, C)
            b_sub, t_sub, _, _, c_sub = t_slice.shape

            flat = t_slice.permute(0, 1, 4, 2, 3).reshape(b_sub * t_sub * c_sub, 1, H, W)
            padded = F.pad(flat, (pw, pw, ph, ph), mode='reflect')
            conv_out = F.conv2d(padded, kernels, stride=1, padding=0)

            even = conv_out[:, 0::2, :, :]
            odd = conv_out[:, 1::2, :, :]
            energy = even**2 + odd**2

            cross_pool = torch.mean(energy, dim=1, keepdim=True)
            padded_pool = F.pad(cross_pool, (pw, pw, ph, ph), mode='reflect')
            denom = F.avg_pool2d(padded_pool, kernel_size=(kh, kw), stride=1, padding=0)

            norm_sub = energy / (self.sigma_norm**2 + denom)
            norm_energies.append(norm_sub.reshape(b_sub, t_sub, c_sub, num_dirs, H, W))

        all_norm = torch.cat(norm_energies, dim=1)  # (B, T, C, num_dirs, H, W)
        return all_norm.permute(0, 2, 3, 1, 4, 5)   # (B, C=3, num_dirs=4, T, H, W)


# ==============================================================================
# 5. Full Canonical Direct-Stream V1-MT Model
# ==============================================================================

class CanonicalV1MTModel(nn.Module):
    """Canonical 6-channel V1-MT cortical motion & sustained form model."""

    def __init__(
        self,
        fps: float = 240.0,
        pixels_per_degree: float = 30.0,
        initial_speeds: Tuple[float, float, float] = (1.0, 4.0, 16.0),
        v1_kernel_size: Tuple[int, int, int] = (15, 15, 15),
        trainable_v1: bool = False,
        trainable_mt_speeds: bool = False,
        trainable_form: bool = False
    ):
        super().__init__()
        self.fps = fps
        self.pixels_per_degree = pixels_per_degree
        self.v1_kernel_size = v1_kernel_size

        v1_dirs = [i * (2.0 * math.pi / 8.0) for i in range(8)]
        self.v1_filter = V1FilterLayerTorch(
            directions=v1_dirs,
            sf=1.0,
            tf=4.0,
            kernel_size=v1_kernel_size,
            pixels_per_degree=pixels_per_degree,
            fps=fps,
            freeze_gabor=not trainable_v1
        )
        self.v1_norm = V1NormalizationTorch(spatial_pool_size=(5, 5), trainable_sigma=True)

        mt_dirs = [i * (2.0 * math.pi / 8.0) for i in range(8)]
        self.mt_integ = MTIntegrationTorch(
            v1_filter=self.v1_filter,
            mt_directions=mt_dirs,
            mt_speeds=initial_speeds,
            sigma_p=0.5,
            spatial_pool_size=(15, 15),
            trainable_speeds=trainable_mt_speeds
        )
        self.mt_norm = MTNormalizationTorch(trainable_sigma=True)

        form_dirs = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        self.form_pathway = SustainedFormModelTorch(
            directions=form_dirs,
            sf=1.5,
            kernel_size=(9, 9),
            spatial_pool_size=(5, 5),
            pixels_per_degree=pixels_per_degree,
            freeze_gabor=not trainable_form
        )

        self.w_slow = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.w_med = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.w_fast = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.w_achrom = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.w_rg = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.w_by = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward_features(self, rgb_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes motion and form representations.
        
        Args:
            rgb_video: Tensor of shape (B, T, H, W, 3) in [0, 1].
            
        Returns:
            mt_out: (B, 24, T, H, W)
            form_out: (B, 3, 4, T, H, W)
        """
        dkl = rgb_to_dkl_torch(rgb_video)
        lum = dkl[..., 0:1].permute(0, 4, 1, 2, 3)  # (B, 1, T, H, W)

        v1_energy = self.v1_filter(lum)
        v1_norm = self.v1_norm(v1_energy)
        mt_lin = self.mt_integ(v1_norm)
        mt_out = self.mt_norm(mt_lin)

        form_out = self.form_pathway(dkl)
        return mt_out, form_out

    def compute_difference(
        self,
        ref_video: torch.Tensor,
        dist_video: torch.Tensor,
        chunk_size: int = 30
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes multi-channel cortical difference between ref and dist video tensors (T, H, W, 3).
        
        Sequentially computes ref and dist features per sub-chunk to strictly keep peak VRAM < 6 GB.
        """
        device = next(self.parameters()).device
        T_len = ref_video.shape[0]
        H_len = ref_video.shape[1]
        kt = self.v1_kernel_size[0]
        halo = (kt - 1)
        s_margin = 16 if H_len >= 64 else max(1, H_len // 8)

        chunk_mae_slow = []
        chunk_mae_med = []
        chunk_mae_fast = []
        chunk_mae_fa = []
        chunk_mae_frg = []
        chunk_mae_fby = []

        step = chunk_size
        num_chunks = math.ceil(T_len / step)

        for c_idx in range(num_chunks):
            t_start = c_idx * step
            t_end = min(t_start + step, T_len)

            t_start_padded = max(0, t_start - halo)
            t_end_padded = min(T_len, t_end + halo)

            c_valid_start = t_start - t_start_padded
            c_valid_end = c_valid_start + (t_end - t_start)

            # 1. Process ref chunk on GPU
            ref_chunk = ref_video[t_start_padded:t_end_padded].unsqueeze(0).to(device)
            ref_mt_c, ref_form_c = self.forward_features(ref_chunk)
            ref_m_c = ref_mt_c[:, :, c_valid_start:c_valid_end, s_margin:-s_margin, s_margin:-s_margin].clone()
            ref_f_c = ref_form_c[:, :, :, c_valid_start:c_valid_end, s_margin:-s_margin, s_margin:-s_margin].clone()
            del ref_chunk, ref_mt_c, ref_form_c

            # 2. Process dist chunk on GPU
            dist_chunk = dist_video[t_start_padded:t_end_padded].unsqueeze(0).to(device)
            dist_mt_c, dist_form_c = self.forward_features(dist_chunk)
            dist_m_c = dist_mt_c[:, :, c_valid_start:c_valid_end, s_margin:-s_margin, s_margin:-s_margin].clone()
            dist_f_c = dist_form_c[:, :, :, c_valid_start:c_valid_end, s_margin:-s_margin, s_margin:-s_margin].clone()
            del dist_chunk, dist_mt_c, dist_form_c

            # 3. Compute MAEs for this chunk
            m_slow = torch.mean(torch.abs(ref_m_c[:, 0:8] - dist_m_c[:, 0:8]))
            m_med = torch.mean(torch.abs(ref_m_c[:, 8:16] - dist_m_c[:, 8:16]))
            m_fast = torch.mean(torch.abs(ref_m_c[:, 16:24] - dist_m_c[:, 16:24]))
            m_fa = torch.mean(torch.abs(ref_f_c[:, 0] - dist_f_c[:, 0]))
            m_frg = torch.mean(torch.abs(ref_f_c[:, 1] - dist_f_c[:, 1]))
            m_fby = torch.mean(torch.abs(ref_f_c[:, 2] - dist_f_c[:, 2]))

            chunk_mae_slow.append(m_slow)
            chunk_mae_med.append(m_med)
            chunk_mae_fast.append(m_fast)
            chunk_mae_fa.append(m_fa)
            chunk_mae_frg.append(m_frg)
            chunk_mae_fby.append(m_fby)

            del ref_m_c, ref_f_c, dist_m_c, dist_f_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mae_slow = torch.stack(chunk_mae_slow).mean()
        mae_med = torch.stack(chunk_mae_med).mean()
        mae_fast = torch.stack(chunk_mae_fast).mean()
        mae_fa = torch.stack(chunk_mae_fa).mean()
        mae_frg = torch.stack(chunk_mae_frg).mean()
        mae_fby = torch.stack(chunk_mae_fby).mean()

        w_s = F.softplus(self.w_slow)
        w_m = F.softplus(self.w_med)
        w_f = F.softplus(self.w_fast)
        w_a = F.softplus(self.w_achrom)
        w_rg = F.softplus(self.w_rg)
        w_by = F.softplus(self.w_by)

        diff_v1mt = (
            w_s * mae_slow +
            w_m * mae_med +
            w_f * mae_fast +
            w_a * mae_fa +
            w_rg * mae_frg +
            w_by * mae_fby
        ).squeeze()

        inter = {
            'mae_slow': mae_slow,
            'mae_med': mae_med,
            'mae_fast': mae_fast,
            'mae_fa': mae_fa,
            'mae_frg': mae_frg,
            'mae_fby': mae_fby,
            'w_slow': w_s,
            'w_med': w_m,
            'w_fast': w_f,
            'w_achrom': w_a,
            'w_rg': w_rg,
            'w_by': w_by
        }
        return diff_v1mt, inter


# ==============================================================================
# 6. Canonical Hybrid ColorVideoVDP Model
# ==============================================================================

def math_softplus_inv(y: float) -> float:
    """Calculates inverse softplus: x = ln(exp(y) - 1)."""
    if y <= 0:
        return -10.0
    return math.log(math.exp(y) - 1.0) if y < 20.0 else y


class CanonicalHybridColorVideoVDP(nn.Module):
    """Canonical Hybrid Model combining ColorVideoVDP / Isotropic Backbone and Canonical V1-MT.
    
    Prediction formula:
      D_hybrid = w_cvvdp * D_cvvdp + w_v1mt * D_v1mt
      JOD_pred = jod_intercept - jod_slope * D_hybrid
    """

    def __init__(
        self,
        fps: float = 240.0,
        pixels_per_degree: float = 30.0,
        v1_kernel_size: Tuple[int, int, int] = (15, 15, 15),
        init_w_cvvdp: float = 0.50,
        init_w_v1mt: float = 0.50,
        jod_intercept: float = 10.0,
        jod_slope: float = 2.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.fps = fps
        self.pixels_per_degree = pixels_per_degree
        self.dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Canonical V1-MT Pathway
        self.v1mt_pathway = CanonicalV1MTModel(
            fps=fps,
            pixels_per_degree=pixels_per_degree,
            initial_speeds=(1.0, 4.0, 16.0),
            v1_kernel_size=v1_kernel_size,
            trainable_v1=False,
            trainable_mt_speeds=False
        ).to(self.dev)

        # Trainable combination weights and JOD calibration
        self.raw_w_cvvdp = nn.Parameter(torch.tensor(math_softplus_inv(init_w_cvvdp), dtype=torch.float32))
        self.raw_w_v1mt = nn.Parameter(torch.tensor(math_softplus_inv(init_w_v1mt), dtype=torch.float32))
        self.jod_intercept = nn.Parameter(torch.tensor(jod_intercept, dtype=torch.float32))
        self.raw_jod_slope = nn.Parameter(torch.tensor(math_softplus_inv(jod_slope), dtype=torch.float32))

    @property
    def w_cvvdp(self) -> torch.Tensor:
        return F.softplus(self.raw_w_cvvdp)

    @property
    def w_v1mt(self) -> torch.Tensor:
        return F.softplus(self.raw_w_v1mt)

    @property
    def jod_slope(self) -> torch.Tensor:
        return F.softplus(self.raw_jod_slope)

    def forward_from_features(
        self,
        d_cvvdp: torch.Tensor,
        d_v1mt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates predicted JOD directly from precomputed difference values.
        
        Args:
            d_cvvdp: Tensor of shape (B,) or scalar.
            d_v1mt: Tensor of shape (B,) or scalar.
            
        Returns:
            pred_jod: Calibrated JOD quality score.
            d_hybrid: Combined difference metric.
        """
        d_hybrid = self.w_cvvdp * d_cvvdp + self.w_v1mt * d_v1mt
        pred_jod = self.jod_intercept - self.jod_slope * d_hybrid
        return pred_jod, d_hybrid
