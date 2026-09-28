"""Middle Temporal (MT / V5) Cortical Motion Stage on Steerable Pyramid Subbands.

Directly integrates multi-scale, multi-orientation, spatio-temporal subbands
from the Steerable Weber Pyramid using the Simoncelli & Heeger (1998) velocity-plane constraint.
Eliminates the need for redundant 3D Gabor convolutions or parallel streams.
"""

from typing import List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_steerable_velocity_plane_weights(
    scale_freqs: torch.Tensor,
    orientations: torch.Tensor,
    temp_freqs: torch.Tensor,
    mt_directions: torch.Tensor,
    mt_speeds: torch.Tensor,
    sigma_p: float = 0.5,
    subtract_mean: bool = False,
) -> torch.Tensor:
    """Computes velocity-plane integration weights from Steerable Pyramid coordinates to MT velocity channels.

    A V1 channel is indexed by (scale rho_i, orientation theta_k, temporal freq omega_m).
    An MT channel is indexed by (direction theta_mt, speed v_mt).

    Condition for maximum response: omega_m = v_mt * rho_i * cos(theta_k - theta_mt)

    Args:
        scale_freqs: 1D tensor of spatial center frequencies (cpd), shape (N_scales,).
        orientations: 1D tensor of orientation angles (rad), shape (N_ori,).
        temp_freqs: 1D tensor of temporal frequencies (Hz), shape (N_temp,) e.g. [0.0, 5.0].
        mt_directions: 1D tensor of target MT directions (rad), shape (N_mt_dirs,).
        mt_speeds: 1D tensor of target MT speeds (deg/s), shape (N_mt_speeds,).
        sigma_p: Velocity plane tuning thickness.
        subtract_mean: If True, subtracts mean across V1 channels for off-plane inhibition.

    Returns:
        Weight matrix of shape (N_v1_total, N_mt_total)
        where N_v1_total = N_scales * N_ori * N_temp, and N_mt_total = N_mt_dirs * N_mt_speeds.
    """
    device = scale_freqs.device

    # Create meshgrid of all V1 coordinates: (N_scales, N_ori, N_temp) -> flatten to (N_v1,)
    grid_rho, grid_theta, grid_omega = torch.meshgrid(scale_freqs, orientations, temp_freqs, indexing="ij")
    v1_rho = grid_rho.flatten()
    v1_theta = grid_theta.flatten()
    v1_omega = grid_omega.flatten()
    n_v1 = v1_rho.shape[0]

    # Create meshgrid of all MT coordinates: (N_mt_dirs, N_mt_speeds) -> flatten to (N_mt,)
    grid_mt_theta, grid_mt_speed = torch.meshgrid(mt_directions, mt_speeds, indexing="ij")
    mt_theta = grid_mt_theta.flatten()
    mt_speed = grid_mt_speed.flatten()
    n_mt = mt_theta.shape[0]

    # Compute dot product and frequency difference: (N_v1, N_mt)
    diff_angles = v1_theta.unsqueeze(1) - mt_theta.unsqueeze(0)
    dot_product = mt_speed.unsqueeze(0) * v1_rho.unsqueeze(1) * torch.abs(torch.cos(diff_angles))

    w = torch.exp(-((v1_omega.unsqueeze(1) - dot_product) ** 2) / (2.0 * sigma_p ** 2))

    if subtract_mean:
        w = w - torch.mean(w, dim=0, keepdim=True)
        pos_sum = torch.sum(torch.clamp(w, min=0.0), dim=0, keepdim=True)
        w = w / torch.clamp(pos_sum, min=1.0)
    else:
        col_sum = torch.sum(w, dim=0, keepdim=True)
        w = w / torch.clamp(col_sum, min=1.0)

    return w


class SteerableMTIntegration(nn.Module):
    """Integrates Steerable Pyramid subbands into MT velocity channels."""

    def __init__(
        self,
        scale_freqs: Sequence[float],
        orientations: Sequence[float],
        temp_freqs: Sequence[float] = (0.0, 5.0),
        mt_directions: Sequence[float] = (
            0.0,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
            math.pi,
            5 * math.pi / 4,
            3 * math.pi / 2,
            7 * math.pi / 4,
        ),
        mt_speeds: Sequence[float] = (1.6,),
        sigma_p: float = 0.5,
        spatial_pool_size: Tuple[int, int] = (15, 15),
        subtract_mean: bool = False,
    ):
        super().__init__()
        self.scale_freqs = list(scale_freqs)
        self.orientations = list(orientations)
        self.temp_freqs = list(temp_freqs)
        self.mt_directions = list(mt_directions)
        self.mt_speeds = list(mt_speeds)
        self.spatial_pool_size = spatial_pool_size

        t_scales = torch.tensor(self.scale_freqs, dtype=torch.float32)
        t_ori = torch.tensor(self.orientations, dtype=torch.float32)
        t_tf = torch.tensor(self.temp_freqs, dtype=torch.float32)
        t_mt_dir = torch.tensor(self.mt_directions, dtype=torch.float32)
        t_mt_spd = torch.tensor(self.mt_speeds, dtype=torch.float32)

        # Precompute integration weights
        weights = compute_steerable_velocity_plane_weights(
            t_scales, t_ori, t_tf, t_mt_dir, t_mt_spd, sigma_p=sigma_p, subtract_mean=subtract_mean
        )
        self.register_buffer("weights", weights)

        # Register MT channel properties for flow decoding
        grid_mt_dir, grid_mt_spd = torch.meshgrid(t_mt_dir, t_mt_spd, indexing="ij")
        self.register_buffer("mt_theta", grid_mt_dir.flatten())
        self.register_buffer("mt_speed", grid_mt_spd.flatten())

    @property
    def num_mt_channels(self) -> int:
        return len(self.mt_directions) * len(self.mt_speeds)

    def forward(self, v1_subbands: torch.Tensor) -> torch.Tensor:
        """Projects stacked V1 steerable subbands onto MT velocity channels.

        Args:
            v1_subbands: Tensor of shape (B, N_v1_total, T, H, W) where
                         N_v1_total = N_scales * N_ori * N_temp.

        Returns:
            mt_linear: Tensor of shape (B, N_mt_channels, T, H, W).
        """
        b, n_v1, t, h, w = v1_subbands.shape
        kh, kw = self.spatial_pool_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2

        # 1. Spatial pooling across larger MT receptive field (~10x V1 RF)
        sub_flat = v1_subbands.reshape(b * n_v1 * t, 1, h, w)
        if ph > 0 or pw > 0:
            padded = F.pad(sub_flat, (pw, pw, ph, ph), mode="replicate")
            pooled = F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1, padding=0)
        else:
            pooled = sub_flat

        pooled_v1 = pooled.view(b, n_v1, t, h, w)

        # 2. Velocity-plane matrix multiplication: (B, T, H, W, N_v1) x (N_v1, N_mt)
        pooled_perm = pooled_v1.permute(0, 2, 3, 4, 1)
        mt_linear = torch.matmul(pooled_perm, self.weights.to(v1_subbands.device))

        return mt_linear.permute(0, 4, 1, 2, 3)  # (B, N_mt, T, H, W)


class MTNormalization(nn.Module):
    """Middle Temporal (MT) divisive normalization with half-squaring rectification."""

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, mt_linear: torch.Tensor) -> torch.Tensor:
        """Applies half-squaring rectification and population divisive normalization.

        Args:
            mt_linear: (B, N_mt, T, H, W)

        Returns:
            mt_norm: (B, N_mt, T, H, W)
        """
        rectified = torch.clamp(mt_linear, min=0.0) ** 2
        denom_pool = torch.mean(rectified, dim=1, keepdim=True)
        return rectified / (self.sigma ** 2 + denom_pool)


class SteerableMTStage(nn.Module):
    """Full MT processing stage operating directly on Steerable Pyramid subbands."""

    def __init__(
        self,
        scale_freqs: Sequence[float],
        orientations: Sequence[float],
        temp_freqs: Sequence[float] = (0.0, 5.0),
        mt_directions: Sequence[float] = (
            0.0,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
            math.pi,
            5 * math.pi / 4,
            3 * math.pi / 2,
            7 * math.pi / 4,
        ),
        mt_speeds: Sequence[float] = (1.6,),
        sigma_p: float = 0.5,
        spatial_pool_size: Tuple[int, int] = (15, 15),
        sigma_norm: float = 0.1,
        subtract_mean: bool = False,
    ):
        super().__init__()
        self.integration = SteerableMTIntegration(
            scale_freqs=scale_freqs,
            orientations=orientations,
            temp_freqs=temp_freqs,
            mt_directions=mt_directions,
            mt_speeds=mt_speeds,
            sigma_p=sigma_p,
            spatial_pool_size=spatial_pool_size,
            subtract_mean=subtract_mean,
        )
        self.normalization = MTNormalization(sigma=sigma_norm)

    def decode_velocity_flow(self, mt_norm: torch.Tensor) -> torch.Tensor:
        """Decodes 2D velocity flow field (vx, vy in deg/s) via population vector readout."""
        vx_k = (self.integration.mt_speed * torch.cos(self.integration.mt_theta)).view(1, -1, 1, 1, 1)
        vy_k = (self.integration.mt_speed * torch.sin(self.integration.mt_theta)).view(1, -1, 1, 1, 1)

        total_act = torch.sum(mt_norm, dim=1, keepdim=True) + 1e-6
        vx_flow = torch.sum(mt_norm * vx_k, dim=1, keepdim=True) / total_act
        vy_flow = torch.sum(mt_norm * vy_k, dim=1, keepdim=True) / total_act

        return torch.cat([vx_flow, vy_flow], dim=1)

    def forward(self, v1_subbands: torch.Tensor) -> torch.Tensor:
        """Processes V1 steerable subbands into normalized MT pattern responses."""
        mt_linear = self.integration(v1_subbands)
        return self.normalization(mt_linear)
