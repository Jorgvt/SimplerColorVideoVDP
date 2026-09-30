"""Middle Temporal (MT / V5) visual motion processing stage in SimplerColorVideoVDP."""

import math
from typing import Optional, Sequence, Tuple
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

    Condition for maximum response: omega_m = v_mt * rho_i * |cos(theta_k - theta_mt)|

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
    """
    grid_rho, grid_theta, grid_omega = torch.meshgrid(scale_freqs, orientations, temp_freqs, indexing="ij")
    v1_rho = grid_rho.flatten()
    v1_theta = grid_theta.flatten()
    v1_omega = grid_omega.flatten()

    grid_mt_theta, grid_mt_speed = torch.meshgrid(mt_directions, mt_speeds, indexing="ij")
    mt_theta = grid_mt_theta.flatten()
    mt_speed = grid_mt_speed.flatten()

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

        weights = compute_steerable_velocity_plane_weights(
            t_scales, t_ori, t_tf, t_mt_dir, t_mt_spd, sigma_p=sigma_p, subtract_mean=subtract_mean
        )
        self.register_buffer("weights", weights)

        grid_mt_dir, grid_mt_spd = torch.meshgrid(t_mt_dir, t_mt_spd, indexing="ij")
        self.register_buffer("mt_theta", grid_mt_dir.flatten())
        self.register_buffer("mt_speed", grid_mt_spd.flatten())

    @property
    def num_mt_channels(self) -> int:
        return len(self.mt_directions) * len(self.mt_speeds)

    @property
    def n_v1_per_scale(self) -> int:
        return len(self.orientations) * len(self.temp_freqs)

    def forward_scale(
        self,
        v1_scale: torch.Tensor,
        scale_idx: int,
        target_size: Optional[Tuple[int, int]] = None,
        slice_size: int = 15,
    ) -> torch.Tensor:
        """Projects a single scale's V1 steerable subbands onto MT velocity channels.

        Args:
            v1_scale: Tensor of shape (B, 2, T, K, H_s, W_s) or (B, N_v1_per_scale, T, H_s, W_s).
            scale_idx: Integer index of the spatial scale.
            target_size: Optional (H, W) to upsample subbands to if multi-scale downsampled.
            slice_size: Temporal batch size to keep VRAM minimal.

        Returns:
            mt_scale_linear: (B, N_mt_channels, T, H_out, W_out)
        """
        device = v1_scale.device
        if v1_scale.dim() == 6:
            # (B, 2, T, K, H, W) -> (B, 2*K, T, H, W)
            b, n_tf, t, k_dim, h_s, w_s = v1_scale.shape
            v1_in = v1_scale.permute(0, 3, 1, 2, 4, 5).reshape(b, k_dim * n_tf, t, h_s, w_s)
        else:
            b, _, t, h_s, w_s = v1_scale.shape
            v1_in = v1_scale

        h_in, w_in = int(v1_in.shape[-2]), int(v1_in.shape[-1])
        h_out, w_out = (int(target_size[0]), int(target_size[1])) if target_size is not None else (h_in, w_in)
        n_v1_scale = v1_in.shape[1]

        # Extract weights for this specific scale
        start_idx = scale_idx * self.n_v1_per_scale
        end_idx = start_idx + n_v1_scale
        scale_weight_matrix = self.weights[start_idx:end_idx, :].to(device)

        kh, kw = self.spatial_pool_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2

        mt_out = torch.empty((b, self.num_mt_channels, t, h_out, w_out), dtype=v1_scale.dtype, device=device)

        eff_slice = slice_size if (t > slice_size and h_out * w_out >= 128 * 128) else t
        for t_start in range(0, t, eff_slice):
            t_end = min(t_start + eff_slice, t)
            t_len = t_end - t_start
            sub_t = v1_in[:, :, t_start:t_end, :, :]

            if (h_in != h_out) or (w_in != w_out):
                flat_sub = sub_t.permute(0, 2, 1, 3, 4).reshape(b * t_len * n_v1_scale, 1, h_in, w_in)
                sub_up = F.interpolate(flat_sub, size=(h_out, w_out), mode="bilinear", align_corners=False)
                sub_flat = sub_up.view(b * n_v1_scale * t_len, 1, h_out, w_out)
            else:
                sub_flat = sub_t.reshape(b * n_v1_scale * t_len, 1, h_out, w_out)

            if ph > 0 or pw > 0:
                padded = F.pad(sub_flat, (pw, pw, ph, ph), mode="replicate")
                pooled = F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1, padding=0)
            else:
                pooled = sub_flat

            pooled_v1 = pooled.view(b, n_v1_scale, t_len, h_out, w_out)
            pooled_perm = pooled_v1.permute(0, 2, 3, 4, 1)
            mt_linear_slice = torch.matmul(pooled_perm, scale_weight_matrix).permute(0, 4, 1, 2, 3)
            mt_out[:, :, t_start:t_end, :, :] = mt_linear_slice

        return mt_out

    def forward(self, v1_subbands: torch.Tensor, slice_size: int = 15) -> torch.Tensor:
        """Projects stacked V1 steerable subbands onto MT velocity channels.

        Args:
            v1_subbands: (B, N_v1_total, T, H, W)
            slice_size: Temporal batch size.

        Returns:
            mt_linear: (B, N_mt_channels, T, H, W)
        """
        b, n_v1, t, h, w = v1_subbands.shape
        device = v1_subbands.device
        kh, kw = self.spatial_pool_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2
        w_mat = self.weights.to(device)

        if t > slice_size and (h * w >= 128 * 128):
            mt_out = torch.empty((b, self.num_mt_channels, t, h, w), dtype=v1_subbands.dtype, device=device)
            for t_start in range(0, t, slice_size):
                t_end = min(t_start + slice_size, t)
                t_len = t_end - t_start
                sub_t = v1_subbands[:, :, t_start:t_end, :, :]
                sub_flat = sub_t.reshape(b * n_v1 * t_len, 1, h, w)

                if ph > 0 or pw > 0:
                    padded = F.pad(sub_flat, (pw, pw, ph, ph), mode="replicate")
                    pooled = F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1, padding=0)
                else:
                    pooled = sub_flat

                pooled_v1 = pooled.view(b, n_v1, t_len, h, w)
                pooled_perm = pooled_v1.permute(0, 2, 3, 4, 1)
                mt_linear_slice = torch.matmul(pooled_perm, w_mat).permute(0, 4, 1, 2, 3)
                mt_out[:, :, t_start:t_end, :, :] = mt_linear_slice
            return mt_out

        sub_flat = v1_subbands.reshape(b * n_v1 * t, 1, h, w)
        if ph > 0 or pw > 0:
            padded = F.pad(sub_flat, (pw, pw, ph, ph), mode="replicate")
            pooled = F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1, padding=0)
        else:
            pooled = sub_flat

        pooled_v1 = pooled.view(b, n_v1, t, h, w)
        pooled_perm = pooled_v1.permute(0, 2, 3, 4, 1)
        mt_linear = torch.matmul(pooled_perm, w_mat)

        return mt_linear.permute(0, 4, 1, 2, 3)


class MTNormalization(nn.Module):
    """Middle Temporal (MT) divisive normalization with half-squaring rectification."""

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, mt_linear: torch.Tensor) -> torch.Tensor:
        rectified = torch.clamp(mt_linear, min=0.0) ** 2
        denom_pool = torch.mean(rectified, dim=1, keepdim=True)
        return rectified / (self.sigma ** 2 + denom_pool)


class SteerableMTStage(nn.Module):
    """Full MT motion processing stage operating directly on Steerable Pyramid subbands."""

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

    @property
    def num_mt_channels(self) -> int:
        return self.integration.num_mt_channels

    def decode_velocity_flow(self, mt_norm: torch.Tensor) -> torch.Tensor:
        """Decodes 2D velocity flow field (vx, vy in deg/s) via population vector readout."""
        vx_k = (self.integration.mt_speed * torch.cos(self.integration.mt_theta)).view(1, -1, 1, 1, 1)
        vy_k = (self.integration.mt_speed * torch.sin(self.integration.mt_theta)).view(1, -1, 1, 1, 1)

        total_act = torch.sum(mt_norm, dim=1, keepdim=True) + 1e-6
        vx_flow = torch.sum(mt_norm * vx_k, dim=1, keepdim=True) / total_act
        vy_flow = torch.sum(mt_norm * vy_k, dim=1, keepdim=True) / total_act

        return torch.cat([vx_flow, vy_flow], dim=1)

    def forward(self, v1_subbands: torch.Tensor) -> torch.Tensor:
        mt_linear = self.integration(v1_subbands)
        return self.normalization(mt_linear)
