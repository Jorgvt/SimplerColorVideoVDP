"""Middle Temporal (MT / V5) visual motion processing stage in PyTorch.

Implements:
1. Simoncelli-Heeger velocity plane integration weights.
2. MT spatial integration with larger receptive fields.
3. MT population-wide divisive normalization with half-squaring rectification.
"""

from typing import Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_velocity_plane_weights(
    v1_directions: torch.Tensor,
    v1_sf: torch.Tensor,
    v1_tf: torch.Tensor,
    mt_directions: torch.Tensor,
    mt_speeds: torch.Tensor,
    sigma_p: float = 0.5,
    subtract_mean: bool = False,
) -> torch.Tensor:
    """Computes the Simoncelli-Heeger V1 -> MT velocity plane constraint integration weights.

    An MT neuron tuned to velocity (mt_direction, mt_speed) integrates responses from
    V1 neurons lying on or near the velocity plane: tf = mt_speed * sf * cos(theta_v1 - theta_mt).

    Args:
        v1_directions: Preferred direction of V1 channels in radians, shape (N_v1,).
        v1_sf: Preferred spatial frequency of V1 channels in cpd, shape (N_v1,).
        v1_tf: Preferred temporal frequency of V1 channels in Hz, shape (N_v1,).
        mt_directions: Preferred direction of MT channels in radians, shape (N_mt,).
        mt_speeds: Preferred speed of MT channels in deg/sec, shape (N_mt,).
        sigma_p: Velocity plane tuning thickness.
        subtract_mean: If True, subtracts mean across V1 channels to produce off-plane inhibition.

    Returns:
        Weight matrix of shape (N_v1, N_mt).
    """
    # (N_v1, 1) and (1, N_mt)
    diff_angles = v1_directions.unsqueeze(1) - mt_directions.unsqueeze(0)
    dot_product = mt_speeds.unsqueeze(0) * v1_sf.unsqueeze(1) * torch.cos(diff_angles)

    w = torch.exp(-((v1_tf.unsqueeze(1) - dot_product) ** 2) / (2.0 * sigma_p ** 2))

    if subtract_mean:
        # Off-plane inhibition
        w = w - torch.mean(w, dim=0, keepdim=True)
        pos_sum = torch.sum(torch.clamp(w, min=0.0), dim=0, keepdim=True) + 1e-8
        w = w / pos_sum
    else:
        w = w / (torch.sum(w, dim=0, keepdim=True) + 1e-8)

    return w


class MTIntegration(nn.Module):
    """Middle Temporal (MT) linear integration layer.

    Spatially pools local V1 motion energy responses (modeling the ~10x larger RF of MT)
    and projects them onto MT velocity channels via the velocity plane weight matrix.
    """

    def __init__(
        self,
        v1_directions: Sequence[float],
        v1_sf: Union[float, Sequence[float]] = 1.5,
        v1_tf: Union[float, Sequence[float]] = 2.4,
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
        mt_speed: Union[float, Sequence[float]] = 1.6,
        mt_speeds: Optional[Sequence[float]] = None,
        sigma_p: float = 0.5,
        spatial_pool_size: Tuple[int, int] = (15, 15),
        subtract_mean: bool = False,
    ):
        super().__init__()
        self.v1_directions = list(v1_directions)
        self.mt_directions = list(mt_directions)
        self.sigma_p = sigma_p
        self.spatial_pool_size = spatial_pool_size
        self.subtract_mean = subtract_mean

        num_v1 = len(self.v1_directions)
        theta_v1 = torch.tensor(self.v1_directions, dtype=torch.float32)

        if isinstance(v1_sf, (list, tuple, np.ndarray)):
            sf_v1 = torch.tensor(v1_sf, dtype=torch.float32)
        else:
            sf_v1 = torch.full((num_v1,), float(v1_sf), dtype=torch.float32)

        if isinstance(v1_tf, (list, tuple, np.ndarray)):
            tf_v1 = torch.tensor(v1_tf, dtype=torch.float32)
        else:
            tf_v1 = torch.full((num_v1,), float(v1_tf), dtype=torch.float32)

        # Build MT direction and speed grids
        if mt_speeds is not None:
            speeds_grid, dirs_grid = [], []
            for s in mt_speeds:
                for d in self.mt_directions:
                    speeds_grid.append(float(s))
                    dirs_grid.append(float(d))
            theta_mt = torch.tensor(dirs_grid, dtype=torch.float32)
            speed_mt = torch.tensor(speeds_grid, dtype=torch.float32)
        elif isinstance(mt_speed, (list, tuple, np.ndarray)):
            if len(mt_speed) == len(self.mt_directions):
                theta_mt = torch.tensor(self.mt_directions, dtype=torch.float32)
                speed_mt = torch.tensor(mt_speed, dtype=torch.float32)
            else:
                speeds_grid, dirs_grid = [], []
                for s in mt_speed:
                    for d in self.mt_directions:
                        speeds_grid.append(float(s))
                        dirs_grid.append(float(d))
                theta_mt = torch.tensor(dirs_grid, dtype=torch.float32)
                speed_mt = torch.tensor(speeds_grid, dtype=torch.float32)
        else:
            theta_mt = torch.tensor(self.mt_directions, dtype=torch.float32)
            speed_mt = torch.full_like(theta_mt, float(mt_speed))

        # Register velocity plane integration weights
        w_plane = compute_velocity_plane_weights(
            theta_v1, sf_v1, tf_v1, theta_mt, speed_mt, sigma_p, subtract_mean
        )
        self.register_buffer("weights", w_plane)
        self.register_buffer("theta_mt", theta_mt)
        self.register_buffer("speed_mt", speed_mt)

    def forward(self, v1_norm: torch.Tensor) -> torch.Tensor:
        """Projects V1 normalized energy onto MT velocity channels.

        Args:
            v1_norm: Normalized V1 responses of shape (B, N_v1, T, H, W) or (B, T, H, W, N_v1).

        Returns:
            mt_linear: Linear MT responses of shape (B, N_mt, T, H, W).
        """
        # Ensure channel format is (B, N_v1, T, H, W)
        if v1_norm.dim() == 5 and v1_norm.shape[-1] == len(self.v1_directions):
            # (B, T, H, W, N_v1) -> (B, N_v1, T, H, W)
            v1_norm = v1_norm.permute(0, 4, 1, 2, 3)

        b, n_v1, t, h, w = v1_norm.shape
        kh, kw = self.spatial_pool_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2

        # 1. Spatial pooling (AvgPool2d across H and W with symmetric replication padding)
        v1_flat = v1_norm.reshape(b * n_v1 * t, 1, h, w)
        if ph > 0 or pw > 0:
            v1_padded = F.pad(v1_flat, (pw, pw, ph, ph), mode="replicate")
            pooled = F.avg_pool2d(v1_padded, kernel_size=(kh, kw), stride=1, padding=0)
        else:
            pooled = v1_flat

        pooled_v1 = pooled.view(b, n_v1, t, h, w)

        # 2. Velocity-plane matrix multiplication: (B, T, H, W, N_v1) x (N_v1, N_mt)
        pooled_v1_perm = pooled_v1.permute(0, 2, 3, 4, 1)  # (B, T, H, W, N_v1)
        mt_linear = torch.matmul(pooled_v1_perm, self.weights)  # (B, T, H, W, N_mt)

        # Permute back to standard PyTorch NCHW format: (B, N_mt, T, H, W)
        return mt_linear.permute(0, 4, 1, 2, 3)


class MTNormalization(nn.Module):
    """Middle Temporal (MT) divisive normalization stage.

    Applies half-squaring rectification followed by population-wide divisive
    normalization across all MT velocity channels:
        R_MT = max(0, x)^2 / (sigma^2 + mean_{v}(max(0, x)^2))
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, mt_linear: torch.Tensor) -> torch.Tensor:
        """Applies rectification and population divisive normalization.

        Args:
            mt_linear: Linear MT responses of shape (B, N_mt, T, H, W).

        Returns:
            mt_norm: Normalized MT responses of shape (B, N_mt, T, H, W).
        """
        # Half-squaring rectification: max(0, x)^2
        rectified = torch.clamp(mt_linear, min=0.0) ** 2

        # Population-wide normalization across all velocity channels (dim 1)
        denom_pool = torch.mean(rectified, dim=1, keepdim=True)
        mt_norm = rectified / (self.sigma ** 2 + denom_pool)

        return mt_norm


class MTStage(nn.Module):
    """Full Middle Temporal (MT / V5) processing stage."""

    def __init__(
        self,
        v1_directions: Sequence[float],
        v1_sf: Union[float, Sequence[float]] = 1.5,
        v1_tf: Union[float, Sequence[float]] = 2.4,
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
        mt_speed: Union[float, Sequence[float]] = 1.6,
        mt_speeds: Optional[Sequence[float]] = None,
        sigma_p: float = 0.5,
        spatial_pool_size: Tuple[int, int] = (15, 15),
        sigma_norm: float = 0.1,
        subtract_mean: bool = False,
    ):
        super().__init__()
        self.integration = MTIntegration(
            v1_directions=v1_directions,
            v1_sf=v1_sf,
            v1_tf=v1_tf,
            mt_directions=mt_directions,
            mt_speed=mt_speed,
            mt_speeds=mt_speeds,
            sigma_p=sigma_p,
            spatial_pool_size=spatial_pool_size,
            subtract_mean=subtract_mean,
        )
        self.normalization = MTNormalization(sigma=sigma_norm)

    def decode_velocity_flow(self, mt_norm: torch.Tensor) -> torch.Tensor:
        """Decodes dense 2D velocity flow field (vx, vy) via vector population average.

        Args:
            mt_norm: Normalized MT responses of shape (B, N_mt, T, H, W).

        Returns:
            flow: Decoded velocity vector field of shape (B, 2, T, H, W) [vx, vy in deg/s].
        """
        # Preferred velocity vectors: vx = speed * cos(theta), vy = speed * sin(theta)
        vx_k = (self.integration.speed_mt * torch.cos(self.integration.theta_mt)).view(1, -1, 1, 1, 1)
        vy_k = (self.integration.speed_mt * torch.sin(self.integration.theta_mt)).view(1, -1, 1, 1, 1)

        total_activity = torch.sum(mt_norm, dim=1, keepdim=True) + 1e-6
        vx_flow = torch.sum(mt_norm * vx_k, dim=1, keepdim=True) / total_activity
        vy_flow = torch.sum(mt_norm * vy_k, dim=1, keepdim=True) / total_activity

        return torch.cat([vx_flow, vy_flow], dim=1)

    def forward(self, v1_norm: torch.Tensor) -> torch.Tensor:
        """Computes normalized MT pattern motion responses from V1 motion energy.

        Args:
            v1_norm: Normalized V1 responses of shape (B, N_v1, T, H, W).

        Returns:
            mt_norm: Normalized MT responses of shape (B, N_mt, T, H, W).
        """
        mt_linear = self.integration(v1_norm)
        mt_norm = self.normalization(mt_linear)
        return mt_norm
