"""Primary Visual Cortex (V1) spatiotemporal motion energy model in PyTorch."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def synthesize_gabor_kernels(
    theta: torch.Tensor,
    sf: torch.Tensor,
    tf: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_t: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    pixels_per_degree: float,
    fps: float,
) -> torch.Tensor:
    """Generates 3D spatiotemporal Gabor filter bank kernels in Cartesian physical units.

    Orientation theta is defined such that:
    - 0 rad is Rightward motion (vertical carrier stripes)
    - pi/2 rad is Upward motion (horizontal carrier stripes)
    - pi rad is Leftward motion
    - 3*pi/2 rad is Downward motion

    Returns:
        Tensor of shape (2 * num_directions, 1, kt, kh, kw) containing even/odd quadrature pairs.
    """
    device = theta.device
    num_directions = theta.shape[0]
    kt, kh, kw = kernel_size

    # Coordinate grids in physical units (degrees and seconds)
    ts = torch.linspace(-kt / 2.0, kt / 2.0, kt, device=device) / fps
    ys = torch.linspace(-kh / 2.0, kh / 2.0, kh, device=device) / pixels_per_degree
    xs = torch.linspace(-kw / 2.0, kw / 2.0, kw, device=device) / pixels_per_degree

    T, Y, X = torch.meshgrid(ts, ys, xs, indexing="ij")  # (kt, kh, kw)

    # Invert Y so positive-Y corresponds to upward motion in Cartesian coordinates
    Y_cartesian = -Y

    # Expand grids: (num_directions, kt, kh, kw)
    X_exp = X.unsqueeze(0).expand(num_directions, -1, -1, -1)
    Y_exp = Y_cartesian.unsqueeze(0).expand(num_directions, -1, -1, -1)
    T_exp = T.unsqueeze(0).expand(num_directions, -1, -1, -1)

    theta_b = theta.view(-1, 1, 1, 1)
    sf_b = sf.view(-1, 1, 1, 1)
    tf_b = tf.view(-1, 1, 1, 1)
    sigma_x_b = (torch.abs(sigma_x) + 1e-5).view(-1, 1, 1, 1)
    sigma_y_b = (torch.abs(sigma_y) + 1e-5).view(-1, 1, 1, 1)
    sigma_t_b = (torch.abs(sigma_t) + 1e-5).view(-1, 1, 1, 1)

    # Wave vector components
    sf_x = sf_b * torch.cos(theta_b)
    sf_y = sf_b * torch.sin(theta_b)

    # Rotated coordinates for Gaussian envelope
    sin_theta = torch.sin(theta_b)
    cos_theta = torch.cos(theta_b)
    X_rot = X_exp * cos_theta + Y_exp * sin_theta
    Y_rot = -X_exp * sin_theta + Y_exp * cos_theta

    # 3D Gaussian envelope
    envelope = torch.exp(
        -(X_rot**2 / (2.0 * sigma_x_b**2) + Y_rot**2 / (2.0 * sigma_y_b**2) + T_exp**2 / (2.0 * sigma_t_b**2))
    )

    # Spatiotemporal carrier
    phase = 2.0 * math.pi * (sf_x * X_exp + sf_y * Y_exp - tf_b * T_exp)
    carrier_even = torch.cos(phase)
    carrier_odd = torch.sin(phase)

    g_even = envelope * carrier_even
    g_odd = envelope * carrier_odd

    # Enforce DC balance (zero spatial mean)
    g_even = g_even - torch.mean(g_even, dim=(-2, -1), keepdim=True)
    g_odd = g_odd - torch.mean(g_odd, dim=(-2, -1), keepdim=True)

    # Normalize filter energy
    eps = 1e-8
    g_even_norm = torch.sqrt(torch.sum(g_even**2, dim=(-3, -2, -1), keepdim=True) + eps)
    g_odd_norm = torch.sqrt(torch.sum(g_odd**2, dim=(-3, -2, -1), keepdim=True) + eps)

    g_even = g_even / g_even_norm
    g_odd = g_odd / g_odd_norm

    # Stack even and odd quadrature pairs: (num_directions, 2, kt, kh, kw)
    kernels_stacked = torch.stack([g_even, g_odd], dim=1)
    kernels_reshaped = kernels_stacked.view(2 * num_directions, 1, kt, kh, kw)

    return kernels_reshaped


class V1FilterLayer(nn.Module):
    """Linear 3D spatiotemporal Gabor filter layer."""

    def __init__(
        self,
        directions: Sequence[float] = (
            0.0,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
            math.pi,
            5 * math.pi / 4,
            3 * math.pi / 2,
            7 * math.pi / 4,
        ),
        sf: Union[float, Sequence[float]] = 1.5,
        tf: Union[float, Sequence[float]] = 2.4,
        sigma_x: Union[float, Sequence[float]] = 0.133,
        sigma_y: Union[float, Sequence[float]] = 0.133,
        sigma_t: Union[float, Sequence[float]] = 0.0667,
        kernel_size: Tuple[int, int, int] = (9, 15, 15),
        pixels_per_degree: float = 30.0,
        fps: float = 30.0,
    ):
        super().__init__()
        self.directions = list(directions)
        self.kernel_size = kernel_size
        self.pixels_per_degree = pixels_per_degree
        self.fps = fps

        num_dirs = len(self.directions)
        theta_arr = torch.tensor(self.directions, dtype=torch.float32)

        def _to_tensor(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                return torch.tensor(val, dtype=torch.float32)
            return torch.full((num_dirs,), float(val), dtype=torch.float32)

        self.register_buffer("theta", theta_arr)
        self.register_buffer("sf", _to_tensor(sf))
        self.register_buffer("tf", _to_tensor(tf))
        self.register_buffer("sigma_x", _to_tensor(sigma_x))
        self.register_buffer("sigma_y", _to_tensor(sigma_y))
        self.register_buffer("sigma_t", _to_tensor(sigma_t))

        # Precompute and register Gabor filters
        kernels = synthesize_gabor_kernels(
            self.theta,
            self.sf,
            self.tf,
            self.sigma_x,
            self.sigma_y,
            self.sigma_t,
            self.kernel_size,
            self.pixels_per_degree,
            self.fps,
        )
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolves video x with 3D Gabor quadrature filters and computes motion energy.

        Args:
            x: Input video tensor of shape (B, 1, T, H, W) or (B, T, H, W).

        Returns:
            energy: V1 motion energy of shape (B, N_v1, T, H, W).
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, T, H, W)

        b, _, t, h, w = x.shape
        kt, kh, kw = self.kernel_size
        pt = (kt - 1) // 2
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

        # Symmetric replicate padding across time and space
        x_pad = F.pad(x, (pw, pw, ph, ph, pt, pt), mode="replicate")

        # 3D convolution: output shape (B, 2 * num_directions, T, H, W)
        conv_out = F.conv3d(x_pad, self.kernels, stride=1, padding=0)

        num_dirs = len(self.directions)
        # Reshape to separate even and odd quadrature filters: (B, num_directions, 2, T, H, W)
        conv_split = conv_out.view(b, num_dirs, 2, t, h, w)

        # Adelson-Bergen motion energy: Even^2 + Odd^2
        motion_energy = torch.sum(conv_split**2, dim=2)  # (B, num_dirs, T, H, W)

        return motion_energy


class V1Normalization(nn.Module):
    """V1 divisive normalization layer.

    Normalizes motion energy by local spatial and cross-channel population activity.
    """

    def __init__(self, spatial_pool_size: Tuple[int, int] = (5, 5), sigma: float = 0.1):
        super().__init__()
        self.spatial_pool_size = spatial_pool_size
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        """Applies population divisive normalization.

        Args:
            energy: V1 motion energy of shape (B, N_v1, T, H, W).

        Returns:
            v1_norm: Normalized V1 responses of shape (B, N_v1, T, H, W).
        """
        b, n_v1, t, h, w = energy.shape
        kh, kw = self.spatial_pool_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2

        # Average across all direction channels: (B, 1, T, H, W)
        cross_channel_pool = torch.mean(energy, dim=1, keepdim=True)

        # Spatial average pooling
        pool_flat = cross_channel_pool.reshape(b * t, 1, h, w)
        if ph > 0 or pw > 0:
            padded = F.pad(pool_flat, (pw, pw, ph, ph), mode="replicate")
            pooled = F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1, padding=0)
        else:
            pooled = pool_flat

        pool_denom = pooled.view(b, 1, t, h, w)

        # Divisive normalization
        v1_norm = energy / (self.sigma ** 2 + pool_denom)
        return v1_norm


class V1Stage(nn.Module):
    """Full Primary Visual Cortex (V1) motion processing stage."""

    def __init__(
        self,
        directions: Sequence[float] = (
            0.0,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
            math.pi,
            5 * math.pi / 4,
            3 * math.pi / 2,
            7 * math.pi / 4,
        ),
        sf: Union[float, Sequence[float]] = 1.5,
        tf: Union[float, Sequence[float]] = 2.4,
        sigma_x: Union[float, Sequence[float]] = 0.133,
        sigma_y: Union[float, Sequence[float]] = 0.133,
        sigma_t: Union[float, Sequence[float]] = 0.0667,
        kernel_size: Tuple[int, int, int] = (9, 15, 15),
        spatial_pool_size: Tuple[int, int] = (5, 5),
        pixels_per_degree: float = 30.0,
        fps: float = 30.0,
        sigma_norm: float = 0.1,
    ):
        super().__init__()
        self.filter_layer = V1FilterLayer(
            directions=directions,
            sf=sf,
            tf=tf,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_t=sigma_t,
            kernel_size=kernel_size,
            pixels_per_degree=pixels_per_degree,
            fps=fps,
        )
        self.normalization = V1Normalization(
            spatial_pool_size=spatial_pool_size,
            sigma=sigma_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input video through V1 filter bank and divisive normalization.

        Args:
            x: Video tensor (B, 1, T, H, W) or (B, T, H, W).

        Returns:
            v1_norm: Normalized V1 motion energy (B, N_v1, T, H, W).
        """
        energy = self.filter_layer(x)
        v1_norm = self.normalization(energy)
        return v1_norm
