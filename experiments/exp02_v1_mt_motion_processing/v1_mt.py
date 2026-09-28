"""Combined V1-MT Cortical Motion Pathway Model in PyTorch."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from .mt import MTStage
from .v1 import V1Stage


class V1MTModel(nn.Module):
    """Full V1-MT Cortical Motion Perception Model.

    Connects:
        1. V1 local direction-selective motion energy filters + divisive normalization.
        2. MT global velocity plane integration (Simoncelli & Heeger) + divisive normalization.
    """

    def __init__(
        self,
        v1_directions: Sequence[float] = (
            0.0,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
            math.pi,
            5 * math.pi / 4,
            3 * math.pi / 2,
            7 * math.pi / 4,
        ),
        v1_sf: Union[float, Sequence[float]] = 1.5,
        v1_tf: Union[float, Sequence[float]] = 2.4,
        v1_sigma_x: Union[float, Sequence[float]] = 0.133,
        v1_sigma_y: Union[float, Sequence[float]] = 0.133,
        v1_sigma_t: Union[float, Sequence[float]] = 0.0667,
        v1_kernel_size: Tuple[int, int, int] = (9, 15, 15),
        v1_spatial_pool_size: Tuple[int, int] = (5, 5),
        v1_sigma_norm: float = 0.1,
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
        mt_sigma_p: float = 0.5,
        mt_spatial_pool_size: Tuple[int, int] = (15, 15),
        mt_sigma_norm: float = 0.1,
        mt_subtract_mean: bool = False,
        pixels_per_degree: float = 30.0,
        fps: float = 30.0,
    ):
        super().__init__()
        self.pixels_per_degree = pixels_per_degree
        self.fps = fps

        # V1 Stage
        self.v1 = V1Stage(
            directions=v1_directions,
            sf=v1_sf,
            tf=v1_tf,
            sigma_x=v1_sigma_x,
            sigma_y=v1_sigma_y,
            sigma_t=v1_sigma_t,
            kernel_size=v1_kernel_size,
            spatial_pool_size=v1_spatial_pool_size,
            pixels_per_degree=pixels_per_degree,
            fps=fps,
            sigma_norm=v1_sigma_norm,
        )

        # MT Stage
        self.mt = MTStage(
            v1_directions=v1_directions,
            v1_sf=v1_sf,
            v1_tf=v1_tf,
            mt_directions=mt_directions,
            mt_speed=mt_speed,
            mt_speeds=mt_speeds,
            sigma_p=mt_sigma_p,
            spatial_pool_size=mt_spatial_pool_size,
            sigma_norm=mt_sigma_norm,
            subtract_mean=mt_subtract_mean,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes video through V1-MT pathway and returns normalized MT pattern responses.

        Args:
            x: Video tensor of shape (B, 1, T, H, W) or (B, T, H, W).

        Returns:
            mt_norm: Normalized MT responses of shape (B, N_mt, T, H, W).
        """
        v1_norm = self.v1(x)
        mt_norm = self.mt(v1_norm)
        return mt_norm

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns V1 normalized energy, MT normalized pattern responses, and decoded velocity flow."""
        v1_norm = self.v1(x)
        mt_norm = self.mt(v1_norm)
        flow = self.mt.decode_velocity_flow(mt_norm)
        return v1_norm, mt_norm, flow
