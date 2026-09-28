"""Dual-Stream Bio-Inspired Video Quality Metric with V1-MT Cortical Motion Processing."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn

from simplercolorvideovdp.colorspace import linear_rgb_to_dkl
from simplercolorvideovdp.csf import CastleCSF
from simplercolorvideovdp.display import (
    DisplayGeometry,
    DisplayPhotometry,
    load_display_model,
)
from simplercolorvideovdp.pooling import MetricPooling, lp_norm, metric_to_jod
from simplercolorvideovdp.temporal import apply_temporal_filtering

from experiments.exp01_orientation_selective_pyramid.masking_oriented import OrientedContrastMasking
from experiments.exp01_orientation_selective_pyramid.steerable_pyramid import SteerableWeberPyramid

from .v1_mt import V1MTModel


class V1MTColorVideoVDP(nn.Module):
    """Dual-stream cortical video quality metric.

    Combines:
    1. Ventral (Form) Stream: Steerable 4-orientation Weber Pyramid + CastleCSF + contrast masking.
    2. Dorsal (Motion) Stream: V1-MT spatiotemporal motion energy + MT velocity plane integration.
    """

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        fps: float = 30.0,
        num_orientations: int = 4,
        motion_weight: float = 0.35,
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
        mt_speed: float = 1.6,
    ):
        super().__init__()
        self.fps = fps
        self.motion_weight = motion_weight
        self.num_orientations = num_orientations

        # Display calibration
        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        # Ventral stream components
        self.csf = CastleCSF()
        self.masking = OrientedContrastMasking(cross_orientation_weight=0.25)
        self.pooling = MetricPooling()

        # Dorsal stream: Cortical V1-MT motion model
        self.v1_mt = V1MTModel(
            pixels_per_degree=self.ppd,
            fps=self.fps,
            mt_directions=mt_directions,
            mt_speed=mt_speed,
        )

        self._pyramid_cache: Dict[Tuple[int, int], SteerableWeberPyramid] = {}

    def get_pyramid(self, width: int, height: int, device: torch.device) -> SteerableWeberPyramid:
        key = (width, height)
        if key not in self._pyramid_cache or next(self._pyramid_cache[key].parameters(), torch.empty(0)).device != device:
            pyr = SteerableWeberPyramid(
                width=width,
                height=height,
                ppd=self.ppd,
                num_orientations=self.num_orientations,
            ).to(device)
            self._pyramid_cache[key] = pyr
        return self._pyramid_cache[key]

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            if tensor.dtype == np.uint8:
                t = torch.tensor(tensor, dtype=torch.float32, device=device) / 255.0
            elif tensor.dtype == np.uint16:
                t = torch.tensor(tensor.astype(np.int32), dtype=torch.float32, device=device) / 65535.0
            else:
                t = torch.tensor(tensor, dtype=torch.float32, device=device)
        else:
            if tensor.dtype == torch.uint8:
                t = tensor.to(device=device, dtype=torch.float32) / 255.0
            elif tensor.dtype in (torch.float16, torch.float64):
                t = tensor.to(device=device, dtype=torch.float32)
            else:
                t = tensor.to(device=device)

        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3):
                t = t.unsqueeze(0).unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            elif t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            else:
                raise ValueError(f"Ambiguous shape: {t.shape}")
        elif t.dim() == 4:
            if t.shape[1] in (1, 3):
                t = t.unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            else:
                t = t.unsqueeze(0)
                if t.shape[2] == 1:
                    t = t.repeat(1, 1, 3, 1, 1)
        elif t.dim() == 5:
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1, 1)

        return t

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: Optional[float] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Calculates Dual-Stream V1-MT perceptual quality score (JOD)."""
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")
        actual_fps = fps if fps is not None else self.fps

        # 1. Normalize
        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)

        b, _, t, h, w = test_t.shape
        is_image = t == 1

        # 2. Linear luminance cd/m² & DKL color
        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)
        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        # 3. Ventral (Form) Stream Processing
        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=actual_fps, temp_padding="replicate")
            all_ch = 4

        pyramid = self.get_pyramid(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)
        omega_lookup = (0.0, 5.0)

        for bb in range(num_bands):
            is_baseband = bb == (num_bands - 1)
            b_bb = b_bands[bb]
            k_dim = b_bb.shape[3]
            t_f = b_bb[:, 0::2, ...]
            r_f = b_bb[:, 1::2, ...]
            log_l_bkg = l_bkg_pyr[bb]

            ch_h, ch_w = log_l_bkg.shape[-2], log_l_bkg.shape[-1]
            s = torch.empty((b, all_ch, t, 1, ch_h, ch_w), dtype=torch.float32, device=device)

            for cc in range(all_ch):
                tch = 0 if cc < 3 else 1
                cch = cc if cc < 3 else 0
                s[:, cc : cc + 1, :, 0, :, :] = self.csf.sensitivity(
                    rho=rho_bands[bb],
                    omega=omega_lookup[tch],
                    log_l_bkg=log_l_bkg[..., 1:2, :, :, :],
                    channel=cch,
                )

            d = self.masking(t_f, r_f, s, is_baseband=is_baseband)
            q_spatial_k = lp_norm(d, self.pooling.beta, dim=(-2, -1), normalize=True, keepdim=False)

            if k_dim > 1:
                q_band = lp_norm(q_spatial_k, 2.0, dim=-1, normalize=True, keepdim=False)
            else:
                q_band = q_spatial_k.squeeze(-1)

            q_per_ch[:, :, :, bb] = q_band

        # 4. Dorsal (Motion) Stream Processing (for videos with T >= 3)
        if not is_image and t >= 3:
            # Achromatic luminance input to V1-MT: (B, 1, T, H, W)
            test_ach = test_dkl[:, 0:1, :, :, :]
            ref_ach = ref_dkl[:, 0:1, :, :, :]

            mt_test = self.v1_mt(test_ach)  # (B, N_mt, T, H, W)
            mt_ref = self.v1_mt(ref_ach)

            # Motion distortion difference across MT velocity channels
            mt_diff = torch.abs(mt_test - mt_ref)  # (B, N_mt, T, H, W)

            # Spatial & velocity channel pooling
            q_motion_spatial = lp_norm(mt_diff, self.pooling.beta, dim=(-2, -1), normalize=True)  # (B, N_mt, T)
            q_motion = lp_norm(q_motion_spatial, 2.0, dim=1, normalize=True)  # (B, T)

            # Motion contribution added as weighted distortion to sustained Ach channel
            q_per_ch[:, 0, :, 0] = q_per_ch[:, 0, :, 0] + self.motion_weight * q_motion

        # 5. Pooling & JOD Mapping
        q_jod = self.pooling.pool_all_and_to_jod(q_per_ch, is_image=is_image).squeeze()

        if return_stats:
            stats = {
                "Q_per_ch": q_per_ch.detach().cpu().numpy(),
                "width": w,
                "height": h,
                "num_frames": t,
                "fps": actual_fps,
            }
            return q_jod, stats

        return q_jod

    def loss(self, test: torch.Tensor, reference: torch.Tensor, fps: Optional[float] = None) -> torch.Tensor:
        """Minimization loss (10.0 - Q_jod)."""
        return 10.0 - self.forward(test, reference, fps=fps)
