"""Unified Steerable-MT Cortical Video Quality Metric.

Unified Single-Stream Architecture:
1. Opponent color transformation & temporal filtering (Sustained vs Transient).
2. Steerable Weber Pyramid (K=4 orientations, multiscale spatial decomposition).
3. CastleCSF contrast sensitivity and orientation-selective mutual masking.
4. Direct MT Velocity Plane Integration from Steerable Subbands (Simoncelli & Heeger).
5. Multi-stage Minkowski pooling and non-linear JOD mapping.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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
from simplercolorvideovdp.pooling import MetricPooling, lp_norm
from simplercolorvideovdp.temporal import apply_temporal_filtering

from experiments.exp01_orientation_selective_pyramid.masking_oriented import OrientedContrastMasking
from experiments.exp01_orientation_selective_pyramid.steerable_pyramid import SteerableWeberPyramid

from .mt import SteerableMTStage


class V1MTColorVideoVDP(nn.Module):
    """Unified Cortical Video Quality Metric with Steerable MT Processing."""

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        fps: float = 30.0,
        num_orientations: int = 4,
        motion_weight: float = 0.5,
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
    ):
        super().__init__()
        self.fps = fps
        self.motion_weight = motion_weight
        self.num_orientations = num_orientations
        self.mt_directions = list(mt_directions)
        self.mt_speeds = list(mt_speeds)
        self.sigma_p = sigma_p
        self.spatial_pool_size = spatial_pool_size

        # Display calibration
        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        # Core modules
        self.csf = CastleCSF()
        self.masking = OrientedContrastMasking(cross_orientation_weight=0.25)
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], SteerableWeberPyramid] = {}
        self._mt_cache: Dict[Tuple[int, int], SteerableMTStage] = {}

    def get_pyramid_and_mt(
        self, width: int, height: int, device: torch.device
    ) -> Tuple[SteerableWeberPyramid, SteerableMTStage]:
        """Retrieves or builds cached Steerable pyramid and matched MT stage."""
        key = (width, height)
        if (
            key not in self._pyramid_cache
            or next(self._pyramid_cache[key].parameters(), torch.empty(0)).device != device
        ):
            pyr = SteerableWeberPyramid(
                width=width,
                height=height,
                ppd=self.ppd,
                num_orientations=self.num_orientations,
            ).to(device)

            scale_freqs = pyr.get_freqs()[:-1]  # Exclude baseband for directional MT
            orientations = pyr.orientations

            mt_stage = SteerableMTStage(
                scale_freqs=scale_freqs,
                orientations=orientations,
                temp_freqs=(0.0, 5.0),
                mt_directions=self.mt_directions,
                mt_speeds=self.mt_speeds,
                sigma_p=self.sigma_p,
                spatial_pool_size=self.spatial_pool_size,
            ).to(device)

            self._pyramid_cache[key] = pyr
            self._mt_cache[key] = mt_stage

        return self._pyramid_cache[key], self._mt_cache[key]

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
        """Calculates Unified Steerable-MT Video Quality score (JOD)."""
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

        # 3. Temporal Decomposition
        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=actual_fps, temp_padding="replicate")
            all_ch = 4

        # 4. Steerable Pyramid Decomposition
        pyramid, mt_stage = self.get_pyramid_and_mt(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)
        omega_lookup = (0.0, 5.0)

        # Collect V1 subbands for MT integration: (B, N_v1_total, T, H, W)
        v1_test_subbands = []
        v1_ref_subbands = []

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
                # Collect intermediate directional subbands for MT motion processing
                if not is_image and all_ch == 4:
                    # Sustained Ach (ch 0) and Transient Ach (ch 3)
                    # Shape per scale: (B, K * 2, T, H, W)
                    v1_test_scale = torch.cat([t_f[:, 0:1], t_f[:, 3:4]], dim=1) * torch.cat([s[:, 0:1], s[:, 3:4]], dim=1)
                    v1_ref_scale = torch.cat([r_f[:, 0:1], r_f[:, 3:4]], dim=1) * torch.cat([s[:, 0:1], s[:, 3:4]], dim=1)
                    # Permute: (B, 2, T, K, H, W) -> (B, K * 2, T, H, W)
                    v1_test_subbands.append(v1_test_scale.permute(0, 3, 1, 2, 4, 5).reshape(b, -1, t, h, w))
                    v1_ref_subbands.append(v1_ref_scale.permute(0, 3, 1, 2, 4, 5).reshape(b, -1, t, h, w))
            else:
                q_band = q_spatial_k.squeeze(-1)

            q_per_ch[:, :, :, bb] = q_band

        # 5. Direct MT Velocity Integration from Steerable Subbands
        if not is_image and len(v1_test_subbands) > 0:
            # Stack across all scales: shape (B, N_scales * K * 2, T, H, W)
            v1_test_stacked = torch.cat(v1_test_subbands, dim=1)
            v1_ref_stacked = torch.cat(v1_ref_subbands, dim=1)

            # MT pattern cell responses
            mt_test = mt_stage(v1_test_stacked)  # (B, N_mt, T, H, W)
            mt_ref = mt_stage(v1_ref_stacked)

            # MT motion difference
            mt_diff = torch.abs(mt_test - mt_ref)
            q_mt_spatial = lp_norm(mt_diff, self.pooling.beta, dim=(-2, -1), normalize=True)  # (B, N_mt, T)
            q_motion = lp_norm(q_mt_spatial, 2.0, dim=1, normalize=True)  # (B, T)

            # Merge motion distortion into Achromatic channel at band 0
            q_per_ch[:, 0, :, 0] = q_per_ch[:, 0, :, 0] + self.motion_weight * q_motion

        # 6. Multi-Stage Minkowski Pooling & JOD Mapping
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
