"""SimplerColorVideoVDP: Streamlined, boilerplate-free visual quality metric."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simplercolorvideovdp.colorspace import linear_rgb_to_dkl
from simplercolorvideovdp.csf import CastleCSF
from simplercolorvideovdp.display import (
    DisplayGeometry,
    DisplayPhotometry,
    load_display_model,
)
from simplercolorvideovdp.masking import ContrastMasking, OrientedContrastMasking
from simplercolorvideovdp.mt import SteerableMTStage
from simplercolorvideovdp.pooling import MetricPooling, lp_norm, metric_to_jod
from simplercolorvideovdp.pyramid import SteerableWeberPyramid, WeberLaplacianPyramid
from simplercolorvideovdp.temporal import apply_temporal_filtering


def _standardize_tensor(tensor: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    """Converts input tensor/array to float32 tensor of shape (B, C, T, H, W) in [0, 1]."""
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
            raise ValueError(f"Ambiguous 3D tensor shape: {t.shape}")
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
    else:
        raise ValueError(f"Unsupported tensor dimension: {t.dim()}")

    return t


class ColorVideoVDP(nn.Module):
    """Full-reference ColorVideoVDP perceptual quality metric (calibrated isotropic baseline)."""

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        temp_padding: str = "replicate",
    ):
        super().__init__()
        self.temp_padding = temp_padding

        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        self.csf = CastleCSF()
        self.masking = ContrastMasking()
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], WeberLaplacianPyramid] = {}

    def get_pyramid(self, width: int, height: int, device: torch.device) -> WeberLaplacianPyramid:
        key = (width, height)
        if key not in self._pyramid_cache or next(self._pyramid_cache[key].parameters(), torch.empty(0)).device != device:
            pyr = WeberLaplacianPyramid(width, height, self.ppd).to(device)
            self._pyramid_cache[key] = pyr
        return self._pyramid_cache[key]

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
        return _standardize_tensor(tensor, device)

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: float = 0.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")

        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)
            else:
                raise ValueError(f"Shape mismatch: {test_t.shape} vs {ref_t.shape}")

        b, _, t, h, w = test_t.shape
        is_image = t == 1

        if not is_image and fps <= 0.0:
            raise ValueError("When evaluating video sequences (T > 1), `fps` must be specified and > 0.")

        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)

        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=fps, temp_padding=self.temp_padding)
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
            t_f = b_bb[:, 0::2, ...]
            r_f = b_bb[:, 1::2, ...]
            log_l_bkg = l_bkg_pyr[bb]

            ch_h, ch_w = log_l_bkg.shape[-2], log_l_bkg.shape[-1]
            s = torch.empty((b, all_ch, t, ch_h, ch_w), dtype=torch.float32, device=device)

            for cc in range(all_ch):
                tch = 0 if cc < 3 else 1
                cch = cc if cc < 3 else 0
                s[:, cc : cc + 1, :, :, :] = self.csf.sensitivity(
                    rho=rho_bands[bb],
                    omega=omega_lookup[tch],
                    log_l_bkg=log_l_bkg[..., 1:2, :, :, :],
                    channel=cch,
                )

            d = self.masking(t_f, r_f, s, is_baseband=is_baseband)
            q_per_ch[:, :, :, bb] = self.pooling.pool_spatial(d)

        q_jod = self.pooling.pool_all_and_to_jod(q_per_ch, is_image=is_image).squeeze()

        if return_stats:
            stats = {
                "Q_per_ch": q_per_ch.detach().cpu().numpy(),
                "rho_band": rho_bands,
                "frames_per_second": fps if not is_image else 0.0,
                "width": w,
                "height": h,
                "num_frames": t,
            }
            return q_jod, stats

        return q_jod

    def loss(self, test: torch.Tensor, reference: torch.Tensor, fps: float = 0.0) -> torch.Tensor:
        return 10.0 - self.forward(test, reference, fps=fps)

    def predict(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: float = 0.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        return self.forward(test, reference, fps=fps, return_stats=return_stats)


class OrientedColorVideoVDP(nn.Module):
    """Orientation-selective ColorVideoVDP using Steerable Weber Pyramid (K=4 orientations)."""

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        num_orientations: int = 4,
        cross_orientation_weight: float = 0.25,
        enable_oblique_effect: bool = True,
        oblique_weight: float = 0.15,
        beta_ori: float = 0.5,
        temp_padding: str = "replicate",
    ):
        super().__init__()
        self.num_orientations = num_orientations
        self.cross_orientation_weight = cross_orientation_weight
        self.enable_oblique_effect = enable_oblique_effect
        self.oblique_weight = oblique_weight
        self.beta_ori = beta_ori
        self.temp_padding = temp_padding

        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        self.csf = CastleCSF()
        self.masking = OrientedContrastMasking(
            cross_orientation_weight=cross_orientation_weight,
            enable_oblique_effect=enable_oblique_effect,
            oblique_weight=oblique_weight,
        )
        self.pooling = MetricPooling()

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
        return _standardize_tensor(tensor, device)

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: float = 0.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")

        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)
            else:
                raise ValueError(f"Shape mismatch: {test_t.shape} vs {ref_t.shape}")

        b, _, t, h, w = test_t.shape
        is_image = t == 1

        if not is_image and fps <= 0.0:
            raise ValueError("When evaluating video sequences (T > 1), `fps` must be specified and > 0.")

        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)

        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=fps, temp_padding=self.temp_padding)
            all_ch = 4

        pyramid = self.get_pyramid(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)
        q_per_ori = []
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

            if return_stats:
                q_per_ori.append(q_spatial_k.detach().cpu())

            if k_dim > 1:
                q_band = lp_norm(q_spatial_k, self.beta_ori, dim=-1, normalize=True, keepdim=False)
            else:
                q_band = q_spatial_k.squeeze(-1)

            q_per_ch[:, :, :, bb] = q_band

        q_jod = self.pooling.pool_all_and_to_jod(q_per_ch, is_image=is_image).squeeze()

        if return_stats:
            stats = {
                "Q_per_ch": q_per_ch.detach().cpu().numpy(),
                "Q_per_ori": q_per_ori,
                "rho_band": rho_bands,
                "orientations_deg": [round(math.degrees(th), 1) for th in pyramid.orientations],
                "frames_per_second": fps if not is_image else 0.0,
                "width": w,
                "height": h,
                "num_frames": t,
            }
            return q_jod, stats

        return q_jod

    def loss(self, test: torch.Tensor, reference: torch.Tensor, fps: float = 0.0) -> torch.Tensor:
        return 10.0 - self.forward(test, reference, fps=fps)


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
        cross_orientation_weight: float = 0.25,
        enable_oblique_effect: bool = True,
        oblique_weight: float = 0.15,
        beta_ori: float = 0.5,
        motion_weight: float = 0.2,
        stream_beta: float = 4.0,
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
        self.stream_beta = stream_beta
        self.num_orientations = num_orientations
        self.cross_orientation_weight = cross_orientation_weight
        self.enable_oblique_effect = enable_oblique_effect
        self.oblique_weight = oblique_weight
        self.beta_ori = beta_ori
        self.mt_directions = list(mt_directions)
        self.mt_speeds = list(mt_speeds)
        self.sigma_p = sigma_p
        self.spatial_pool_size = spatial_pool_size

        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        self.csf = CastleCSF()
        self.masking = OrientedContrastMasking(
            cross_orientation_weight=cross_orientation_weight,
            enable_oblique_effect=enable_oblique_effect,
            oblique_weight=oblique_weight,
        )
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], SteerableWeberPyramid] = {}
        self._mt_cache: Dict[Tuple[int, int], SteerableMTStage] = {}

    def get_pyramid_and_mt(
        self, width: int, height: int, device: torch.device
    ) -> Tuple[SteerableWeberPyramid, SteerableMTStage]:
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

            scale_freqs = pyr.get_freqs()[:-1]
            orientations = pyr.orientations

            mt_stage = SteerableMTStage(
                scale_freqs=scale_freqs,
                orientations=orientations,
                temp_freqs=(5.0,),
                mt_directions=self.mt_directions,
                mt_speeds=self.mt_speeds,
                sigma_p=self.sigma_p,
                spatial_pool_size=self.spatial_pool_size,
            ).to(device)

            self._pyramid_cache[key] = pyr
            self._mt_cache[key] = mt_stage

        return self._pyramid_cache[key], self._mt_cache[key]

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
        return _standardize_tensor(tensor, device)

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: Optional[float] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")
        actual_fps = fps if fps is not None else self.fps

        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)

        b, _, t, h, w = test_t.shape
        is_image = t == 1

        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)
        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=actual_fps, temp_padding="replicate")
            all_ch = 4

        pyramid, mt_stage = self.get_pyramid_and_mt(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)
        omega_lookup = (0.0, 5.0)

        mt_test_linear = None
        mt_ref_linear = None
        if not is_image and all_ch == 4:
            mt_h = min(h, 180)
            mt_w = min(w, 320)
            mt_test_linear = torch.zeros((b, mt_stage.num_mt_channels, t, mt_h, mt_w), dtype=torch.float32, device=device)
            mt_ref_linear = torch.zeros((b, mt_stage.num_mt_channels, t, mt_h, mt_w), dtype=torch.float32, device=device)

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
                q_band = lp_norm(q_spatial_k, self.beta_ori, dim=-1, normalize=True, keepdim=False)
                if not is_image and all_ch == 4 and mt_test_linear is not None and mt_ref_linear is not None:
                    # Pass Achromatic Transient subband ONLY (t_f[:, 3:4] * s[:, 3:4])
                    v1_test_trans = t_f[:, 3:4] * s[:, 3:4]
                    v1_ref_trans = r_f[:, 3:4] * s[:, 3:4]

                    mt_test_linear.add_(mt_stage.integration.project_scale(v1_test_trans, scale_idx=bb, target_size=(mt_h, mt_w)))
                    mt_ref_linear.add_(mt_stage.integration.project_scale(v1_ref_trans, scale_idx=bb, target_size=(mt_h, mt_w)))
            else:
                q_band = q_spatial_k.squeeze(-1)

            q_per_ch[:, :, :, bb] = q_band

        # 1. Ventral Form Stream raw Q (clean across all 4 channels without MT contamination)
        raw_q_form = self.pooling.pool_raw_q(q_per_ch, is_image=is_image).squeeze()

        # 2. Dedicated Dorsal Motion Stream raw Q
        if not is_image and mt_test_linear is not None and mt_ref_linear is not None:
            mt_test_pooled = mt_stage.integration.pool_spatial(mt_test_linear)
            mt_ref_pooled = mt_stage.integration.pool_spatial(mt_ref_linear)
            mt_test = mt_stage.normalization(mt_test_pooled)
            mt_ref = mt_stage.normalization(mt_ref_pooled)

            mt_diff = torch.abs(mt_test - mt_ref)
            q_mt_spatial = lp_norm(mt_diff, self.pooling.beta, dim=(-2, -1), normalize=True)
            q_mt_vel = lp_norm(q_mt_spatial, 2.0, dim=1, normalize=True)  # (B, T)
            raw_q_motion = lp_norm(q_mt_vel, self.pooling.beta_t, dim=1, normalize=True).squeeze()

            # 3. Stream-level Minkowski combination
            beta_st = self.stream_beta
            raw_q_total = (raw_q_form ** beta_st + self.motion_weight * (raw_q_motion ** beta_st)) ** (1.0 / beta_st)
        else:
            raw_q_motion = torch.zeros_like(raw_q_form)
            raw_q_total = raw_q_form

        q_jod = metric_to_jod(raw_q_total, self.pooling.jod_a.item(), self.pooling.jod_exp.item()).squeeze()

        if return_stats:
            stats = {
                "Q_per_ch": q_per_ch.detach().cpu().numpy(),
                "raw_q_form": float(raw_q_form.item()) if raw_q_form.dim() == 0 else raw_q_form.detach().cpu().numpy(),
                "raw_q_motion": float(raw_q_motion.item()) if raw_q_motion.dim() == 0 else raw_q_motion.detach().cpu().numpy(),
                "raw_q_total": float(raw_q_total.item()) if raw_q_total.dim() == 0 else raw_q_total.detach().cpu().numpy(),
                "width": w,
                "height": h,
                "num_frames": t,
                "fps": actual_fps,
            }
            return q_jod, stats

        return q_jod

    def loss(self, test: torch.Tensor, reference: torch.Tensor, fps: Optional[float] = None) -> torch.Tensor:
        return 10.0 - self.forward(test, reference, fps=fps)


class HybridColorVideoVDP(nn.Module):
    """Hybrid Video Quality Metric: Calibrated Isotropic Laplacian Backbone + Parallel Steerable MT Stream."""

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        fps: float = 30.0,
        num_orientations: int = 4,
        motion_weight: float = 500.0,
        stream_beta: float = 2.0,
        temp_padding: str = "replicate",
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
        self.stream_beta = stream_beta
        self.num_orientations = num_orientations
        self.temp_padding = temp_padding
        self.mt_directions = list(mt_directions)
        self.mt_speeds = list(mt_speeds)
        self.sigma_p = sigma_p
        self.spatial_pool_size = spatial_pool_size

        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        self.csf = CastleCSF()
        self.masking = ContrastMasking()
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], WeberLaplacianPyramid] = {}
        self._steerable_cache: Dict[Tuple[int, int], Tuple[SteerableWeberPyramid, SteerableMTStage]] = {}

    def get_pyramid(self, width: int, height: int, device: torch.device) -> WeberLaplacianPyramid:
        key = (width, height)
        if key not in self._pyramid_cache or next(self._pyramid_cache[key].parameters(), torch.empty(0)).device != device:
            pyr = WeberLaplacianPyramid(width, height, self.ppd).to(device)
            self._pyramid_cache[key] = pyr
        return self._pyramid_cache[key]

    def get_steerable_and_mt(
        self, width: int, height: int, device: torch.device
    ) -> Tuple[SteerableWeberPyramid, SteerableMTStage]:
        key = (width, height)
        if (
            key not in self._steerable_cache
            or next(self._steerable_cache[key][0].parameters(), torch.empty(0)).device != device
        ):
            pyr = SteerableWeberPyramid(
                width=width,
                height=height,
                ppd=self.ppd,
                num_orientations=self.num_orientations,
            ).to(device)

            scale_freqs = pyr.get_freqs()[:-1]
            orientations = pyr.orientations

            mt_stage = SteerableMTStage(
                scale_freqs=scale_freqs,
                orientations=orientations,
                temp_freqs=(5.0,),
                mt_directions=self.mt_directions,
                mt_speeds=self.mt_speeds,
                sigma_p=self.sigma_p,
                spatial_pool_size=self.spatial_pool_size,
            ).to(device)

            self._steerable_cache[key] = (pyr, mt_stage)

        return self._steerable_cache[key]

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
        return _standardize_tensor(tensor, device)

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: Optional[float] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")
        actual_fps = fps if fps is not None else self.fps

        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)

        b, _, t, h, w = test_t.shape
        is_image = t == 1

        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)
        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=actual_fps, temp_padding=self.temp_padding)
            all_ch = 4

        # 1. Path A: Calibrated Isotropic Laplacian Backbone
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
            t_f = b_bb[:, 0::2, ...]
            r_f = b_bb[:, 1::2, ...]
            log_l_bkg = l_bkg_pyr[bb]

            ch_h, ch_w = log_l_bkg.shape[-2], log_l_bkg.shape[-1]
            s = torch.empty((b, all_ch, t, ch_h, ch_w), dtype=torch.float32, device=device)

            for cc in range(all_ch):
                tch = 0 if cc < 3 else 1
                cch = cc if cc < 3 else 0
                s[:, cc : cc + 1, :, :, :] = self.csf.sensitivity(
                    rho=rho_bands[bb],
                    omega=omega_lookup[tch],
                    log_l_bkg=log_l_bkg[..., 1:2, :, :, :],
                    channel=cch,
                )

            d = self.masking(t_f, r_f, s, is_baseband=is_baseband)
            q_per_ch[:, :, :, bb] = self.pooling.pool_spatial(d)

        raw_q_iso = self.pooling.pool_raw_q(q_per_ch, is_image=is_image).squeeze()

        # 2. Path B: Dedicated Transient MT Motion Stream
        if not is_image and all_ch == 4 and self.motion_weight > 0.0:
            steerable_pyr, mt_stage = self.get_steerable_and_mt(w, h, device)
            # Decompose Achromatic Transient channels only (r[:, 6:8, ...])
            r_trans = r[:, 6:8, ...]
            trans_bands, _ = steerable_pyr.decompose(r_trans)

            mt_h = min(h, 180)
            mt_w = min(w, 320)
            mt_test_linear = torch.zeros((b, mt_stage.num_mt_channels, t, mt_h, mt_w), dtype=torch.float32, device=device)
            mt_ref_linear = torch.zeros((b, mt_stage.num_mt_channels, t, mt_h, mt_w), dtype=torch.float32, device=device)

            for bb in range(num_bands - 1):
                b_bb = trans_bands[bb]  # (B, 2, T, K, H_bb, W_bb)
                t_f = b_bb[:, 0:1, ...]
                r_f = b_bb[:, 1:2, ...]

                s_trans = self.csf.sensitivity(
                    rho=rho_bands[bb],
                    omega=5.0,
                    log_l_bkg=l_bkg_pyr[bb][..., 1:2, :, :, :],
                    channel=0,
                ).unsqueeze(3)

                v1_test_trans = t_f * s_trans
                v1_ref_trans = r_f * s_trans

                mt_test_linear.add_(mt_stage.integration.project_scale(v1_test_trans, scale_idx=bb, target_size=(mt_h, mt_w)))
                mt_ref_linear.add_(mt_stage.integration.project_scale(v1_ref_trans, scale_idx=bb, target_size=(mt_h, mt_w)))

            mt_test_pooled = mt_stage.integration.pool_spatial(mt_test_linear)
            mt_ref_pooled = mt_stage.integration.pool_spatial(mt_ref_linear)
            mt_test = mt_stage.normalization(mt_test_pooled)
            mt_ref = mt_stage.normalization(mt_ref_pooled)

            mt_diff = torch.abs(mt_test - mt_ref)
            q_mt_spatial = lp_norm(mt_diff, self.pooling.beta, dim=(-2, -1), normalize=True)
            q_mt_vel = lp_norm(q_mt_spatial, 2.0, dim=1, normalize=True)
            raw_q_motion = lp_norm(q_mt_vel, self.pooling.beta_t, dim=1, normalize=True).squeeze()

            beta_st = self.stream_beta
            raw_q_total = (raw_q_iso ** beta_st + self.motion_weight * (raw_q_motion ** beta_st)) ** (1.0 / beta_st)
        else:
            raw_q_motion = torch.zeros_like(raw_q_iso)
            raw_q_total = raw_q_iso

        q_jod = metric_to_jod(raw_q_total, self.pooling.jod_a.item(), self.pooling.jod_exp.item()).squeeze()

        if return_stats:
            stats = {
                "Q_per_ch": q_per_ch.detach().cpu().numpy(),
                "raw_q_iso": float(raw_q_iso.item()) if raw_q_iso.dim() == 0 else raw_q_iso.detach().cpu().numpy(),
                "raw_q_motion": float(raw_q_motion.item()) if raw_q_motion.dim() == 0 else raw_q_motion.detach().cpu().numpy(),
                "raw_q_total": float(raw_q_total.item()) if raw_q_total.dim() == 0 else raw_q_total.detach().cpu().numpy(),
                "width": w,
                "height": h,
                "num_frames": t,
                "fps": actual_fps,
            }
            return q_jod, stats

        return q_jod

    def loss(self, test: torch.Tensor, reference: torch.Tensor, fps: Optional[float] = None) -> torch.Tensor:
        return 10.0 - self.forward(test, reference, fps=fps)


cvvdp = ColorVideoVDP
