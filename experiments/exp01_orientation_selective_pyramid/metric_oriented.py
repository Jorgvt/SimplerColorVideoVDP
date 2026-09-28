import math
from typing import Any, Dict, Optional, Tuple, Union
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

from .masking_oriented import OrientedContrastMasking
from .steerable_pyramid import SteerableWeberPyramid


class OrientedColorVideoVDP(nn.Module):
    """Orientation-selective ColorVideoVDP using Steerable Weber Pyramid.

    Decomposes images/videos into spatial frequency and orientation subbands
    (e.g., K=4: 0°, 45°, 90°, 135°), applying orientation-tuned contrast masking
    and Minkowski pooling across space, orientation, scale, and time.
    """

    def __init__(
        self,
        display_name: str = "standard_4k",
        photometry: Optional[DisplayPhotometry] = None,
        geometry: Optional[DisplayGeometry] = None,
        ppd: Optional[float] = None,
        num_orientations: int = 4,
        cross_orientation_weight: float = 0.25,
        enable_oblique_effect: bool = True,
        beta_ori: float = 2.0,
        temp_padding: str = "replicate",
    ):
        super().__init__()
        self.num_orientations = num_orientations
        self.cross_orientation_weight = cross_orientation_weight
        self.enable_oblique_effect = enable_oblique_effect
        self.beta_ori = beta_ori
        self.temp_padding = temp_padding

        # Display setup
        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        # Core modules
        self.csf = CastleCSF()
        self.masking = OrientedContrastMasking(
            cross_orientation_weight=cross_orientation_weight,
            enable_oblique_effect=enable_oblique_effect,
        )
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], SteerableWeberPyramid] = {}

    def get_pyramid(self, width: int, height: int, device: torch.device) -> SteerableWeberPyramid:
        """Retrieves or builds cached Steerable pyramid for given resolution."""
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

    def _normalize_tensor(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
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

    def forward(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: float = 0.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Calculates Oriented ColorVideoVDP quality score (JOD)."""
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")

        # 1. Normalize inputs
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

        # 2. Photometric conversion to linear cd/m²
        test_lin = self.photometry.forward(test_t)
        ref_lin = self.photometry.forward(ref_t)

        # 3. Color conversion to DKL
        test_dkl = linear_rgb_to_dkl(test_lin, self.photometry.source_colorspace)
        ref_dkl = linear_rgb_to_dkl(ref_lin, self.photometry.source_colorspace)

        # 4. Spatio-temporal representation R
        if is_image:
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=fps, temp_padding=self.temp_padding)
            all_ch = 4

        # 5. Steerable Pyramid Decomposition
        pyramid = self.get_pyramid(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1  # Baseband frequency

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)
        q_per_ori = []

        # 6. Evaluate bands with CSF and Masking
        omega_lookup = (0.0, 5.0)

        for bb in range(num_bands):
            is_baseband = bb == (num_bands - 1)
            b_bb = b_bands[bb]  # (B, 2*C, T, K, H, W)
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

            # Masked difference: (B, C, T, K, H, W)
            d = self.masking(t_f, r_f, s, is_baseband=is_baseband)

            # Spatial pooling over (H, W) -> (B, C, T, K)
            q_spatial_k = lp_norm(d, self.pooling.beta, dim=(-2, -1), normalize=True, keepdim=False)

            if return_stats:
                q_per_ori.append(q_spatial_k.detach().cpu())

            # Orientation pooling across K subbands: Minkowski norm normalized by K
            if k_dim > 1:
                q_band = lp_norm(q_spatial_k, self.beta_ori, dim=-1, normalize=True, keepdim=False)
            else:
                q_band = q_spatial_k.squeeze(-1)

            q_per_ch[:, :, :, bb] = q_band

        # 7. Multi-stage pooling across bands, channels, frames -> JOD
        q_jod = self.pooling.pool_all_and_to_jod(q_per_ch, is_image=is_image)
        q_jod = q_jod.squeeze()

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

    def loss(
        self,
        test: torch.Tensor,
        reference: torch.Tensor,
        fps: float = 0.0,
    ) -> torch.Tensor:
        """Computes minimization loss: (10.0 - Q_jod)."""
        q_jod = self.forward(test, reference, fps=fps)
        return 10.0 - q_jod

    def predict(
        self,
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
        fps: float = 0.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Convenience alias for forward quality prediction."""
        return self.forward(test, reference, fps=fps, return_stats=return_stats)
