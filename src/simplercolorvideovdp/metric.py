"""SimplerColorVideoVDP: Streamlined, boilerplate-free visual quality metric."""

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
from simplercolorvideovdp.masking import ContrastMasking
from simplercolorvideovdp.pooling import MetricPooling
from simplercolorvideovdp.pyramid import WeberLaplacianPyramid
from simplercolorvideovdp.temporal import apply_temporal_filtering


class ColorVideoVDP(nn.Module):
    """Full-reference ColorVideoVDP perceptual quality metric.

    Predicts visual difference in Just Objectionable Differences (JOD) between
    test and reference images or video sequences.

    Typical quality range:
        10 JOD: Visually indistinguishable (no perceptible difference)
        9-10 JOD: Barely noticeable difference (threshold quality)
        6-8 JOD: Mild to moderate visible distortion
        <6 JOD: Severe degradation
    """

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

        # Display setup
        self.photometry, self.geometry = load_display_model(
            display_name=display_name,
            photometry=photometry,
            geometry=geometry,
            ppd=ppd,
        )
        self.ppd = self.geometry.get_ppd()

        # Core computational modules
        self.csf = CastleCSF()
        self.masking = ContrastMasking()
        self.pooling = MetricPooling()

        self._pyramid_cache: Dict[Tuple[int, int], WeberLaplacianPyramid] = {}

    def get_pyramid(self, width: int, height: int, device: torch.device) -> WeberLaplacianPyramid:
        """Retrieves or constructs cached Laplacian pyramid for given dimensions."""
        key = (width, height)
        if key not in self._pyramid_cache or next(self._pyramid_cache[key].parameters(), torch.empty(0)).device != device:
            pyr = WeberLaplacianPyramid(width, height, self.ppd).to(device)
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
            elif tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
                t = tensor.to(device=device, dtype=torch.float32)
            else:
                t = tensor.to(device=device)

        # Standardize dimensions to (B, C, T, H, W)
        if t.dim() == 2:
            # (H, W) -> (1, 3, 1, H, W)
            t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        elif t.dim() == 3:
            # (C, H, W) -> (1, C, 1, H, W)
            if t.shape[0] in (1, 3):
                t = t.unsqueeze(0).unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            elif t.shape[-1] in (1, 3):
                # (H, W, C) -> (1, C, 1, H, W)
                t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            else:
                raise ValueError(f"Ambiguous 3D tensor shape: {t.shape}")
        elif t.dim() == 4:
            # (B, C, H, W) image batch -> (B, C, 1, H, W)
            if t.shape[1] in (1, 3):
                t = t.unsqueeze(2)
                if t.shape[1] == 1:
                    t = t.repeat(1, 3, 1, 1, 1)
            else:
                # Could be (T, C, H, W)
                t = t.unsqueeze(0)
                if t.shape[2] == 1:
                    t = t.repeat(1, 1, 3, 1, 1)
        elif t.dim() == 5:
            # (B, C, T, H, W)
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
        """Calculates ColorVideoVDP quality score (JOD) between test and reference.

        Args:
            test: Test image or video tensor/array (e.g. B, C, H, W or B, C, T, H, W).
            reference: Reference image or video tensor/array of matching shape.
            fps: Frames per second (must be >0 for video with T > 1).
            return_stats: If True, returns auxiliary statistics dictionary.

        Returns:
            Q_jod: JOD quality score tensor (scalar for single pair, or (B,) for batch).
            stats: (Optional) Dict containing Q_per_ch, rho_band, etc.
        """
        device = test.device if isinstance(test, torch.Tensor) else torch.device("cpu")

        # 1. Normalize inputs to (B, 3, T, H, W) in [0, 1]
        test_t = self._normalize_tensor(test, device)
        ref_t = self._normalize_tensor(reference, device)

        if test_t.shape != ref_t.shape:
            # Allow singleton batch broadcasting
            if test_t.shape[0] == 1 and ref_t.shape[0] > 1:
                test_t = test_t.repeat(ref_t.shape[0], 1, 1, 1, 1)
            elif ref_t.shape[0] == 1 and test_t.shape[0] > 1:
                ref_t = ref_t.repeat(test_t.shape[0], 1, 1, 1, 1)
            else:
                raise ValueError(f"Shape mismatch between test {test_t.shape} and ref {ref_t.shape}")

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
            # (B, 6, 1, H, W): [T_Y, R_Y, T_RG, R_RG, T_YV, R_YV]
            r = torch.empty((b, 6, 1, h, w), dtype=torch.float32, device=device)
            r[:, 0::2, :, :, :] = test_dkl
            r[:, 1::2, :, :, :] = ref_dkl
            all_ch = 3
        else:
            # (B, 8, T, H, W): [T_YS, R_YS, T_RG, R_RG, T_YV, R_YV, T_YT, R_YT]
            r = apply_temporal_filtering(test_dkl, ref_dkl, fps=fps, temp_padding=self.temp_padding)
            all_ch = 4

        # 5. Spatial Weber Laplacian Pyramid Decomposition
        pyramid = self.get_pyramid(w, h, device)
        b_bands, l_bkg_pyr = pyramid.decompose(r)
        num_bands = pyramid.band_count
        rho_bands = pyramid.get_freqs()
        rho_bands[-1] = 0.1  # Baseband frequency

        q_per_ch = torch.empty((b, all_ch, t, num_bands), dtype=torch.float32, device=device)

        # 6. Evaluate bands with CSF and Masking
        omega_lookup = (0.0, 5.0)

        for bb in range(num_bands):
            is_baseband = bb == (num_bands - 1)
            b_bb = b_bands[bb]
            t_f = b_bb[:, 0::2, ...]
            r_f = b_bb[:, 1::2, ...]
            log_l_bkg = l_bkg_pyr[bb]

            # Sensitivity S
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

            # Masked difference
            d = self.masking(t_f, r_f, s, is_baseband=is_baseband)

            # Spatial pooling (L2 norm over H and W)
            q_per_ch[:, :, :, bb] = self.pooling.pool_spatial(d)

        # 7. Pooling across bands, channels, and frames -> JOD
        q_jod = self.pooling.pool_all_and_to_jod(q_per_ch, is_image=is_image)
        q_jod = q_jod.squeeze()

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


# Backward-compatibility alias
cvvdp = ColorVideoVDP
