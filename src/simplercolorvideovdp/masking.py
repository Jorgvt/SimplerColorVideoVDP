"""Contrast masking and phase uncertainty modules for SimplerColorVideoVDP."""

import math
from typing import Tuple, Union
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

DEFAULT_MASK_P = 2.264355182647705
DEFAULT_MASK_C = -0.7954971194267273
DEFAULT_MASK_Q = (
    1.302622675895691,
    2.8885908126831055,
    3.6807713508605957,
    3.588787317276001,
)
DEFAULT_XCM_WEIGHTS = (
    -0.18950104713439941, -5.962151050567627, -4.31834602355957, -1.9321587085723877,
    2.5655593872070312, 0.34406712651252747, -2.719646453857422, -0.4970424771308899,
    3.8118371963500977, -1.0051705837249756, -0.5193376541137695, -0.5653647780418396,
    -7.054771423339844, -5.527150630950928, -3.5106418132781982, -2.08804988861084,
)
DEFAULT_D_MAX = 2.5642454624176025
DEFAULT_PU_DILATE = 3.0


def safe_pow(x: torch.Tensor, p: Union[float, torch.Tensor]) -> torch.Tensor:
    """Exact differentiable power function matching ColorVideoVDP."""
    epsilon = torch.as_tensor(0.00001, device=x.device, dtype=x.dtype)
    return (x + epsilon) ** p - epsilon ** p


class ContrastMasking(nn.Module):
    """Applies cross-channel mutual masking with phase uncertainty to band-pass contrasts."""

    def __init__(
        self,
        mask_p: float = DEFAULT_MASK_P,
        mask_c: float = DEFAULT_MASK_C,
        mask_q: Tuple[float, ...] = DEFAULT_MASK_Q,
        xcm_weights: Tuple[float, ...] = DEFAULT_XCM_WEIGHTS,
        d_max: float = DEFAULT_D_MAX,
        pu_dilate: float = DEFAULT_PU_DILATE,
    ):
        super().__init__()
        self.mask_p = mask_p
        self.mask_c = mask_c
        self.d_max = d_max
        self.pu_dilate = pu_dilate
        self.pu_padsize = int(pu_dilate * 2.0)

        self.register_buffer("mask_q", torch.tensor(mask_q, dtype=torch.float32))

        # 4x4 cross-channel masking weights
        xcm_w = torch.tensor(xcm_weights, dtype=torch.float32).reshape(4, 4)
        self.register_buffer("xcm_weights", 2.0 ** xcm_w)

        self.register_buffer("ch_gain", torch.tensor([1.0, 1.45, 1.0, 1.0], dtype=torch.float32))

        # Phase uncertainty blur
        kernel_size = int(pu_dilate * 4.0) + 1
        self.pu_blur = GaussianBlur(kernel_size, pu_dilate)

    def phase_uncertainty(self, m: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian spatial blur to model phase uncertainty."""
        h, w = m.shape[-2], m.shape[-1]
        scale = 10.0 ** self.mask_c

        if self.pu_dilate > 0 and h > self.pu_padsize and w > self.pu_padsize:
            m_flat = m.reshape(-1, 1, h, w)
            m_blurred = self.pu_blur(m_flat).reshape(m.shape)
            return m_blurred * scale
        else:
            return m * scale

    def mask_pool(self, c: torch.Tensor) -> torch.Tensor:
        """Cross-channel masking summation."""
        num_ch = c.shape[1]
        w_mat = self.xcm_weights[:num_ch, :num_ch]

        m = torch.empty_like(c)
        for cc in range(num_ch):
            # Sum over channels weighted by column cc
            col_weights = w_mat[:, cc].view(1, num_ch, 1, 1, 1)
            m[:, cc : cc + 1, ...] = torch.sum(c * col_weights, dim=1, keepdim=True)

        return m

    def forward(self, t: torch.Tensor, r: torch.Tensor, s: torch.Tensor, is_baseband: bool = False) -> torch.Tensor:
        """Computes masked contrast differences.

        Args:
            t: Test band contrast (B, C, T, H, W)
            r: Reference band contrast (B, C, T, H, W)
            s: Sensitivity tensor (B, C, T, H, W)
            is_baseband: Whether this band is the baseband

        Returns:
            Difference tensor D of shape (B, C, T, H, W)
        """
        if is_baseband:
            return torch.abs(t - r) * s

        num_ch = t.shape[1]
        gain = self.ch_gain[:num_ch].view(1, num_ch, 1, 1, 1)

        t_p = t * s * gain
        r_p = r * s * gain

        # Mutual masking component
        m_mm = self.phase_uncertainty(torch.min(torch.abs(t_p), torch.abs(r_p)))

        q = self.mask_q[:num_ch].view(1, num_ch, 1, 1, 1)
        m_q = safe_pow(torch.abs(m_mm), q)

        m = self.mask_pool(m_q)

        # Unclamped difference
        d_u = safe_pow(torch.abs(t_p - r_p), self.mask_p) / (1.0 + m)

        # Soft difference clamping
        max_v = 10.0 ** self.d_max
        d = max_v * d_u / (max_v + d_u)

        return d


class OrientedContrastMasking(nn.Module):
    """Applies cross-channel and cross-orientation mutual masking with phase uncertainty."""

    def __init__(
        self,
        mask_p: float = DEFAULT_MASK_P,
        mask_c: float = DEFAULT_MASK_C,
        mask_q: Tuple[float, ...] = DEFAULT_MASK_Q,
        xcm_weights: Tuple[float, ...] = DEFAULT_XCM_WEIGHTS,
        d_max: float = DEFAULT_D_MAX,
        pu_dilate: float = DEFAULT_PU_DILATE,
        cross_orientation_weight: float = 0.25,
        enable_oblique_effect: bool = True,
    ):
        super().__init__()
        self.mask_p = mask_p
        self.mask_c = mask_c
        self.d_max = d_max
        self.pu_dilate = pu_dilate
        self.pu_padsize = int(pu_dilate * 2.0)
        self.cross_orientation_weight = cross_orientation_weight
        self.enable_oblique_effect = enable_oblique_effect

        self.register_buffer("mask_q", torch.tensor(mask_q, dtype=torch.float32))

        # 4x4 cross-channel masking weights: 2^xcm_w
        xcm_w = torch.tensor(xcm_weights, dtype=torch.float32).reshape(4, 4)
        self.register_buffer("xcm_weights", 2.0 ** xcm_w)

        self.register_buffer("ch_gain", torch.tensor([1.0, 1.45, 1.0, 1.0], dtype=torch.float32))

        # Gaussian blur for phase uncertainty
        kernel_size = int(pu_dilate * 4.0) + 1
        self.pu_blur = GaussianBlur(kernel_size, pu_dilate)

    def phase_uncertainty(self, m: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian spatial blur to model phase uncertainty."""
        h, w = m.shape[-2], m.shape[-1]
        scale = 10.0 ** self.mask_c

        if self.pu_dilate > 0 and h > self.pu_padsize and w > self.pu_padsize:
            m_flat = m.reshape(-1, 1, h, w)
            m_blurred = self.pu_blur(m_flat).reshape(m.shape)
            return m_blurred * scale
        else:
            return m * scale

    def _build_orientation_mask_matrix(self, num_orientations: int, device: torch.device) -> torch.Tensor:
        """Constructs cross-orientation interaction matrix W_ori of shape (K, K)."""
        if num_orientations == 1:
            return torch.ones((1, 1), device=device)

        angles = torch.linspace(0, math.pi * (1 - 1 / num_orientations), num_orientations, device=device)
        d_theta = torch.abs(angles.unsqueeze(0) - angles.unsqueeze(1))
        d_theta = torch.minimum(d_theta, math.pi - d_theta)

        w_ori = (1.0 - self.cross_orientation_weight) * (torch.cos(d_theta) ** 2) + self.cross_orientation_weight
        w_ori = w_ori / w_ori.sum(dim=-1, keepdim=True)
        return w_ori

    def mask_pool(self, c: torch.Tensor) -> torch.Tensor:
        """Cross-channel masking summation across C dimension."""
        num_ch = c.shape[1]
        w_mat = self.xcm_weights[:num_ch, :num_ch]

        m = torch.empty_like(c)
        for cc in range(num_ch):
            col_weights = w_mat[:, cc].view(1, num_ch, 1, 1, 1, 1)
            m[:, cc : cc + 1, ...] = torch.sum(c * col_weights, dim=1, keepdim=True)

        return m

    def forward(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        is_baseband: bool = False,
    ) -> torch.Tensor:
        """Computes masked contrast difference tensor D for oriented subbands.

        Args:
            t: Test band contrast (B, C, T, K, H, W)
            r: Reference band contrast (B, C, T, K, H, W)
            s: Sensitivity tensor (B, C, T, 1, H, W) or (B, C, T, K, H, W)
            is_baseband: Whether this band is the baseband

        Returns:
            Difference tensor D of shape (B, C, T, K, H, W)
        """
        if is_baseband:
            return torch.abs(t - r) * s

        num_ch = t.shape[1]
        k_dim = t.shape[3]
        device = t.device

        if self.enable_oblique_effect and k_dim > 1:
            angles = torch.linspace(0, math.pi * (1 - 1 / k_dim), k_dim, device=device).view(1, 1, 1, k_dim, 1, 1)
            oblique_factor = 1.0 - 0.15 * (torch.sin(2.0 * angles) ** 2)
            s = s * oblique_factor

        gain = self.ch_gain[:num_ch].view(1, num_ch, 1, 1, 1, 1)
        t_p = t * s * gain
        r_p = r * s * gain

        # 1. Mutual masking component: min(|T|, |R|) with Phase Uncertainty
        m_mm = self.phase_uncertainty(torch.min(torch.abs(t_p), torch.abs(r_p)))

        # 2. Power-law transducer: M^q
        q = self.mask_q[:num_ch].view(1, num_ch, 1, 1, 1, 1)
        m_q = safe_pow(torch.abs(m_mm), q)

        # 3. Cross-orientation masking summation
        if k_dim > 1:
            w_ori = self._build_orientation_mask_matrix(k_dim, device)
            m_q_perm = m_q.permute(0, 1, 2, 4, 5, 3)
            m_ori = torch.matmul(m_q_perm, w_ori.T).permute(0, 1, 2, 5, 3, 4)
        else:
            m_ori = m_q

        # 4. Cross-channel masking summation
        m = self.mask_pool(m_ori)

        # 5. Masked difference
        d_u = safe_pow(torch.abs(t_p - r_p), self.mask_p) / (1.0 + m)

        # 6. Soft difference clamping
        max_v = 10.0 ** self.d_max
        d = max_v * d_u / (max_v + d_u)

        return d
