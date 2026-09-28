"""Contrast Sensitivity Function (CastleCSF) for SimplerColorVideoVDP."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn


def _interp1_batch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Batch-wise 1D linear interpolation.

    Args:
        x: Query points of shape (N,)
        xp: Grid points of shape (M,) in strictly increasing order
        fp: Table values of shape (N, M)

    Returns:
        Interpolated values of shape (N,)
    """
    x = x.contiguous()
    xp = xp.contiguous()
    indices = torch.searchsorted(xp, x) - 1
    indices = torch.clamp(indices, 0, len(xp) - 2)

    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[torch.arange(fp.shape[0], device=fp.device), indices]
    y1 = fp[torch.arange(fp.shape[0], device=fp.device), indices + 1]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def _interp1_uniform(x_grid: torch.Tensor, v_table: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Fast linear interpolation for uniformly spaced 1D grid x_grid.

    Args:
        x_grid: 1D grid tensor of uniformly spaced coordinates
        v_table: 1D table values tensor
        x_q: Arbitrary shaped query tensor

    Returns:
        Interpolated values with the same shape as x_q
    """
    shp = x_q.shape
    flat_q = x_q.flatten()

    ind = ((flat_q - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (x_grid.numel() - 1)).clamp(0, x_grid.shape[0] - 1)
    ifrc = torch.frac(ind)
    imin = ind.to(dtype=torch.long)
    imax = (imin + 1).clamp(max=x_grid.shape[0] - 1)

    filtered = v_table[imin] * (1.0 - ifrc) + v_table[imax] * ifrc
    return filtered.reshape(shp)


class CastleCSF(nn.Module):
    """CastleCSF module providing contrast sensitivity lookups.

    Evaluates sensitivity as a function of spatial frequency (rho, in cpd),
    temporal frequency (omega, 0 or 5 Hz), background luminance (L_bkg, cd/m²),
    and chromatic channel.
    """

    def __init__(self, lut_path: Optional[Path] = None):
        super().__init__()
        if lut_path is None:
            lut_path = Path(__file__).parent / "data" / "csf_lut_weber_fixed_size.json"

        with open(lut_path, "r", encoding="utf-8") as f:
            csf_lut = json.load(f)

        log_l_bkg = torch.log10(torch.tensor(csf_lut["L_bkg"], dtype=torch.float32))
        log_rho = torch.log10(torch.tensor(csf_lut["rho"], dtype=torch.float32))
        omega = csf_lut["omega"]

        self.register_buffer("log_L_bkg", log_l_bkg)
        self.register_buffer("log_rho", log_rho)
        self.omega = omega

        # logS structure: [temp_freq_idx][chroma_idx]
        # oo = 0 (omega=0 Hz): 3 channels (Y, RG, YV)
        # oo = 1 (omega=5 Hz): 1 channel (Y transient)
        self.logS: List[List[torch.Tensor]] = []
        for oo in range(2):
            self.logS.append([])
            ch_num = 3 if oo == 0 else 1
            for cc in range(ch_num):
                field_name = f"o{self.omega[oo]}_c{cc + 1}"
                buf_name = f"logS_{oo}_{cc}"
                tensor = torch.tensor(csf_lut[field_name], dtype=torch.float32)
                self.register_buffer(buf_name, tensor)

        self._rho_cache: Dict[str, torch.Tensor] = {}

    def get_logS(self, oo: int, cc: int) -> torch.Tensor:
        return getattr(self, f"logS_{oo}_{cc}")

    def sensitivity(
        self,
        rho: float,
        omega: float,
        log_l_bkg: torch.Tensor,
        channel: int,
        sensitivity_correction_db: float = -0.2797423303127289,
    ) -> torch.Tensor:
        """Computes contrast sensitivity S.

        Args:
            rho: Spatial frequency in cycles per degree (cpd).
            omega: Temporal frequency in Hz (0 for sustained, 5 for transient).
            log_l_bkg: Log10 background luminance tensor (..., H, W).
            channel: Chromatic channel (0: Y, 1: RG, 2: YV).
            sensitivity_correction_db: Global sensitivity correction in dB.

        Returns:
            Sensitivity tensor S with same shape as log_l_bkg.
        """
        oo = 0 if omega == 0 else 1
        log_s = self.get_logS(oo, channel)

        cache_key = f"o{oo}_c{channel}_rho{rho}_{log_l_bkg.device}"
        if cache_key in self._rho_cache and self._rho_cache[cache_key].device == log_l_bkg.device:
            log_s_r = self._rho_cache[cache_key]
        else:
            n = self.log_L_bkg.numel()
            rho_tensor = torch.log10(torch.tensor(rho, dtype=torch.float32, device=log_l_bkg.device)).expand(n)
            log_s_r = _interp1_batch(rho_tensor, self.log_rho, log_s)
            self._rho_cache[cache_key] = log_s_r

        # Interpolate across background luminance
        s = 10.0 ** _interp1_uniform(self.log_L_bkg, log_s_r, log_l_bkg)

        if sensitivity_correction_db != 0.0:
            s = s * (10.0 ** (sensitivity_correction_db / 20.0))

        return s
