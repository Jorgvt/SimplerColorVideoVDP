"""Temporal filter bank and convolution for SimplerColorVideoVDP."""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SIGMA_TF = (5.79336, 14.1255, 6.63661, 0.12314)
DEFAULT_BETA_TF = (1.3314, 1.1196, 0.947901, 0.1898)


def compute_temporal_filters(
    fps: float,
    sigma_tf: Tuple[float, ...] = DEFAULT_SIGMA_TF,
    beta_tf: Tuple[float, ...] = DEFAULT_BETA_TF,
    device: torch.device = None,
) -> Tuple[List[torch.Tensor], int]:
    """Synthesizes the 4 temporal impulse responses via real iFFT.

    Channels:
        0: Achromatic Sustained (Y)
        1: Red-Green Sustained (RG)
        2: Yellow-Violet Sustained (YV)
        3: Achromatic Transient (Y)

    Returns:
        filters: List of 4 1D tensor impulse responses of length N.
        filter_len: Odd length N of the impulse response.
    """
    n = int(math.ceil(0.250 * fps / 2.0) * 2) + 1
    n_omega = int(n / 2) + 1

    dev = device if device is not None else torch.device("cpu")
    omega = torch.linspace(0, fps / 2.0, n_omega, device=dev).view(1, n_omega)

    sigma_t = torch.tensor(sigma_tf, dtype=torch.float32, device=dev)
    beta_t = torch.tensor(beta_tf, dtype=torch.float32, device=dev)

    r_freq = torch.empty((4, n_omega), dtype=torch.float32, device=dev)

    # Sustained channels (0, 1, 2)
    r_freq[0:3, :] = torch.exp(-(omega ** beta_t[0:3].view(3, 1)) / sigma_t[0:3].view(3, 1))

    # Transient channel (3) centered around 5 Hz
    omega_trans = 5.0
    r_freq[3:4, :] = torch.exp(-((omega ** beta_t[3] - (omega_trans ** beta_t[3])) ** 2) / sigma_t[3])

    filters = []
    for k in range(4):
        # Real iFFT + fftshift
        ir = torch.fft.fftshift(torch.real(torch.fft.irfft(r_freq[k, :], norm="backward", n=n)))
        filters.append(ir)

    return filters, n


def apply_temporal_filtering(
    test_dkl: torch.Tensor,
    ref_dkl: torch.Tensor,
    fps: float,
    temp_padding: str = "replicate",
) -> torch.Tensor:
    """Applies temporal filtering to test and reference DKL video tensors.

    Args:
        test_dkl: (B, 3, T, H, W) DKL test video.
        ref_dkl: (B, 3, T, H, W) DKL reference video.
        fps: Frames per second.
        temp_padding: Padding method ('replicate' or 'symmetric').

    Returns:
        Interleaved tensor R of shape (B, 8, T, H, W):
        [Test-AchS, Ref-AchS, Test-RG, Ref-RG, Test-YV, Ref-YV, Test-AchT, Ref-AchT].
    """
    b, _, t, h, w = test_dkl.shape
    device = test_dkl.device

    filters, fl = compute_temporal_filters(fps, device=device)

    # Prepare padded buffers for test and ref: shape (B, 3, fl - 1 + T, H, W)
    pad_len = fl - 1
    if temp_padding == "replicate":
        test_first = test_dkl[:, :, 0:1, :, :].repeat(1, 1, pad_len, 1, 1)
        ref_first = ref_dkl[:, :, 0:1, :, :].repeat(1, 1, pad_len, 1, 1)
        test_padded = torch.cat([test_first, test_dkl], dim=2)
        ref_padded = torch.cat([ref_first, ref_dkl], dim=2)
    elif temp_padding == "symmetric":
        # Mirror frames before frame 0
        mirror_indices = []
        for fi in range(-pad_len, 0):
            # Symmetric reflection index
            is_even = (math.floor((abs(fi) - 1) / (t - 1)) % 2) == 0 if t > 1 else True
            if is_even:
                pos_ind = ((abs(fi) - 1) % (t - 1)) + 1 if t > 1 else 0
            else:
                pos_ind = (fi % (t - 1)) if t > 1 else 0
            mirror_indices.append(pos_ind)

        test_pad = test_dkl[:, :, mirror_indices, :, :]
        ref_pad = ref_dkl[:, :, mirror_indices, :, :]
        test_padded = torch.cat([test_pad, test_dkl], dim=2)
        ref_padded = torch.cat([ref_pad, ref_dkl], dim=2)
    else:
        raise ValueError(f"Unknown padding method '{temp_padding}'")

    # Output interleaved tensor
    r = torch.empty((b, 8, t, h, w), dtype=torch.float32, device=device)

    for cc in range(4):
        sw_ch = 0 if cc == 3 else cc
        corr_filter = filters[cc].flip(0).view(1, 1, fl, 1, 1)

        # Apply 1D temporal convolution using unfold or slicing
        # Padded length is pad_len + T = fl - 1 + T
        for fi in range(t):
            t_slice = test_padded[:, sw_ch : sw_ch + 1, fi : fi + fl, :, :]
            r_slice = ref_padded[:, sw_ch : sw_ch + 1, fi : fi + fl, :, :]

            r[:, cc * 2 + 0, fi, :, :] = (t_slice * corr_filter).sum(dim=2)
            r[:, cc * 2 + 1, fi, :, :] = (r_slice * corr_filter).sum(dim=2)

    return r
