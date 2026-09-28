"""Minkowski pooling and JOD mapping for SimplerColorVideoVDP."""

from typing import Tuple, Union
import torch
import torch.nn as nn

DEFAULT_BETA = 2.0
DEFAULT_BETA_T = 2.0
DEFAULT_BETA_TCH = 4.0
DEFAULT_BETA_SCH = 4.0
DEFAULT_JOD_A = 0.0439569391310215
DEFAULT_JOD_EXP = 0.9302042722702026
DEFAULT_IMAGE_INT = 0.577918291091919
DEFAULT_CH_CHROM_W = 1.0
DEFAULT_CH_TRANS_W = 0.8081134557723999
DEFAULT_BASEBAND_WEIGHT = (
    0.0036334486212581396,
    1.6627724170684814,
    4.11874532699585,
    25.25969886779785,
)


def safe_pow(x: torch.Tensor, p: Union[float, torch.Tensor]) -> torch.Tensor:
    """Exact differentiable power function matching ColorVideoVDP."""
    epsilon = torch.as_tensor(0.00001, device=x.device, dtype=x.dtype)
    return (x + epsilon) ** p - epsilon ** p


def lp_norm(
    x: torch.Tensor,
    p: Union[float, torch.Tensor],
    dim: Union[int, Tuple[int, ...]] = (-2, -1),
    normalize: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the Minkowski Lp-norm using safe_pow matching ColorVideoVDP."""
    if normalize:
        if isinstance(dim, tuple):
            n = 1.0
            for d in dim:
                n *= x.shape[d]
        else:
            n = float(x.shape[dim])
    else:
        n = 1.0

    summed = torch.sum(safe_pow(x, p), dim=dim, keepdim=keepdim)
    return safe_pow(summed / float(n), 1.0 / p)


def metric_to_jod(q: torch.Tensor, jod_a: float = DEFAULT_JOD_A, jod_exp: float = DEFAULT_JOD_EXP) -> torch.Tensor:
    """Converts contrast distortion metric Q to JOD quality scores in [0, 10]."""
    q_t = 0.1
    jod_a_p = jod_a * (q_t ** (jod_exp - 1.0))

    q_jod = torch.empty_like(q)
    mask_linear = q <= q_t
    mask_power = ~mask_linear

    q_jod[mask_linear] = 10.0 - jod_a_p * q[mask_linear]
    q_jod[mask_power] = 10.0 - jod_a * (q[mask_power] ** jod_exp)

    return q_jod


class MetricPooling(nn.Module):
    """Multi-stage Minkowski pooling and JOD conversion."""

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        beta_t: float = DEFAULT_BETA_T,
        beta_tch: float = DEFAULT_BETA_TCH,
        beta_sch: float = DEFAULT_BETA_SCH,
        jod_a: float = DEFAULT_JOD_A,
        jod_exp: float = DEFAULT_JOD_EXP,
        image_int: float = DEFAULT_IMAGE_INT,
        ch_chrom_w: float = DEFAULT_CH_CHROM_W,
        ch_trans_w: float = DEFAULT_CH_TRANS_W,
        baseband_weight: Tuple[float, ...] = DEFAULT_BASEBAND_WEIGHT,
    ):
        super().__init__()
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("beta_t", torch.tensor(beta_t, dtype=torch.float32))
        self.register_buffer("beta_tch", torch.tensor(beta_tch, dtype=torch.float32))
        self.register_buffer("beta_sch", torch.tensor(beta_sch, dtype=torch.float32))

        self.register_buffer("jod_a", torch.tensor(jod_a, dtype=torch.float32))
        self.register_buffer("jod_exp", torch.tensor(jod_exp, dtype=torch.float32))
        self.register_buffer("image_int", torch.tensor(image_int, dtype=torch.float32))

        # Channel weights: [Y-sust, RG-sust, YV-sust, Y-trans]
        self.register_buffer(
            "ch_weights",
            torch.tensor([1.0, ch_chrom_w, ch_chrom_w, ch_trans_w], dtype=torch.float32),
        )
        self.register_buffer(
            "baseband_weight",
            torch.tensor(baseband_weight, dtype=torch.float32),
        )

    def pool_spatial(self, d: torch.Tensor) -> torch.Tensor:
        """Spatial pooling over H and W dimensions (L_beta norm, normalized)."""
        return lp_norm(d, self.beta, dim=(-2, -1), normalize=True, keepdim=False)

    def pool_all_and_to_jod(self, q_per_ch: torch.Tensor, is_image: bool) -> torch.Tensor:
        """Pools quality tensor Q_per_ch (B, C, T, num_bands) across bands, channels, and time.

        Returns:
            JOD quality scores of shape (B,) or scalar if B=1.
        """
        num_ch = q_per_ch.shape[1]
        num_bands = q_per_ch.shape[3]

        # 1. Channel weights
        per_ch_w = self.ch_weights[:num_ch].view(1, num_ch, 1, 1)

        # 2. Spatial band weights (baseband has custom weight)
        per_sband_w = torch.ones((1, num_ch, 1, num_bands), dtype=torch.float32, device=q_per_ch.device)
        per_sband_w[:, :, 0, -1] = self.baseband_weight[:num_ch]

        # 3. Sum across spatial bands (L_beta_sch, non-normalized)
        q_sc = lp_norm(q_per_ch * per_ch_w * per_sband_w, self.beta_sch, dim=3, normalize=False, keepdim=False)

        # 4. Sum across chromatic and temporal channels (L_beta_tch, non-normalized)
        q_tc = lp_norm(q_sc, self.beta_tch, dim=1, normalize=False, keepdim=False)

        # 5. Sum across time (frames)
        if is_image:
            q = q_tc * self.image_int
        else:
            q = lp_norm(q_tc, self.beta_t, dim=1, normalize=True, keepdim=False)

        q = q.squeeze(-1) if (q.dim() > 1 and q.shape[-1] == 1) else q
        q_jod = metric_to_jod(q, self.jod_a.item(), self.jod_exp.item())

        return q_jod
