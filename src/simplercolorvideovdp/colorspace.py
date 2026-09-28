"""Color space transformations for SimplerColorVideoVDP."""

import torch

# Standard color conversion matrices
XYZ_TO_LMS2006 = (
    (0.187596268556126, 0.585168649077728, -0.026384263306304),
    (-0.133397430663221, 0.405505777260049, 0.034502127690364),
    (0.000244379021663, -0.000542995890619, 0.019406849066323),
)

LMS2006_TO_DKLD65 = (
    (1.0, 1.0, 0.0),
    (1.0, -2.311130179947035, 0.0),
    (-1.0, -1.0, 50.977571328718781),
)

RGB709_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

RGB2020_TO_XYZ = (
    (0.6370, 0.1446, 0.1689),
    (0.2627, 0.6780, 0.0593),
    (0.0, 0.0281, 1.0610),
)


def get_rgb_to_dkl_matrix(source_colorspace: str = "sRGB", dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:
    """Computes the combined 3x3 matrix mapping linear RGB to DKLd65."""
    if source_colorspace in ("sRGB", "BT.709", "BT.709-linear"):
        rgb_to_xyz = torch.tensor(RGB709_TO_XYZ, dtype=dtype, device=device)
    elif source_colorspace in ("BT.2020", "BT.2020-PQ", "BT.2020-HLG", "BT.2020-linear"):
        rgb_to_xyz = torch.tensor(RGB2020_TO_XYZ, dtype=dtype, device=device)
    else:
        # Default to BT.709 / sRGB primaries
        rgb_to_xyz = torch.tensor(RGB709_TO_XYZ, dtype=dtype, device=device)

    xyz_to_lms = torch.tensor(XYZ_TO_LMS2006, dtype=dtype, device=device)
    lms_to_dkl = torch.tensor(LMS2006_TO_DKLD65, dtype=dtype, device=device)

    return lms_to_dkl @ xyz_to_lms @ rgb_to_xyz


def linear_rgb_to_dkl(rgb_lin: torch.Tensor, source_colorspace: str = "sRGB") -> torch.Tensor:
    """Converts linear RGB tensor (B, 3, T, H, W) or (B, 3, H, W) to DKLd65 color space.

    Returns:
        DKL tensor with channels: [0: Achromatic-Y, 1: Red-Green (RG), 2: Yellow-Violet (YV)].
    """
    m = get_rgb_to_dkl_matrix(source_colorspace, dtype=rgb_lin.dtype, device=rgb_lin.device)

    is_5d = rgb_lin.dim() == 5
    if is_5d:
        # Shape: (B, 3, T, H, W)
        dkl = torch.empty_like(rgb_lin)
        for c in range(3):
            dkl[:, c : c + 1, :, :, :] = torch.sum(rgb_lin * m[c, :].view(1, 3, 1, 1, 1), dim=1, keepdim=True)
    else:
        # Shape: (B, 3, H, W)
        dkl = torch.empty_like(rgb_lin)
        for c in range(3):
            dkl[:, c : c + 1, :, :] = torch.sum(rgb_lin * m[c, :].view(1, 3, 1, 1), dim=1, keepdim=True)

    return dkl
