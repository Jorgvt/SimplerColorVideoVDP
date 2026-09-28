"""SimplerColorVideoVDP: Streamlined, boilerplate-free implementation of ColorVideoVDP."""

from simplercolorvideovdp.colorspace import (
    get_rgb_to_dkl_matrix,
    linear_rgb_to_dkl,
)
from simplercolorvideovdp.csf import CastleCSF
from simplercolorvideovdp.display import (
    DISPLAY_PRESETS,
    DisplayGeometry,
    DisplayPhotometry,
    load_display_model,
)
from simplercolorvideovdp.masking import ContrastMasking
from simplercolorvideovdp.metric import ColorVideoVDP, cvvdp
from simplercolorvideovdp.pooling import MetricPooling, metric_to_jod
from simplercolorvideovdp.pyramid import WeberLaplacianPyramid
from simplercolorvideovdp.temporal import (
    apply_temporal_filtering,
    compute_temporal_filters,
)

__version__ = "0.1.0"

__all__ = [
    "ColorVideoVDP",
    "cvvdp",
    "DisplayPhotometry",
    "DisplayGeometry",
    "DISPLAY_PRESETS",
    "load_display_model",
    "CastleCSF",
    "WeberLaplacianPyramid",
    "ContrastMasking",
    "MetricPooling",
    "metric_to_jod",
    "linear_rgb_to_dkl",
    "get_rgb_to_dkl_matrix",
    "compute_temporal_filters",
    "apply_temporal_filtering",
]
