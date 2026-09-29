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
from simplercolorvideovdp.masking import (
    ContrastMasking,
    OrientedContrastMasking,
)
from simplercolorvideovdp.metric import (
    ColorVideoVDP,
    OrientedColorVideoVDP,
    V1MTColorVideoVDP,
    cvvdp,
)
from simplercolorvideovdp.mt import (
    MTNormalization,
    SteerableMTIntegration,
    SteerableMTStage,
    compute_steerable_velocity_plane_weights,
)
from simplercolorvideovdp.pooling import MetricPooling, metric_to_jod
from simplercolorvideovdp.pyramid import (
    SteerableWeberPyramid,
    WeberLaplacianPyramid,
)
from simplercolorvideovdp.temporal import (
    apply_temporal_filtering,
    compute_temporal_filters,
)

from simplercolorvideovdp.datasets import (
    ALL_SCENES,
    DEFAULT_VAL_SCENES,
    GAIM240TorchDataset,
    create_gaim240_dataloader,
    read_video_ffmpeg,
)

__version__ = "0.1.0"

__all__ = [
    # Metrics
    "ColorVideoVDP",
    "OrientedColorVideoVDP",
    "V1MTColorVideoVDP",
    "cvvdp",
    # Display & Color
    "DisplayPhotometry",
    "DisplayGeometry",
    "DISPLAY_PRESETS",
    "load_display_model",
    "linear_rgb_to_dkl",
    "get_rgb_to_dkl_matrix",
    # CSF & Pyramids
    "CastleCSF",
    "WeberLaplacianPyramid",
    "SteerableWeberPyramid",
    # Masking & MT Motion
    "ContrastMasking",
    "OrientedContrastMasking",
    "compute_steerable_velocity_plane_weights",
    "SteerableMTIntegration",
    "MTNormalization",
    "SteerableMTStage",
    # Temporal & Pooling
    "compute_temporal_filters",
    "apply_temporal_filtering",
    "MetricPooling",
    "metric_to_jod",
    # Datasets
    "GAIM240TorchDataset",
    "create_gaim240_dataloader",
    "read_video_ffmpeg",
    "ALL_SCENES",
    "DEFAULT_VAL_SCENES",
]

