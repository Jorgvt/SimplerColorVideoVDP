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
    HybridColorVideoVDP,
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
from simplercolorvideovdp.v1mt_canonical import (
    CanonicalHybridColorVideoVDP,
    CanonicalV1MTModel,
    MTIntegrationTorch,
    MTNormalizationTorch,
    SustainedFormModelTorch,
    V1FilterLayerTorch,
    V1NormalizationTorch,
    compute_velocity_plane_weights_torch,
    pad_symmetric_5d,
    rgb_to_dkl_torch,
    synthesize_2d_gabor_kernels_torch,
    synthesize_3d_gabor_kernels_torch,
)

from simplercolorvideovdp.evaluation import (
    CrossValidationSummary,
    FoldResult,
    compute_bootstrap_ci,
    compute_metrics,
    cross_validate_by_group,
    fit_calibration_curve,
    leave_one_distortion_out_cv,
    leave_one_scene_out_cv,
    power_law_jod,
)

__version__ = "0.1.0"

__all__ = [
    # Metrics
    "ColorVideoVDP",
    "HybridColorVideoVDP",
    "OrientedColorVideoVDP",
    "V1MTColorVideoVDP",
    "CanonicalV1MTModel",
    "CanonicalHybridColorVideoVDP",
    "cvvdp",
    # Display & Color
    "DisplayPhotometry",
    "DisplayGeometry",
    "DISPLAY_PRESETS",
    "load_display_model",
    "linear_rgb_to_dkl",
    "get_rgb_to_dkl_matrix",
    "rgb_to_dkl_torch",
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
    # Canonical V1-MT components
    "V1FilterLayerTorch",
    "V1NormalizationTorch",
    "MTIntegrationTorch",
    "MTNormalizationTorch",
    "SustainedFormModelTorch",
    "synthesize_3d_gabor_kernels_torch",
    "synthesize_2d_gabor_kernels_torch",
    "compute_velocity_plane_weights_torch",
    "pad_symmetric_5d",
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
    # Evaluation & Cross-Validation
    "power_law_jod",
    "fit_calibration_curve",
    "compute_metrics",
    "compute_bootstrap_ci",
    "cross_validate_by_group",
    "leave_one_scene_out_cv",
    "leave_one_distortion_out_cv",
    "FoldResult",
    "CrossValidationSummary",
]




