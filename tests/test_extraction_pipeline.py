"""Unit tests for model prediction extraction pipeline."""

import os
import tempfile
import torch
import pandas as pd
import pytest

from simplercolorvideovdp.metric import ColorVideoVDP, OrientedColorVideoVDP, V1MTColorVideoVDP
from experiments.exp04_gaim240_model_calibration.extract_model_predictions import (
    compute_model_quality_in_chunks,
    create_model,
)


@pytest.mark.parametrize("model_name", ["baseline", "oriented", "v1mt"])
def test_compute_model_quality_in_chunks_smoke(model_name: str):
    """Verifies that chunked prediction extraction runs end-to-end for all models."""
    device = torch.device("cpu")
    model = create_model(model_name, fps=30.0, device=device)

    # 10 frames @ 64x64 resolution, chunk_size=4
    t_len, h, w = 10, 64, 64
    ref_tensor = torch.rand((1, 3, t_len, h, w), dtype=torch.float32)
    test_tensor = (ref_tensor + 0.05 * torch.randn_like(ref_tensor)).clamp(0, 1)

    raw_q_total, raw_q_form, raw_q_motion, default_jod = compute_model_quality_in_chunks(
        model=model,
        test_tensor=test_tensor,
        ref_tensor=ref_tensor,
        fps=30.0,
        chunk_size=4,
        device=device,
    )

    assert isinstance(raw_q_total, float) and raw_q_total > 0.0
    assert isinstance(raw_q_form, float) and raw_q_form > 0.0
    assert isinstance(raw_q_motion, float) and raw_q_motion >= 0.0
    assert isinstance(default_jod, float) and (0.0 <= default_jod <= 10.0)

    if model_name in ("baseline", "oriented"):
        assert raw_q_motion == 0.0
        assert raw_q_total == raw_q_form
    elif model_name == "v1mt":
        assert raw_q_motion > 0.0
