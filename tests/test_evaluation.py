"""Unit tests for evaluation and cross-validation routines."""

import numpy as np
import pandas as pd
import pytest

from simplercolorvideovdp.evaluation import (
    compute_metrics,
    cross_validate_by_group,
    fit_calibration_curve,
    leave_one_distortion_out_cv,
    leave_one_scene_out_cv,
    power_law_jod,
)


def test_power_law_and_fitting():
    q = np.linspace(1.0, 100.0, 50)
    true_a, true_b = 0.05, 0.9
    jod_clean = 10.0 - true_a * (q ** true_b)

    fitted_a, fitted_b = fit_calibration_curve(q, jod_clean)
    assert np.isclose(fitted_a, true_a, atol=1e-3)
    assert np.isclose(fitted_b, true_b, atol=1e-3)

    pred = power_law_jod(q, fitted_a, fitted_b)
    metrics = compute_metrics(jod_clean, pred)
    assert metrics["pearson_r"] > 0.999
    assert metrics["rmse"] < 0.05


def test_compute_metrics():
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([5.1, 5.9, 7.2, 7.8, 9.1, 9.9])

    m = compute_metrics(y_true, y_pred)
    assert m["pearson_r"] > 0.98
    assert m["spearman_rho"] > 0.98
    assert m["rmse"] < 0.2
    assert m["outlier_ratio"] == 0.0


def test_loso_and_lodo_cross_validation():
    # Construct synthetic dataset with 4 scenes and 3 distortions
    np.random.seed(42)
    scenes = ["attic", "classroom", "landscape", "subway"]
    distortions = ["judder", "noise", "dlss"]

    rows = []
    for s in scenes:
        for d in distortions:
            for lvl in [0, 1, 2]:
                raw_q = np.random.uniform(5.0, 120.0)
                jod_gt = 10.0 - 0.045 * (raw_q ** 0.92) + np.random.normal(0, 0.2)
                rows.append({
                    "scene_name": s,
                    "distortion_type": d,
                    "distortion_level": f"level{lvl}",
                    "raw_q": raw_q,
                    "jod_gt": jod_gt,
                })

    df = pd.DataFrame(rows)

    # Test Leave-One-Scene-Out (LOSO)
    loso_res = leave_one_scene_out_cv(df)
    assert loso_res.n_folds == 4
    assert len(loso_res.fold_results) == 4
    assert "pearson_r" in loso_res.oof_metrics
    assert loso_res.oof_metrics["pearson_r"] > 0.9
    assert "jod_oof_pred" in loso_res.oof_dataframe.columns
    assert len(loso_res.oof_dataframe) == len(df)

    # Test Leave-One-Distortion-Out (LODO)
    lodo_res = leave_one_distortion_out_cv(df)
    assert lodo_res.n_folds == 3
    assert len(lodo_res.fold_results) == 3
    assert lodo_res.oof_metrics["pearson_r"] > 0.9
