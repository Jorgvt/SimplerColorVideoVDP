#!/usr/bin/env python3
"""
Experiment 04: Global Correlation & Cross-Validation Evaluation Script.

Evaluates and compares the global performance of:
1. Baseline ColorVideoVDP (Isotropic Laplacian Weber Pyramid)
2. Oriented ColorVideoVDP (Steerable Weber Pyramid, K=4, Form Only)
3. V1-MT Dedicated Dorsal Stream (Achromatic Transient MT)
4. Stream-Level Minkowski Combinations (Form + Dedicated MT at various weights)

Fits non-linear least squares power-law calibration (JOD = 10 - a·Qᵇ)
on 7 training scenes (168 video pairs) and evaluates generalization
on 2 unseen holdout validation scenes (subway, zeroday, 48 video pairs).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD power-law mapping."""
    return 10.0 - a * (np.maximum(q, 1e-8) ** b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Pearson r, Spearman rho, and RMSE."""
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def fit_power_law(
    q_train: np.ndarray,
    jod_train: np.ndarray,
    p0: Tuple[float, float] = (0.13, 0.66),
) -> Tuple[float, float]:
    """Fits (a, b) parameters on training split via least-squares loss."""
    def loss_fn(p):
        pred = power_law_jod(q_train, p[0], p[1])
        return np.mean((jod_train - pred) ** 2)

    res = minimize(
        loss_fn,
        p0,
        bounds=[(1e-5, 10.0), (0.01, 3.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Global Model Correlations on GAIM-240")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "global_correlation_comparison.csv"),
        help="Path to save the summary comparison CSV.",
    )
    args = parser.parse_args()

    dir_path = os.path.dirname(__file__)
    csv_base = os.path.join(dir_path, "predictions_baseline_720p_full.csv")
    csv_ori = os.path.join(dir_path, "predictions_oriented_720p_full.csv")
    csv_v1mt = os.path.join(dir_path, "predictions_v1mt_720p_full.csv")
    csv_v1mt_old = os.path.join(dir_path, "predictions_v1mt_720p_full_old_linear.csv")

    df_base = pd.read_csv(csv_base)
    df_ori = pd.read_csv(csv_ori)
    df_v1mt = pd.read_csv(csv_v1mt)

    # Exclude reference comparisons if present
    df_base = df_base[df_base["distortion_level"] != "Ref"].reset_index(drop=True)
    df_ori = df_ori[df_ori["distortion_level"] != "Ref"].reset_index(drop=True)
    df_v1mt = df_v1mt[df_v1mt["distortion_level"] != "Ref"].reset_index(drop=True)

    train_mask = ~df_ori["scene_name"].isin(DEFAULT_VAL_SCENES)
    val_mask = df_ori["scene_name"].isin(DEFAULT_VAL_SCENES)

    j_gt = df_ori["jod_gt"].to_numpy(dtype=np.float64)
    q_base = df_base["raw_q"].to_numpy(dtype=np.float64)
    q_ori = df_ori["raw_q"].to_numpy(dtype=np.float64)
    q_v1mt = df_v1mt["raw_q"].to_numpy(dtype=np.float64)
    q_motion = df_v1mt["raw_q_motion"].to_numpy(dtype=np.float64)

    # Define model configurations to benchmark
    model_configs = [
        ("Baseline ColorVideoVDP (Isotropic)", q_base),
        ("Oriented ColorVideoVDP (Form Only)", q_ori),
        ("V1-MT (Dedicated Dorsal Native)", q_v1mt),
        ("Form + Transient MT (β=2, w=100)", (q_ori ** 2.0 + 100.0 * (q_motion ** 2.0)) ** 0.5),
        ("Form + Transient MT (β=2, w=500)", (q_ori ** 2.0 + 500.0 * (q_motion ** 2.0)) ** 0.5),
        ("Form + Transient MT (β=2, w=1000)", (q_ori ** 2.0 + 1000.0 * (q_motion ** 2.0)) ** 0.5),
        ("Form + Transient MT (β=2, w=5000)", (q_ori ** 2.0 + 5000.0 * (q_motion ** 2.0)) ** 0.5),
        ("Form + Transient MT (β=1, w=50)", q_ori + 50.0 * q_motion),
    ]

    if os.path.exists(csv_v1mt_old):
        df_old = pd.read_csv(csv_v1mt_old)
        df_old = df_old[df_old["distortion_level"] != "Ref"].reset_index(drop=True)
        model_configs.insert(2, ("V1-MT (Old Linear Contaminated)", df_old["raw_q"].to_numpy(dtype=np.float64)))

    summary_rows = []
    calibrated_predictions = {}

    print("=" * 110)
    print("GAIM-240 MODEL GLOBAL CORRELATION COMPARISON")
    print(f"Dataset : 216 video pairs (Train: {train_mask.sum()}, Holdout Val: {val_mask.sum()})")
    print("=" * 110)

    for name, q_arr in model_configs:
        q_tr = q_arr[train_mask]
        j_tr = j_gt[train_mask]

        q_val = q_arr[val_mask]
        j_val = j_gt[val_mask]

        # Fit calibration on train split
        a_fit, b_fit = fit_power_law(q_tr, j_tr)

        p_tr = power_law_jod(q_tr, a_fit, b_fit)
        p_val = power_law_jod(q_val, a_fit, b_fit)
        p_all = power_law_jod(q_arr, a_fit, b_fit)

        calibrated_predictions[name] = p_all

        m_tr = compute_metrics(j_tr, p_tr)
        m_val = compute_metrics(j_val, p_val)
        m_all = compute_metrics(j_gt, p_all)

        summary_rows.append({
            "Model Configuration": name,
            "Calib (a, b)": f"({a_fit:.4f}, {b_fit:.4f})",
            "Train Pearson r": m_tr["pearson_r"],
            "Train Spearman rho": m_tr["spearman_rho"],
            "Train RMSE": m_tr["rmse"],
            "Val Pearson r": m_val["pearson_r"],
            "Val Spearman rho": m_val["spearman_rho"],
            "Val RMSE": m_val["rmse"],
            "All Pearson r": m_all["pearson_r"],
            "All Spearman rho": m_all["spearman_rho"],
            "All RMSE": m_all["rmse"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.output_csv, index=False)

    print("\n" + df_summary.to_string(index=False))
    print(f"\nSaved summary comparison to: {args.output_csv}")

    # Per-distortion breakdown
    distortions = sorted(df_ori["distortion_type"].unique())
    print("\n" + "=" * 95)
    print("PER-DISTORTION BREAKDOWN (Pearson Correlation r)")
    print("=" * 95)
    print(f"{'Distortion Type':<32} | {'Baseline':<12} | {'Oriented':<12} | {'V1-MT (Native)':<14} | {'Form+MT (w=1k)':<14}")
    print("-" * 95)

    p_base = calibrated_predictions["Baseline ColorVideoVDP (Isotropic)"]
    p_ori = calibrated_predictions["Oriented ColorVideoVDP (Form Only)"]
    p_v1mt = calibrated_predictions["V1-MT (Dedicated Dorsal Native)"]
    p_comb = calibrated_predictions["Form + Transient MT (β=2, w=1000)"]

    for d in distortions:
        idx = df_ori["distortion_type"] == d
        r_b, _ = pearsonr(j_gt[idx], p_base[idx])
        r_o, _ = pearsonr(j_gt[idx], p_ori[idx])
        r_v, _ = pearsonr(j_gt[idx], p_v1mt[idx])
        r_c, _ = pearsonr(j_gt[idx], p_comb[idx])
        print(f"{d:<32} | {r_b:>10.4f}   | {r_o:>10.4f}   | {r_v:>12.4f}   | {r_c:>12.4f}")

    print("=" * 95)


if __name__ == "__main__":
    main()
