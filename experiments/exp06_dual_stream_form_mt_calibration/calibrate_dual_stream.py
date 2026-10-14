#!/usr/bin/env python3
"""
Experiment 06: Dual-Stream Cortical Architecture (Form + MT) Calibration Script.

Integrates:
- Ventral Form Stream: Steerable Weber Pyramid with calibrated beta_ori = 0.50
- Dorsal Motion Stream: Dedicated Achromatic Transient (5 Hz) MT stage

Optimizes stream-level Minkowski coupling parameters (w_m, beta_stream) and power-law (a, b)
on 7 training scenes (168 video pairs) and evaluates holdout generalization on 2 validation scenes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, pearsonr, spearmanr
import torch

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES
from simplercolorvideovdp.pooling import MetricPooling, lp_norm


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD power-law mapping."""
    return 10.0 - a * (np.maximum(q, 1e-8) ** b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Pearson r, Spearman rho, Kendall tau, RMSE, and MAE."""
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    outlier_ratio = float(np.mean(np.abs(y_true - y_pred) > 1.0))

    return {
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "kendall_tau": round(float(kt), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "outlier_ratio": round(outlier_ratio * 100, 2),
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


def pool_stream_minkowski(q_form: np.ndarray, q_motion: np.ndarray, beta_stream: float, w_m: float) -> np.ndarray:
    """Combines form and motion streams via stream Minkowski norm."""
    q_f = np.maximum(q_form, 1e-8)
    q_m = np.maximum(q_motion, 1e-8)
    if abs(beta_stream - 1.0) < 1e-4:
        return q_f + w_m * q_m
    else:
        return (q_f ** beta_stream + w_m * (q_m ** beta_stream)) ** (1.0 / beta_stream)


def main():
    parser = argparse.ArgumentParser(description="Calibrate Dual-Stream Form+MT on GAIM-240")
    parser.add_argument(
        "--cache_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../exp05_oriented_form_calibration/spatial_ori_cache.pt"),
    )
    parser.add_argument(
        "--csv_v1mt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../exp04_gaim240_model_calibration/predictions_v1mt_720p_full.csv"),
    )
    parser.add_argument(
        "--csv_base",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../exp04_gaim240_model_calibration/predictions_baseline_720p_full.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(__file__),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 110)
    print("EXPERIMENT 06: DUAL-STREAM CORTICAL ARCHITECTURE (FORM + MT) CALIBRATION")
    print("=" * 110)

    # 1. Load Form Stream cache and compute Q_form with calibrated beta_ori = 0.50
    print(f"Loading Form Stream spatial-orientation cache from: {args.cache_path}")
    cache = torch.load(args.cache_path, map_location="cpu")
    stats = cache["extracted_stats"]
    meta_form = cache["metadata"]

    pooling = MetricPooling()
    q_form_list = []
    for q_ori in stats:
        num_bands = len(q_ori)
        b, c, t = q_ori[0].shape[0], q_ori[0].shape[1], q_ori[0].shape[2]
        q_per_ch = torch.empty((b, c, t, num_bands), dtype=torch.float32)
        for bb in range(num_bands):
            b_tensor = q_ori[bb]
            k_dim = b_tensor.shape[-1]
            if k_dim > 1:
                q_band = lp_norm(b_tensor, 0.50, dim=-1, normalize=True, keepdim=False)
            else:
                q_band = b_tensor.squeeze(-1)
            q_per_ch[:, :, :, bb] = q_band
        raw_q = float(pooling.pool_raw_q(q_per_ch, is_image=False).squeeze().item())
        q_form_list.append(raw_q)

    q_form = np.array(q_form_list, dtype=np.float64)

    # 2. Load MT Motion Stream
    print(f"Loading MT Motion Stream from: {args.csv_v1mt}")
    df_v1mt = pd.read_csv(args.csv_v1mt)
    df_v1mt = df_v1mt[df_v1mt["distortion_level"] != "Ref"].reset_index(drop=True)
    q_motion = df_v1mt["raw_q_motion"].to_numpy(dtype=np.float64)
    j_gt = df_v1mt["jod_gt"].to_numpy(dtype=np.float64)

    # 3. Load Baseline for benchmarking
    df_base = pd.read_csv(args.csv_base)
    df_base = df_base[df_base["distortion_level"] != "Ref"].reset_index(drop=True)
    q_base = df_base["raw_q"].to_numpy(dtype=np.float64)

    scenes = df_v1mt["scene_name"].to_numpy()
    val_scenes = set(DEFAULT_VAL_SCENES)
    train_mask = ~np.isin(scenes, list(val_scenes))
    val_mask = np.isin(scenes, list(val_scenes))

    print(f"Dataset : {len(df_v1mt)} video pairs (Train: {train_mask.sum()}, Holdout Val: {val_mask.sum()})")

    # 4. Grid Sweep over (beta_stream, w_m)
    beta_stream_candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    wm_candidates = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]

    grid_results = []
    print("\nRunning 2D Grid Sweep over stream parameters...")

    for beta_s in beta_stream_candidates:
        for wm in wm_candidates:
            q_tot = pool_stream_minkowski(q_form, q_motion, beta_s, wm)

            q_tr, j_tr = q_tot[train_mask], j_gt[train_mask]
            q_val, j_val = q_tot[val_mask], j_gt[val_mask]

            a_fit, b_fit = fit_power_law(q_tr, j_tr)

            p_tr = power_law_jod(q_tr, a_fit, b_fit)
            p_val = power_law_jod(q_val, a_fit, b_fit)
            p_all = power_law_jod(q_tot, a_fit, b_fit)

            m_tr = compute_metrics(j_tr, p_tr)
            m_val = compute_metrics(j_val, p_val)
            m_all = compute_metrics(j_gt, p_all)

            grid_results.append({
                "beta_stream": beta_s,
                "w_m": wm,
                "calib_a": round(a_fit, 4),
                "calib_b": round(b_fit, 4),
                "train_r": m_tr["pearson_r"],
                "train_rho": m_tr["spearman_rho"],
                "train_rmse": m_tr["rmse"],
                "val_r": m_val["pearson_r"],
                "val_rho": m_val["spearman_rho"],
                "val_rmse": m_val["rmse"],
                "all_r": m_all["pearson_r"],
                "all_rho": m_all["spearman_rho"],
                "all_rmse": m_all["rmse"],
            })

    df_grid = pd.DataFrame(grid_results)
    grid_csv = os.path.join(args.output_dir, "dual_stream_grid_results.csv")
    df_grid.to_csv(grid_csv, index=False)
    print(f"Saved grid sweep results to: {grid_csv}")

    # Find best grid configuration on train loss / correlation
    best_grid_row = df_grid.sort_values(by=["train_r", "val_r"], ascending=[False, False]).iloc[0]
    print(f"\nBest Grid Config: beta_stream={best_grid_row['beta_stream']}, w_m={best_grid_row['w_m']} -> Train r={best_grid_row['train_r']}, Val r={best_grid_row['val_r']}, All r={best_grid_row['all_r']}")

    # 5. Continuous Joint Optimization on Training Split
    print("\nRunning continuous joint optimization for (w_m, beta_stream, a, b)...")
    def joint_loss(params):
        log_wm, beta_s, a, b = params
        wm = 10.0 ** log_wm
        q_tot = pool_stream_minkowski(q_form[train_mask], q_motion[train_mask], beta_s, wm)
        pred = power_law_jod(q_tot, a, b)
        return np.mean((j_gt[train_mask] - pred) ** 2)

    res_opt = minimize(
        joint_loss,
        x0=[np.log10(max(1.0, float(best_grid_row["w_m"]))), float(best_grid_row["beta_stream"]), float(best_grid_row["calib_a"]), float(best_grid_row["calib_b"])],
        bounds=[(0.0, 5.0), (0.2, 5.0), (1e-5, 10.0), (0.01, 3.0)],
        method="L-BFGS-B",
    )
    opt_wm = 10.0 ** float(res_opt.x[0])
    opt_beta_s = float(res_opt.x[1])
    opt_a = float(res_opt.x[2])
    opt_b = float(res_opt.x[3])

    q_tot_opt = pool_stream_minkowski(q_form, q_motion, opt_beta_s, opt_wm)
    p_tr_opt = power_law_jod(q_tot_opt[train_mask], opt_a, opt_b)
    p_val_opt = power_law_jod(q_tot_opt[val_mask], opt_a, opt_b)
    p_all_opt = power_law_jod(q_tot_opt, opt_a, opt_b)

    m_tr_opt = compute_metrics(j_gt[train_mask], p_tr_opt)
    m_val_opt = compute_metrics(j_gt[val_mask], p_val_opt)
    m_all_opt = compute_metrics(j_gt, p_all_opt)

    print(f"Optimal Continuous Parameters: w_m={opt_wm:.2f}, beta_stream={opt_beta_s:.3f}, a={opt_a:.4f}, b={opt_b:.4f}")
    print(f"Optimal Metrics: Train r={m_tr_opt['pearson_r']}, Val r={m_val_opt['pearson_r']}, All r={m_all_opt['pearson_r']}")

    # 6. Benchmark Comparison Table
    # Fit baseline
    a_b, b_b = fit_power_law(q_base[train_mask], j_gt[train_mask])
    p_base_all = power_law_jod(q_base, a_b, b_b)
    m_base_tr = compute_metrics(j_gt[train_mask], p_base_all[train_mask])
    m_base_val = compute_metrics(j_gt[val_mask], p_base_all[val_mask])
    m_base_all = compute_metrics(j_gt, p_base_all)

    # Fit Form only (beta_ori=0.50)
    a_f, b_f = fit_power_law(q_form[train_mask], j_gt[train_mask])
    p_form_all = power_law_jod(q_form, a_f, b_f)
    m_form_tr = compute_metrics(j_gt[train_mask], p_form_all[train_mask])
    m_form_val = compute_metrics(j_gt[val_mask], p_form_all[val_mask])
    m_form_all = compute_metrics(j_gt, p_form_all)

    summary_rows = [
        {
            "Model Configuration": "Baseline ColorVideoVDP (Isotropic)",
            "Parameters": f"a={a_b:.4f}, b={b_b:.4f}",
            "Train r": m_base_tr["pearson_r"],
            "Train rho": m_base_tr["spearman_rho"],
            "Train RMSE": m_base_tr["rmse"],
            "Val r": m_base_val["pearson_r"],
            "Val rho": m_base_val["spearman_rho"],
            "Val RMSE": m_base_val["rmse"],
            "All r": m_base_all["pearson_r"],
            "All rho": m_base_all["spearman_rho"],
            "All RMSE": m_base_all["rmse"],
        },
        {
            "Model Configuration": "Oriented Form Stream Only (beta_ori=0.50)",
            "Parameters": f"a={a_f:.4f}, b={b_f:.4f}",
            "Train r": m_form_tr["pearson_r"],
            "Train rho": m_form_tr["spearman_rho"],
            "Train RMSE": m_form_tr["rmse"],
            "Val r": m_form_val["pearson_r"],
            "Val rho": m_form_val["spearman_rho"],
            "Val RMSE": m_form_val["rmse"],
            "All r": m_form_all["pearson_r"],
            "All rho": m_form_all["spearman_rho"],
            "All RMSE": m_form_all["rmse"],
        },
        {
            "Model Configuration": f"Dual-Stream (Grid Best: beta={best_grid_row['beta_stream']}, w={best_grid_row['w_m']:.0f})",
            "Parameters": f"a={best_grid_row['calib_a']:.4f}, b={best_grid_row['calib_b']:.4f}",
            "Train r": best_grid_row["train_r"],
            "Train rho": best_grid_row["train_rho"],
            "Train RMSE": best_grid_row["train_rmse"],
            "Val r": best_grid_row["val_r"],
            "Val rho": best_grid_row["val_rho"],
            "Val RMSE": best_grid_row["val_rmse"],
            "All r": best_grid_row["all_r"],
            "All rho": best_grid_row["all_rho"],
            "All RMSE": best_grid_row["all_rmse"],
        },
        {
            "Model Configuration": f"Dual-Stream (Opt: beta={opt_beta_s:.2f}, w={opt_wm:.1f})",
            "Parameters": f"a={opt_a:.4f}, b={opt_b:.4f}",
            "Train r": m_tr_opt["pearson_r"],
            "Train rho": m_tr_opt["spearman_rho"],
            "Train RMSE": m_tr_opt["rmse"],
            "Val r": m_val_opt["pearson_r"],
            "Val rho": m_val_opt["spearman_rho"],
            "Val RMSE": m_val_opt["rmse"],
            "All r": m_all_opt["pearson_r"],
            "All rho": m_all_opt["spearman_rho"],
            "All RMSE": m_all_opt["rmse"],
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "dual_stream_calibration_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 110)
    print("GAIM-240 GLOBAL CALIBRATION COMPARISON TABLE")
    print("=" * 110)
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")

    # 7. Per-Distortion Breakdown
    distortions = sorted(df_v1mt["distortion_type"].unique())
    print("\n" + "=" * 105)
    print("PER-DISTORTION BREAKDOWN (Pearson Correlation r)")
    print("=" * 105)
    print(f"{'Distortion Type':<32} | {'Baseline':<12} | {'Form Only':<12} | {'Dual-Stream (Opt)':<18}")
    print("-" * 105)

    per_dist_rows = []
    for d in distortions:
        idx = (df_v1mt["distortion_type"] == d).to_numpy()
        r_b, _ = pearsonr(j_gt[idx], p_base_all[idx])
        r_f, _ = pearsonr(j_gt[idx], p_form_all[idx])
        r_opt, _ = pearsonr(j_gt[idx], p_all_opt[idx])
        print(f"{d:<32} | {r_b:>10.4f}   | {r_f:>10.4f}   | {r_opt:>16.4f}")
        per_dist_rows.append({
            "Distortion Type": d,
            "Baseline (Isotropic)": round(float(r_b), 4),
            "Form Only (beta=0.5)": round(float(r_f), 4),
            "Dual-Stream (Form+MT)": round(float(r_opt), 4),
        })

    print("=" * 105)
    df_per_dist = pd.DataFrame(per_dist_rows)
    dist_csv = os.path.join(args.output_dir, "per_distortion_comparison.csv")
    df_per_dist.to_csv(dist_csv, index=False)
    print(f"Saved per-distortion breakdown to: {dist_csv}")

    # 8. Plot 2D Contour / Heatmap of Train and Val Pearson r
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pivot_train = df_grid.pivot(index="beta_stream", columns="w_m", values="train_r")
    pivot_val = df_grid.pivot(index="beta_stream", columns="w_m", values="val_r")

    im0 = axes[0].imshow(pivot_train.values, origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_title(r"Train Pearson Correlation ($r_{\mathrm{train}}$)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"Motion Weight ($w_m$)", fontsize=11)
    axes[0].set_ylabel(r"Stream Minkowski Exponent ($\beta_{\mathrm{stream}}$)", fontsize=11)
    axes[0].set_xticks(range(len(pivot_train.columns)))
    axes[0].set_xticklabels([f"{c:g}" for c in pivot_train.columns], rotation=45)
    axes[0].set_yticks(range(len(pivot_train.index)))
    axes[0].set_yticklabels([f"{r:g}" for r in pivot_train.index])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pivot_val.values, origin="lower", cmap="plasma", aspect="auto")
    axes[1].set_title(r"Holdout Val Pearson Correlation ($r_{\mathrm{val}}$)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"Motion Weight ($w_m$)", fontsize=11)
    axes[1].set_ylabel(r"Stream Minkowski Exponent ($\beta_{\mathrm{stream}}$)", fontsize=11)
    axes[1].set_xticks(range(len(pivot_val.columns)))
    axes[1].set_xticklabels([f"{c:g}" for c in pivot_val.columns], rotation=45)
    axes[1].set_yticks(range(len(pivot_val.index)))
    axes[1].set_yticklabels([f"{r:g}" for r in pivot_val.index])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    heatmap_out = os.path.join(args.output_dir, "dual_stream_parameter_heatmap.png")
    fig.savefig(heatmap_out, dpi=150)
    plt.close(fig)
    print(f"Saved parameter heatmap to: {heatmap_out}")

    # 9. Plot Scatter Comparison: Baseline vs Dual-Stream
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Train scatter
    axes[0].scatter(j_gt[train_mask], p_base_all[train_mask], alpha=0.6, color="tab:blue", label=f"Baseline (r={m_base_tr['pearson_r']:.3f})")
    axes[0].scatter(j_gt[train_mask], p_all_opt[train_mask], alpha=0.6, color="tab:orange", label=f"Dual-Stream (r={m_tr_opt['pearson_r']:.3f})")
    axes[0].plot([6, 10], [6, 10], "k--", alpha=0.7, label="Ideal")
    axes[0].set_title(f"Training Scenes (N={train_mask.sum()})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Ground Truth JOD", fontsize=11)
    axes[0].set_ylabel("Predicted JOD", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Val scatter
    axes[1].scatter(j_gt[val_mask], p_base_all[val_mask], alpha=0.7, color="tab:blue", marker="s", label=f"Baseline (r={m_base_val['pearson_r']:.3f})")
    axes[1].scatter(j_gt[val_mask], p_all_opt[val_mask], alpha=0.7, color="tab:orange", marker="^", label=f"Dual-Stream (r={m_val_opt['pearson_r']:.3f})")
    axes[1].plot([6, 10], [6, 10], "k--", alpha=0.7, label="Ideal")
    axes[1].set_title(f"Holdout Validation Scenes: subway & zeroday (N={val_mask.sum()})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ground Truth JOD", fontsize=11)
    axes[1].set_ylabel("Predicted JOD", fontsize=11)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    scatter_out = os.path.join(args.output_dir, "dual_stream_scatter_comparison.png")
    fig.savefig(scatter_out, dpi=150)
    plt.close(fig)
    print(f"Saved scatter comparison plot to: {scatter_out}")


if __name__ == "__main__":
    main()
