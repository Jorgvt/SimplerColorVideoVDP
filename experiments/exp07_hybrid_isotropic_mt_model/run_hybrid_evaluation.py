#!/usr/bin/env python3
"""
Experiment 07: Hybrid Isotropic Backbone + Parallel Steerable MT Model Evaluation.

Integrates:
- Path A: Calibrated Isotropic Laplacian Weber Pyramid (Form Backbone, Q_iso)
- Path B: Dedicated Achromatic Transient (5.0 Hz) Steerable MT Stream (Motion Detector, Q_MT)

Coupling:
    Q_total = (Q_iso^beta + w_m * Q_MT^beta)^(1/beta)
    JOD = 10 - a * (Q_total^b)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, pearsonr, spearmanr

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD power-law mapping."""
    return 10.0 - a * (np.maximum(q, 1e-8) ** b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes visual quality assessment metrics."""
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
    p0: Tuple[float, float] = (0.18, 0.54),
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


def pool_hybrid(q_iso: np.ndarray, q_motion: np.ndarray, beta: float, w_m: float) -> np.ndarray:
    """Combines isotropic form and MT motion streams via Minkowski norm."""
    q_i = np.maximum(q_iso, 1e-8)
    q_m = np.maximum(q_motion, 1e-8)
    if abs(beta - 1.0) < 1e-4:
        return q_i + w_m * q_m
    else:
        return (q_i ** beta + w_m * (q_m ** beta)) ** (1.0 / beta)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Isotropic + Parallel MT Model")
    parser.add_argument(
        "--csv_base",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../exp04_gaim240_model_calibration/predictions_baseline_720p_full.csv"),
    )
    parser.add_argument(
        "--csv_v1mt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../exp04_gaim240_model_calibration/predictions_v1mt_720p_full.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(__file__),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 110)
    print("EXPERIMENT 07: HYBRID ISOTROPIC BACKBONE + PARALLEL STEERABLE MT MODEL")
    print("=" * 110)

    # 1. Load Data
    df_base = pd.read_csv(args.csv_base)
    df_v1mt = pd.read_csv(args.csv_v1mt)

    df_base = df_base[df_base["distortion_level"] != "Ref"].reset_index(drop=True)
    df_v1mt = df_v1mt[df_v1mt["distortion_level"] != "Ref"].reset_index(drop=True)

    q_iso = df_base["raw_q"].to_numpy(dtype=np.float64)
    q_motion = df_v1mt["raw_q_motion"].to_numpy(dtype=np.float64)
    j_gt = df_base["jod_gt"].to_numpy(dtype=np.float64)

    val_scenes = set(DEFAULT_VAL_SCENES)
    train_mask = ~df_base["scene_name"].isin(val_scenes).to_numpy()
    val_mask = df_base["scene_name"].isin(val_scenes).to_numpy()

    print(f"Total video pairs: {len(df_base)} (Train: {train_mask.sum()}, Holdout Val: {val_mask.sum()})")

    # 2. Parameter Sweep over (w_m, beta)
    wm_list = [0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0]
    beta_list = [1.0, 1.5, 2.0, 3.0]

    sweep_rows = []
    for beta in beta_list:
        for wm in wm_list:
            q_tot = pool_hybrid(q_iso, q_motion, beta, wm)

            q_tr, j_tr = q_tot[train_mask], j_gt[train_mask]
            q_val, j_val = q_tot[val_mask], j_gt[val_mask]

            a_fit, b_fit = fit_power_law(q_tr, j_tr)

            p_tr = power_law_jod(q_tr, a_fit, b_fit)
            p_val = power_law_jod(q_val, a_fit, b_fit)
            p_all = power_law_jod(q_tot, a_fit, b_fit)

            m_tr = compute_metrics(j_tr, p_tr)
            m_val = compute_metrics(j_val, p_val)
            m_all = compute_metrics(j_gt, p_all)

            sweep_rows.append({
                "beta_stream": beta,
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

    df_sweep = pd.DataFrame(sweep_rows)
    sweep_csv = os.path.join(args.output_dir, "hybrid_model_sweep_results.csv")
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"\nSaved sweep results to: {sweep_csv}")

    # 3. Model Benchmark Comparison
    # Baseline
    a_b, b_b = fit_power_law(q_iso[train_mask], j_gt[train_mask])
    p_base = power_law_jod(q_iso, a_b, b_b)
    m_b_tr = compute_metrics(j_gt[train_mask], p_base[train_mask])
    m_b_val = compute_metrics(j_gt[val_mask], p_base[val_mask])
    m_b_all = compute_metrics(j_gt, p_base)

    # Hybrid Candidate 1: beta=2.0, w_m=500
    q_h500 = pool_hybrid(q_iso, q_motion, 2.0, 500.0)
    a_h500, b_h500 = fit_power_law(q_h500[train_mask], j_gt[train_mask])
    p_h500 = power_law_jod(q_h500, a_h500, b_h500)
    m_500_tr = compute_metrics(j_gt[train_mask], p_h500[train_mask])
    m_500_val = compute_metrics(j_gt[val_mask], p_h500[val_mask])
    m_500_all = compute_metrics(j_gt, p_h500)

    # Hybrid Candidate 2: beta=2.0, w_m=1000
    q_h1000 = pool_hybrid(q_iso, q_motion, 2.0, 1000.0)
    a_h1000, b_h1000 = fit_power_law(q_h1000[train_mask], j_gt[train_mask])
    p_h1000 = power_law_jod(q_h1000, a_h1000, b_h1000)
    m_1000_tr = compute_metrics(j_gt[train_mask], p_h1000[train_mask])
    m_1000_val = compute_metrics(j_gt[val_mask], p_h1000[val_mask])
    m_1000_all = compute_metrics(j_gt, p_h1000)

    # Hybrid Candidate 3: beta=2.0, w_m=2000 (Maximum Val Generalization)
    q_h2000 = pool_hybrid(q_iso, q_motion, 2.0, 2000.0)
    a_h2000, b_h2000 = fit_power_law(q_h2000[train_mask], j_gt[train_mask])
    p_h2000 = power_law_jod(q_h2000, a_h2000, b_h2000)
    m_2000_tr = compute_metrics(j_gt[train_mask], p_h2000[train_mask])
    m_2000_val = compute_metrics(j_gt[val_mask], p_h2000[val_mask])
    m_2000_all = compute_metrics(j_gt, p_h2000)

    summary_rows = [
        {
            "Model Architecture": "Baseline ColorVideoVDP (Isotropic Laplacian)",
            "Parameters": f"a={a_b:.4f}, b={b_b:.4f}",
            "Train r": m_b_tr["pearson_r"],
            "Train rho": m_b_tr["spearman_rho"],
            "Train RMSE": m_b_tr["rmse"],
            "Val r": m_b_val["pearson_r"],
            "Val rho": m_b_val["spearman_rho"],
            "Val RMSE": m_b_val["rmse"],
            "All r": m_b_all["pearson_r"],
            "All rho": m_b_all["spearman_rho"],
            "All RMSE": m_b_all["rmse"],
        },
        {
            "Model Architecture": "Hybrid Model (Isotropic + MT, w=500)",
            "Parameters": f"a={a_h500:.4f}, b={b_h500:.4f}",
            "Train r": m_500_tr["pearson_r"],
            "Train rho": m_500_tr["spearman_rho"],
            "Train RMSE": m_500_tr["rmse"],
            "Val r": m_500_val["pearson_r"],
            "Val rho": m_500_val["spearman_rho"],
            "Val RMSE": m_500_val["rmse"],
            "All r": m_500_all["pearson_r"],
            "All rho": m_500_all["spearman_rho"],
            "All RMSE": m_500_all["rmse"],
        },
        {
            "Model Architecture": "Hybrid Model (Isotropic + MT, w=1000)",
            "Parameters": f"a={a_h1000:.4f}, b={b_h1000:.4f}",
            "Train r": m_1000_tr["pearson_r"],
            "Train rho": m_1000_tr["spearman_rho"],
            "Train RMSE": m_1000_tr["rmse"],
            "Val r": m_1000_val["pearson_r"],
            "Val rho": m_1000_val["spearman_rho"],
            "Val RMSE": m_1000_val["rmse"],
            "All r": m_1000_all["pearson_r"],
            "All rho": m_1000_all["spearman_rho"],
            "All RMSE": m_1000_all["rmse"],
        },
        {
            "Model Architecture": "Hybrid Model (Isotropic + MT, w=2000)",
            "Parameters": f"a={a_h2000:.4f}, b={b_h2000:.4f}",
            "Train r": m_2000_tr["pearson_r"],
            "Train rho": m_2000_tr["spearman_rho"],
            "Train RMSE": m_2000_tr["rmse"],
            "Val r": m_2000_val["pearson_r"],
            "Val rho": m_2000_val["spearman_rho"],
            "Val RMSE": m_2000_val["rmse"],
            "All r": m_2000_all["pearson_r"],
            "All rho": m_2000_all["spearman_rho"],
            "All RMSE": m_2000_all["rmse"],
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "hybrid_vs_baseline_comparison.csv")
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 110)
    print("GAIM-240 BENCHMARK: HYBRID ISOTROPIC+MT vs BASELINE COLORVIDEOVDP")
    print("=" * 110)
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary comparison to: {summary_csv}")

    # 4. Per-Distortion Breakdown
    distortions = sorted(df_base["distortion_type"].unique())
    print("\n" + "=" * 110)
    print("PER-DISTORTION BREAKDOWN (Pearson Correlation r)")
    print("=" * 110)
    print(f"{'Distortion Type':<32} | {'Baseline':<12} | {'Hybrid (w=500)':<16} | {'Hybrid (w=1000)':<16} | {'Hybrid (w=2000)':<16}")
    print("-" * 110)

    per_dist_rows = []
    for d in distortions:
        idx = (df_base["distortion_type"] == d).to_numpy()
        rb, _ = pearsonr(j_gt[idx], p_base[idx])
        r500, _ = pearsonr(j_gt[idx], p_h500[idx])
        r1000, _ = pearsonr(j_gt[idx], p_h1000[idx])
        r2000, _ = pearsonr(j_gt[idx], p_h2000[idx])
        print(f"{d:<32} | {rb:>10.4f}   | {r500:>14.4f}   | {r1000:>14.4f}   | {r2000:>14.4f}")
        per_dist_rows.append({
            "Distortion Type": d,
            "Baseline (Isotropic)": round(float(rb), 4),
            "Hybrid (w=500)": round(float(r500), 4),
            "Hybrid (w=1000)": round(float(r1000), 4),
            "Hybrid (w=2000)": round(float(r2000), 4),
        })

    print("=" * 110)
    df_per_dist = pd.DataFrame(per_dist_rows)
    dist_csv = os.path.join(args.output_dir, "per_distortion_comparison.csv")
    df_per_dist.to_csv(dist_csv, index=False)
    print(f"Saved per-distortion breakdown to: {dist_csv}")

    # 5. Plot Comparison Figures
    # Sweep curve plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    df_b2 = df_sweep[df_sweep["beta_stream"] == 2.0]
    ax1.plot(df_b2["w_m"], df_b2["train_r"], "b-o", label=r"Train $r$")
    ax1.plot(df_b2["w_m"], df_b2["val_r"], "r-s", label=r"Holdout Val $r$ (`subway`, `zeroday`)")
    ax1.plot(df_b2["w_m"], df_b2["all_r"], "k--^", label=r"Global All $r$")
    ax1.axhline(m_b_val["pearson_r"], color="r", linestyle=":", alpha=0.7, label=f"Baseline Val $r$ ({m_b_val['pearson_r']:.3f})")
    ax1.axhline(m_b_all["pearson_r"], color="k", linestyle=":", alpha=0.7, label=f"Baseline All $r$ ({m_b_all['pearson_r']:.3f})")
    ax1.set_xscale("symlog", linthresh=25.0)
    ax1.set_xlabel(r"Motion Stream Coupling Weight ($w_m$)", fontsize=12)
    ax1.set_ylabel("Pearson Correlation ($r_p$)", fontsize=12)
    ax1.set_title(r"Hybrid Model: Generalization & Correlation vs Motion Weight ($w_m$)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plot_out = os.path.join(args.output_dir, "hybrid_model_performance_plot.png")
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"Saved performance plot to: {plot_out}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(j_gt[train_mask], p_base[train_mask], alpha=0.6, color="tab:blue", label=f"Baseline (r={m_b_tr['pearson_r']:.3f})")
    axes[0].scatter(j_gt[train_mask], p_h1000[train_mask], alpha=0.6, color="tab:orange", label=f"Hybrid w=1k (r={m_1000_tr['pearson_r']:.3f})")
    axes[0].plot([6, 10], [6, 10], "k--", alpha=0.7, label="Ideal")
    axes[0].set_title("Training Split (7 scenes, 168 video pairs)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Ground Truth JOD", fontsize=11)
    axes[0].set_ylabel("Predicted JOD", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].scatter(j_gt[val_mask], p_base[val_mask], alpha=0.7, color="tab:blue", marker="s", label=f"Baseline (r={m_b_val['pearson_r']:.3f})")
    axes[1].scatter(j_gt[val_mask], p_h1000[val_mask], alpha=0.7, color="tab:orange", marker="^", label=f"Hybrid w=1k (r={m_1000_val['pearson_r']:.3f})")
    axes[1].plot([6, 10], [6, 10], "k--", alpha=0.7, label="Ideal")
    axes[1].set_title("Holdout Validation Split: subway & zeroday (48 pairs)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ground Truth JOD", fontsize=11)
    axes[1].set_ylabel("Predicted JOD", fontsize=11)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    scatter_out = os.path.join(args.output_dir, "scatter_comparison.png")
    fig.savefig(scatter_out, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to: {scatter_out}")


if __name__ == "__main__":
    main()
