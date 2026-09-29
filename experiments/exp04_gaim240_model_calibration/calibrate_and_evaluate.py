#!/usr/bin/env python3
"""
Experiment 04: Calibrate Model Predictions to GAIM-240 and Evaluate.

Fits non-linear calibration mapping (JOD = 10 - a·Qᵇ) on training scenes
and evaluates holdout generalization performance across models on GAIM-240.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD conversion power law."""
    q_safe = np.maximum(q, 1e-8)
    return 10.0 - a * (q_safe ** b)


def fit_calibration_curve(
    q_train: np.ndarray,
    jod_train: np.ndarray,
    p0: Tuple[float, float] = (0.04395, 0.9302),
) -> Tuple[float, float]:
    """Fits parameters (a, b) using non-linear least squares on training scenes."""
    try:
        popt, _ = curve_fit(
            power_law_jod,
            q_train,
            jod_train,
            p0=p0,
            bounds=([1e-6, 1e-3], [10.0, 5.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"Warning: Curve fitting failed ({e}). Using default initial parameters.")
        return p0[0], p0[1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive visual quality assessment evaluation metrics."""
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    outlier_ratio = float(np.mean(np.abs(y_true - y_pred) > 1.0))  # Percentage of errors > 1 JOD

    return {
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "kendall_tau": round(float(kt), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "outlier_ratio": round(outlier_ratio * 100, 2),
    }


def evaluate_single_model_csv(
    csv_path: str,
    val_scenes: List[str] = DEFAULT_VAL_SCENES,
) -> Dict[str, Any]:
    """Calibrates and evaluates a single model predictions CSV file."""
    df = pd.read_csv(csv_path)

    # Exclude identical reference comparison if present
    if "distortion_level" in df.columns:
        df = df[df["distortion_level"] != "Ref"].reset_index(drop=True)

    # Split train and validation scenes
    train_mask = ~df["scene_name"].isin(val_scenes)
    val_mask = df["scene_name"].isin(val_scenes)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    q_train = df_train["raw_q"].to_numpy(dtype=np.float64)
    jod_train = df_train["jod_gt"].to_numpy(dtype=np.float64)

    q_val = df_val["raw_q"].to_numpy(dtype=np.float64)
    jod_val = df_val["jod_gt"].to_numpy(dtype=np.float64)

    q_all = df["raw_q"].to_numpy(dtype=np.float64)
    jod_all = df["jod_gt"].to_numpy(dtype=np.float64)

    # 1. Fit calibration parameters on train set
    calib_a, calib_b = fit_calibration_curve(q_train, jod_train)

    # 2. Apply calibrated predictions
    df_train["jod_calibrated"] = power_law_jod(q_train, calib_a, calib_b)
    df_val["jod_calibrated"] = power_law_jod(q_val, calib_a, calib_b)
    df["jod_calibrated"] = power_law_jod(q_all, calib_a, calib_b)

    # 3. Compute metrics for uncalibrated default and calibrated JOD
    metrics_train_uncalib = compute_metrics(jod_train, df_train["default_jod"].to_numpy())
    metrics_train_calib = compute_metrics(jod_train, df_train["jod_calibrated"].to_numpy())

    metrics_val_uncalib = compute_metrics(jod_val, df_val["default_jod"].to_numpy())
    metrics_val_calib = compute_metrics(jod_val, df_val["jod_calibrated"].to_numpy())

    metrics_all_uncalib = compute_metrics(jod_all, df["default_jod"].to_numpy())
    metrics_all_calib = compute_metrics(jod_all, df["jod_calibrated"].to_numpy())

    # 4. Per-distortion breakdown on full dataset
    distortion_breakdown = {}
    for dist_type, group in df.groupby("distortion_type"):
        if len(group) > 2:
            metrics_dist = compute_metrics(
                group["jod_gt"].to_numpy(),
                group["jod_calibrated"].to_numpy(),
            )
            distortion_breakdown[dist_type] = metrics_dist

    return {
        "csv_path": csv_path,
        "calib_params": {"a": round(calib_a, 6), "b": round(calib_b, 6)},
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "total_samples": len(df),
        "metrics_train_calib": metrics_train_calib,
        "metrics_train_uncalib": metrics_train_uncalib,
        "metrics_val_calib": metrics_val_calib,
        "metrics_val_uncalib": metrics_val_uncalib,
        "metrics_all_calib": metrics_all_calib,
        "metrics_all_uncalib": metrics_all_uncalib,
        "distortion_breakdown": distortion_breakdown,
        "df_with_calib": df,
    }


def plot_model_comparisons(
    results_list: List[Dict[str, Any]],
    output_dir: str,
):
    """Generates visual comparison figures for evaluated models."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 5.5), sharey=True)
    if len(results_list) == 1:
        axes = [axes]

    colors = {"train": "#2b5c8f", "val": "#d95f02"}

    for ax, res in zip(axes, results_list):
        model_title = os.path.basename(res["csv_path"]).replace("predictions_", "").replace(".csv", "")
        df = res["df_with_calib"]

        val_mask = df["scene_name"].isin(DEFAULT_VAL_SCENES)
        train_mask = ~val_mask

        # Scatter train points
        ax.scatter(
            df.loc[train_mask, "jod_calibrated"],
            df.loc[train_mask, "jod_gt"],
            alpha=0.6,
            c=colors["train"],
            label=f"Train Scenes (N={train_mask.sum()})",
            edgecolors="none",
            s=40,
        )

        # Scatter val points
        ax.scatter(
            df.loc[val_mask, "jod_calibrated"],
            df.loc[val_mask, "jod_gt"],
            alpha=0.8,
            c=colors["val"],
            label=f"Val Scenes (N={val_mask.sum()})",
            edgecolors="k",
            s=55,
            marker="^",
        )

        # Diagonal identity line
        ax.plot([4, 10], [4, 10], "k--", alpha=0.5, label="Perfect agreement")

        val_m = res["metrics_val_calib"]
        all_m = res["metrics_all_calib"]
        ax.set_title(
            f"{model_title.upper()}\nVal Pearson r: {val_m['pearson_r']:.3f} | RMSE: {val_m['rmse']:.2f}\nAll Pearson r: {all_m['pearson_r']:.3f} | RMSE: {all_m['rmse']:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("Calibrated Predicted JOD", fontsize=11)
        ax.set_ylabel("Ground Truth JOD (GAIM-240)", fontsize=11)
        ax.set_xlim([3.5, 10.5])
        ax.set_ylim([3.5, 10.5])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_calibration_scatter_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved scatter comparison plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate and Evaluate Models on GAIM-240")
    parser.add_argument(
        "--csv_files",
        nargs="+",
        required=True,
        help="List of prediction CSV files to calibrate and evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory to save comparison metrics and plots.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("GAIM-240 Model Calibration & Cross-Validation Evaluation")
    print(f"Input CSV files : {len(args.csv_files)}")
    print(f"Output dir      : {args.output_dir}")
    print("=" * 80)

    all_results = []
    summary_rows = []

    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}")
            continue

        print(f"\nProcessing: {csv_file}")
        res = evaluate_single_model_csv(csv_file)
        all_results.append(res)

        model_name = os.path.basename(csv_file).replace("predictions_", "").replace(".csv", "")
        calib_a = res["calib_params"]["a"]
        calib_b = res["calib_params"]["b"]
        val_m = res["metrics_val_calib"]
        train_m = res["metrics_train_calib"]
        all_m = res["metrics_all_calib"]

        summary_rows.append({
            "Model": model_name,
            "Calib (a, b)": f"({calib_a:.4f}, {calib_b:.4f})",
            "Val Pearson r": val_m["pearson_r"],
            "Val Spearman rho": val_m["spearman_rho"],
            "Val RMSE": val_m["rmse"],
            "Train Pearson r": train_m["pearson_r"],
            "Train Spearman rho": train_m["spearman_rho"],
            "Train RMSE": train_m["rmse"],
            "All Pearson r": all_m["pearson_r"],
            "All Spearman rho": all_m["spearman_rho"],
            "All RMSE": all_m["rmse"],
        })

        print(f"  Fitted Parameters: a = {calib_a:.6f}, b = {calib_b:.6f}")
        print(f"  Validation (subway, zeroday) -> Pearson r: {val_m['pearson_r']:.4f} | Spearman rho: {val_m['spearman_rho']:.4f} | RMSE: {val_m['rmse']:.4f}")
        print(f"  Training (7 scenes)          -> Pearson r: {train_m['pearson_r']:.4f} | Spearman rho: {train_m['spearman_rho']:.4f} | RMSE: {train_m['rmse']:.4f}")
        print(f"  Overall (All 9 scenes)       -> Pearson r: {all_m['pearson_r']:.4f} | Spearman rho: {all_m['spearman_rho']:.4f} | RMSE: {all_m['rmse']:.4f}")

    if all_results:
        # Save summary CSV
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "calibration_summary_metrics.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSaved summary table to {summary_csv}")

        # Plot comparisons
        plot_model_comparisons(all_results, args.output_dir)


if __name__ == "__main__":
    main()
