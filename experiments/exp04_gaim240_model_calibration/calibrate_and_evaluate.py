#!/usr/bin/env python3
"""
Experiment 04: Calibrate Model Predictions to GAIM-240 and Evaluate.

Fits non-linear calibration mapping (JOD = 10 - a·Qᵇ) and evaluates:
1. Fixed Holdout split (7 train scenes, 2 validation scenes).
2. Leave-One-Scene-Out (LOSO) 9-fold cross-validation (Content Generalization: Mean ± Std across scenes).
3. Leave-One-Distortion-Out (LODO) 8-fold cross-validation (Distortion Generalization: Mean ± Std across distortions).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES
from simplercolorvideovdp.evaluation import (
    CrossValidationSummary,
    compute_metrics,
    fit_calibration_curve,
    leave_one_distortion_out_cv,
    leave_one_scene_out_cv,
    power_law_jod,
)


def evaluate_single_model_csv(
    csv_path: str,
    val_scenes: List[str] = DEFAULT_VAL_SCENES,
) -> Dict[str, Any]:
    """Calibrates and evaluates a single model predictions CSV file across Holdout, LOSO, and LODO."""
    df = pd.read_csv(csv_path)

    # Exclude identical reference comparison if present
    if "distortion_level" in df.columns:
        df = df[df["distortion_level"] != "Ref"].reset_index(drop=True)

    # 1. Holdout Train / Val evaluation
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

    calib_a, calib_b = fit_calibration_curve(q_train, jod_train)

    df_train["jod_calibrated"] = power_law_jod(q_train, calib_a, calib_b)
    df_val["jod_calibrated"] = power_law_jod(q_val, calib_a, calib_b)
    df["jod_calibrated"] = power_law_jod(q_all, calib_a, calib_b)

    metrics_train_calib = compute_metrics(jod_train, df_train["jod_calibrated"].to_numpy())
    metrics_val_calib = compute_metrics(jod_val, df_val["jod_calibrated"].to_numpy())
    metrics_all_calib = compute_metrics(jod_all, df["jod_calibrated"].to_numpy())

    # 2. Leave-One-Scene-Out (LOSO) Cross-Validation
    loso_summary = leave_one_scene_out_cv(df)

    # 3. Leave-One-Distortion-Out (LODO) Cross-Validation
    lodo_summary = leave_one_distortion_out_cv(df)

    # 4. Per-distortion breakdown on holdout-calibrated predictions
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
        "metrics_val_calib": metrics_val_calib,
        "metrics_all_calib": metrics_all_calib,
        "loso_summary": loso_summary,
        "lodo_summary": lodo_summary,
        "distortion_breakdown": distortion_breakdown,
        "df_with_calib": df,
    }


def plot_model_comparisons(
    results_list: List[Dict[str, Any]],
    output_dir: str,
):
    """Generates visual comparison figures for evaluated models (Holdout and LOSO OOF)."""
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
        ax.plot([4, 10], [4, 10], "k--", alpha=0.5, label="Identity")

        val_m = res["metrics_val_calib"]
        loso = res["loso_summary"]
        lodo = res["lodo_summary"]

        ax.set_title(
            f"{model_title.upper()}\n"
            f"Holdout Val r: {val_m['pearson_r']:.3f} | RMSE: {val_m['rmse']:.2f}\n"
            f"LOSO Mean r: {loso.fold_mean_metrics['pearson_r']:.3f} ± {loso.fold_std_metrics['pearson_r']:.3f}\n"
            f"LODO Mean r: {lodo.fold_mean_metrics['pearson_r']:.3f} ± {lodo.fold_std_metrics['pearson_r']:.3f}",
            fontsize=9.5,
        )
        ax.set_xlabel("Calibrated Predicted JOD", fontsize=11)
        ax.set_ylabel("Ground Truth JOD (GAIM-240)", fontsize=11)
        ax.set_xlim([3.5, 10.5])
        ax.set_ylim([3.5, 10.5])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_calibration_scatter_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved scatter comparison plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate and Cross-Validate Models on GAIM-240")
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

    print("=" * 85)
    print("GAIM-240 Model Calibration: Holdout, LOSO (Scenes), & LODO (Distortions)")
    print(f"Input CSV files : {len(args.csv_files)}")
    print(f"Output dir      : {args.output_dir}")
    print("=" * 85)

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

        loso = res["loso_summary"]
        lodo = res["lodo_summary"]

        summary_rows.append({
            "Model": model_name,
            "Calib (a, b)": f"({calib_a:.4f}, {calib_b:.4f})",
            # Holdout Fixed Split
            "Holdout Val PLCC": val_m["pearson_r"],
            "Holdout Val RMSE": val_m["rmse"],
            # LOSO (Content Generalization)
            "LOSO Mean±Std PLCC": f"{loso.fold_mean_metrics['pearson_r']:.4f} ± {loso.fold_std_metrics['pearson_r']:.4f}",
            "LOSO Mean±Std SROCC": f"{loso.fold_mean_metrics['spearman_rho']:.4f} ± {loso.fold_std_metrics['spearman_rho']:.4f}",
            "LOSO Mean±Std RMSE": f"{loso.fold_mean_metrics['rmse']:.4f} ± {loso.fold_std_metrics['rmse']:.4f}",
            "LOSO OOF PLCC": loso.oof_metrics["pearson_r"],
            "LOSO OOF RMSE": loso.oof_metrics["rmse"],
            # LODO (Distortion Generalization)
            "LODO Mean±Std PLCC": f"{lodo.fold_mean_metrics['pearson_r']:.4f} ± {lodo.fold_std_metrics['pearson_r']:.4f}",
            "LODO Mean±Std SROCC": f"{lodo.fold_mean_metrics['spearman_rho']:.4f} ± {lodo.fold_std_metrics['spearman_rho']:.4f}",
            "LODO Mean±Std RMSE": f"{lodo.fold_mean_metrics['rmse']:.4f} ± {lodo.fold_std_metrics['rmse']:.4f}",
            "LODO OOF PLCC": lodo.oof_metrics["pearson_r"],
            "LODO OOF RMSE": lodo.oof_metrics["rmse"],
            # Global
            "Train PLCC": train_m["pearson_r"],
            "All PLCC": all_m["pearson_r"],
        })

        print(f"  Fitted Parameters: a = {calib_a:.6f}, b = {calib_b:.6f}")
        print(f"  Holdout Val (subway, zeroday) -> PLCC: {val_m['pearson_r']:.4f} | RMSE: {val_m['rmse']:.4f}")
        print(f"  LOSO (9 scenes, Mean ± Std)  -> PLCC: {loso.fold_mean_metrics['pearson_r']:.4f} ± {loso.fold_std_metrics['pearson_r']:.4f} | SROCC: {loso.fold_mean_metrics['spearman_rho']:.4f} ± {loso.fold_std_metrics['spearman_rho']:.4f} | RMSE: {loso.fold_mean_metrics['rmse']:.4f} ± {loso.fold_std_metrics['rmse']:.4f}")
        print(f"  LOSO Range [Min, Max]        -> PLCC: [{loso.fold_range_metrics['pearson_r'][0]:.4f}, {loso.fold_range_metrics['pearson_r'][1]:.4f}]")
        print(f"  LODO (8 dists, Mean ± Std)   -> PLCC: {lodo.fold_mean_metrics['pearson_r']:.4f} ± {lodo.fold_std_metrics['pearson_r']:.4f} | SROCC: {lodo.fold_mean_metrics['spearman_rho']:.4f} ± {lodo.fold_std_metrics['spearman_rho']:.4f} | RMSE: {lodo.fold_mean_metrics['rmse']:.4f} ± {lodo.fold_std_metrics['rmse']:.4f}")
        print(f"  LODO Range [Min, Max]        -> PLCC: [{lodo.fold_range_metrics['pearson_r'][0]:.4f}, {lodo.fold_range_metrics['pearson_r'][1]:.4f}]")

        # Save per-fold breakdown CSVs
        loso_folds_df = pd.DataFrame([
            {
                "Fold": f.fold_id,
                "Heldout Scene": f.test_group,
                "Train Samples": f.train_size,
                "Val Samples": f.test_size,
                "Calib a": f.calib_a,
                "Calib b": f.calib_b,
                "PLCC": f.metrics["pearson_r"],
                "SROCC": f.metrics["spearman_rho"],
                "RMSE": f.metrics["rmse"],
                "MAE": f.metrics["mae"],
            }
            for f in loso.fold_results
        ])
        loso_folds_path = os.path.join(args.output_dir, f"loso_breakdown_{model_name}.csv")
        loso_folds_df.to_csv(loso_folds_path, index=False)

        lodo_folds_df = pd.DataFrame([
            {
                "Fold": f.fold_id,
                "Heldout Distortion": f.test_group,
                "Train Samples": f.train_size,
                "Val Samples": f.test_size,
                "Calib a": f.calib_a,
                "Calib b": f.calib_b,
                "PLCC": f.metrics["pearson_r"],
                "SROCC": f.metrics["spearman_rho"],
                "RMSE": f.metrics["rmse"],
                "MAE": f.metrics["mae"],
            }
            for f in lodo.fold_results
        ])
        lodo_folds_path = os.path.join(args.output_dir, f"lodo_breakdown_{model_name}.csv")
        lodo_folds_df.to_csv(lodo_folds_path, index=False)

    if all_results:
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "calibration_summary_metrics.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSaved summary table to {summary_csv}")

        plot_model_comparisons(all_results, args.output_dir)


if __name__ == "__main__":
    main()
