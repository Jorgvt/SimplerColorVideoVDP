#!/usr/bin/env python3
"""
Generate publication-quality box plots comparing model generalization under
Leave-One-Scene-Out (LOSO) and Leave-One-Distortion-Out (LODO) cross-validation.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Ensure workspace imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(repo_root, "src")
sys.path.insert(0, repo_root)
sys.path.insert(0, src_dir)

from simplercolorvideovdp.evaluation import (
    compute_metrics,
    fit_calibration_curve,
    leave_one_distortion_out_cv,
    leave_one_scene_out_cv,
    power_law_jod,
)


def safe_pearson_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Robust differentiable Pearson loss."""
    pm = pred - np.mean(pred)
    tm = target - np.mean(target)
    sp = np.sum(pm ** 2)
    st = np.sum(tm ** 2)
    if sp < 1e-8 or st < 1e-8:
        return 1.0
    r = np.sum(pm * tm) / (np.sqrt(sp * st) + 1e-8)
    return float(1.0 - r)


def extract_fold_data() -> pd.DataFrame:
    """Runs cross-validation across models and extracts per-fold metric dataframes."""
    exp04_dir = os.path.join(repo_root, "experiments", "exp04_gaim240_model_calibration")
    exp08_dir = os.path.join(repo_root, "experiments", "exp08_canonical_v1mt_full_resolution")

    df_base = pd.read_csv(os.path.join(exp04_dir, "predictions_baseline_720p_full.csv"))
    df_ori = pd.read_csv(os.path.join(exp04_dir, "predictions_oriented_720p_full.csv"))

    cvvdp_df = pd.read_csv(os.path.join(exp08_dir, "cvvdp_gaim240_features_720x1280_1200f_240hz.csv"))
    v1mt_df = pd.read_csv(os.path.join(exp08_dir, "v1mt_gaim240_features_720x1280_1200f_240hz.csv"))

    merged = pd.merge(
        cvvdp_df,
        v1mt_df[["scene_name", "distortion_type", "distortion_level", "d_v1mt"]],
        on=["scene_name", "distortion_type", "distortion_level"],
    )
    merged = merged[merged["distortion_level"] != "Ref"].reset_index(drop=True)

    records: List[Dict[str, Any]] = []

    def process_single_q(model_name: str, df: pd.DataFrame, q_col: str, gt_col: str):
        loso = leave_one_scene_out_cv(df, q_col=q_col, gt_col=gt_col)
        for f in loso.fold_results:
            records.append({
                "Model": model_name,
                "CV_Scheme": "LOSO (Content / Scenes)",
                "Group": f.test_group,
                "PLCC": f.metrics["pearson_r"],
                "SROCC": f.metrics["spearman_rho"],
                "RMSE": f.metrics["rmse"],
                "MAE": f.metrics["mae"],
            })

        lodo = leave_one_distortion_out_cv(df, q_col=q_col, gt_col=gt_col)
        for f in lodo.fold_results:
            records.append({
                "Model": model_name,
                "CV_Scheme": "LODO (Distortions)",
                "Group": f.test_group,
                "PLCC": f.metrics["pearson_r"],
                "SROCC": f.metrics["spearman_rho"],
                "RMSE": f.metrics["rmse"],
                "MAE": f.metrics["mae"],
            })

    # Process baseline, oriented, and standalone canonical V1-MT
    process_single_q("Baseline CVVDP", df_base, "raw_q", "jod_gt")
    process_single_q("Oriented CVVDP", df_ori, "raw_q", "jod_gt")
    process_single_q("Canonical V1-MT", merged, "d_v1mt", "gt_jod")

    # Process Hybrid models (Linear fusion with MSE and Pearson loss)
    def process_hybrid(model_name: str, loss_type: str):
        for group_col, cv_scheme in [("scene_name", "LOSO (Content / Scenes)"), ("distortion_type", "LODO (Distortions)")]:
            groups = sorted(merged[group_col].unique())
            for g in groups:
                train_mask = (merged[group_col] != g).to_numpy()
                test_mask = (merged[group_col] == g).to_numpy()

                X_train = np.column_stack([merged.loc[train_mask, "d_cvvdp"], merged.loc[train_mask, "d_v1mt"]])
                y_train = merged.loc[train_mask, "gt_jod"].to_numpy()

                X_test = np.column_stack([merged.loc[test_mask, "d_cvvdp"], merged.loc[test_mask, "d_v1mt"]])
                y_test = merged.loc[test_mask, "gt_jod"].to_numpy()

                def obj(params):
                    w1, w2, a, b = params
                    q_comb = w1 * X_train[:, 0] + w2 * X_train[:, 1]
                    pred = power_law_jod(q_comb, a, b)
                    if loss_type == "pearson":
                        return safe_pearson_loss(pred, y_train)
                    return float(np.mean((pred - y_train) ** 2))

                res = minimize(
                    obj,
                    [1.0, 1.0, 0.1, 0.6],
                    bounds=[(0.01, 50.0), (0.01, 50.0), (1e-5, 5.0), (1e-3, 3.0)],
                    method="L-BFGS-B",
                    options={"maxiter": 80},
                )
                w1, w2, a, b = res.x

                q_comb_test = w1 * X_test[:, 0] + w2 * X_test[:, 1]
                pred_test = power_law_jod(q_comb_test, a, b)
                m = compute_metrics(y_test, pred_test)

                records.append({
                    "Model": model_name,
                    "CV_Scheme": cv_scheme,
                    "Group": str(g),
                    "PLCC": m["pearson_r"],
                    "SROCC": m["spearman_rho"],
                    "RMSE": m["rmse"],
                    "MAE": m["mae"],
                })

    process_hybrid("Canonical Hybrid (MSE)", "mse")
    process_hybrid("Canonical Hybrid (Pearson)", "pearson")

    return pd.DataFrame(records)


def plot_boxplots(df: pd.DataFrame, output_dir: str):
    """Plots multi-panel box plots with strip points and mean markers."""
    os.makedirs(output_dir, exist_ok=True)

    # Style configuration
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.1

    models = [
        "Baseline CVVDP",
        "Oriented CVVDP",
        "Canonical V1-MT",
        "Canonical Hybrid (MSE)",
        "Canonical Hybrid (Pearson)",
    ]

    palette = [
        "#386cb0",  # Dark Blue
        "#fdb462",  # Amber
        "#7fc97f",  # Green
        "#e7298a",  # Magenta / Coral
        "#66c2a5",  # Teal
    ]

    model_colors = dict(zip(models, palette))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True)

    metrics_configs = [
        ("PLCC", "Pearson Correlation (PLCC)", [-0.15, 1.05]),
        ("RMSE", "Root Mean Square Error (RMSE in JOD)", [0.35, 1.85]),
    ]

    cv_schemes = [
        ("LOSO (Content / Scenes)", "Leave-One-Scene-Out (9 Scenes) - Content Generalization"),
        ("LODO (Distortions)", "Leave-One-Distortion-Out (8 Distortions) - Distortion Generalization"),
    ]

    for row_idx, (cv_scheme, scheme_title) in enumerate(cv_schemes):
        for col_idx, (metric_key, metric_label, y_lim) in enumerate(metrics_configs):
            ax = axes[row_idx, col_idx]

            sub_df = df[df["CV_Scheme"] == cv_scheme]

            # Prepare data per model
            data_per_model = [
                sub_df[sub_df["Model"] == m][metric_key].dropna().to_numpy()
                for m in models
            ]

            # 1. Box Plot
            bp = ax.boxplot(
                data_per_model,
                positions=np.arange(len(models)),
                widths=0.48,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 6.5,
                    "markeredgewidth": 1.2,
                },
                medianprops={"color": "black", "linewidth": 1.8},
                boxprops={"alpha": 0.70, "edgecolor": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 1.2, "linestyle": "--"},
                capprops={"color": "black", "linewidth": 1.2},
                flierprops={"marker": "x", "markeredgecolor": "red", "markersize": 5},
            )

            # Color each box
            for patch, m in zip(bp["boxes"], models):
                patch.set_facecolor(model_colors[m])

            # 2. Jittered Scatter Points (Swarm / Strip overlay)
            np.random.seed(42)
            for m_idx, m in enumerate(models):
                vals = sub_df[sub_df["Model"] == m][metric_key].dropna().to_numpy()
                x_jitter = np.random.normal(m_idx, 0.06, size=len(vals))
                ax.scatter(
                    x_jitter,
                    vals,
                    color=model_colors[m],
                    edgecolors="black",
                    linewidths=0.7,
                    s=45,
                    alpha=0.9,
                    zorder=3,
                )

            # Annotate Mean ± Std
            for m_idx, m in enumerate(models):
                vals = sub_df[sub_df["Model"] == m][metric_key].dropna().to_numpy()
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                y_pos = y_lim[1] - (0.08 * (y_lim[1] - y_lim[0])) if metric_key == "PLCC" else y_lim[0] + (0.06 * (y_lim[1] - y_lim[0]))
                ax.text(
                    m_idx,
                    y_pos,
                    f"{mean_val:.3f}\n±{std_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
                )

            # Subplot titles and labels
            ax.set_title(f"{scheme_title}\n{metric_label}", fontsize=11, fontweight="bold", pad=6)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_ylim(y_lim)
            ax.grid(True, linestyle=":", alpha=0.5, axis="y")
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(models, rotation=14, ha="right", fontsize=9.5, fontweight="medium")

    # Super Title
    fig.suptitle(
        "GAIM-240 Cross-Validation Performance Distributions: Content (LOSO) vs Distortion (LODO)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cross_validation_distribution_boxplots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution box plots figure to: {plot_path}")

    # Also save CSV of all fold records
    csv_path = os.path.join(output_dir, "cross_validation_all_folds_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved all folds data CSV to: {csv_path}")


def main():
    print("=" * 80)
    print("Generating Cross-Validation Performance Distribution Box Plots")
    print("=" * 80)

    output_dir = os.path.join(repo_root, "experiments", "exp08_canonical_v1mt_full_resolution", "plots")
    df_folds = extract_fold_data()
    plot_boxplots(df_folds, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
