"""Experiment 08: Canonical 6-Channel Direct-Stream V1-MT Cortical Model and Hybrid Metric.

Calibrates and evaluates the Canonical Hybrid ColorVideoVDP model on the GAIM-240 benchmark:
- 6-channel V1-MT cortical pathway: 3 MT motion speeds (1.0, 4.0, 16.0 deg/s) + 3 DKL sustained form channels (A, RG, BY)
- ColorVideoVDP / Isotropic backbone
- Linear fusion with holdout cross-validation (Train / Val / All)
- Per-distortion performance breakdown (dlss_rr, judder, restir, trm)
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add workspace paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir = os.path.join(repo_root, "src")
sys.path.insert(0, repo_root)
sys.path.insert(0, src_dir)

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES
from simplercolorvideovdp.v1mt_canonical import CanonicalHybridColorVideoVDP


def compute_pearson_loss(pred_jod: torch.Tensor, gt_jod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes differentiable Pearson correlation loss: L = 1 - r."""
    pred_m = pred_jod - torch.mean(pred_jod)
    gt_m = gt_jod - torch.mean(gt_jod)

    S_pp = torch.sum(pred_m**2) + 1e-8
    S_yy = torch.sum(gt_m**2) + 1e-8
    S_py = torch.sum(pred_m * gt_m)

    r = S_py / torch.sqrt(S_pp * S_yy)
    loss = 1.0 - r
    return loss, r


def evaluate_metric(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Computes PLCC, SRCC, and RMSE between predicted and ground truth values."""
    plcc, _ = pearsonr(y_pred, y_true)
    srcc, _ = spearmanr(y_pred, y_true)
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {
        "plcc": float(plcc),
        "srcc": float(srcc),
        "rmse": float(rmse),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Canonical V1-MT + CVVDP Hybrid Model Evaluation on GAIM-240")
    parser.add_argument("--v1mt_csv", type=str, default=None)
    parser.add_argument("--cvvdp_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--val_scenes", nargs="+", default=DEFAULT_VAL_SCENES)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("EXPERIMENT 08: CANONICAL DIRECT-STREAM V1-MT + COLORVIDEOVDP HYBRID EVALUATION")
    print("=" * 80)
    print(f"Device:               {device}")
    print(f"Holdout Val Scenes:   {args.val_scenes}")
    print(f"Optimizing:           w_cvvdp, w_v1mt, JOD intercept, JOD slope")
    print("=" * 80)

    # 1. Resolve CSV Feature Paths
    v1mt_csv_candidates = [
        args.v1mt_csv,
        os.path.join(args.output_dir, "v1mt_gaim240_features_720x1280_1200f_240hz.csv"),
        os.path.join(repo_root, "..", "v1mt", "experiments", "30_frozen_v1mt_cvvdp_full_resolution", "v1mt_gaim240_features_720x1280_1200f_240hz.csv"),
    ]
    v1mt_csv_path = next((p for p in v1mt_csv_candidates if p and os.path.exists(p)), None)

    cvvdp_csv_candidates = [
        args.cvvdp_csv,
        os.path.join(args.output_dir, "cvvdp_gaim240_features_720x1280_1200f_240hz.csv"),
        os.path.join(repo_root, "..", "v1mt", "experiments", "30_frozen_v1mt_cvvdp_full_resolution", "cvvdp_gaim240_features_720x1280_1200f_240hz.csv"),
    ]
    cvvdp_csv_path = next((p for p in cvvdp_csv_candidates if p and os.path.exists(p)), None)

    if not v1mt_csv_path or not cvvdp_csv_path:
        raise FileNotFoundError(f"Feature CSVs not found!\n  v1mt: {v1mt_csv_path}\n  cvvdp: {cvvdp_csv_path}")

    print(f"Loading V1-MT features from: {v1mt_csv_path}")
    v1mt_df = pd.read_csv(v1mt_csv_path)
    print(f"Loading CVVDP features from: {cvvdp_csv_path}")
    cvvdp_df = pd.read_csv(cvvdp_csv_path)

    # Merge features
    merged = pd.merge(
        cvvdp_df[['ref_vid_path', 'dist_vid_path', 'scene_name', 'distortion_type', 'distortion_level', 'gt_jod', 'cvvdp_jod']],
        v1mt_df[['ref_vid_path', 'dist_vid_path', 'd_v1mt', 'mae_slow', 'mae_med', 'mae_fast', 'mae_fa', 'mae_frg', 'mae_fby']],
        on=['ref_vid_path', 'dist_vid_path']
    )
    print(f"Total video pairs merged: {len(merged)}")

    # Calibrate CVVDP difference in JOD impairment units (10 - cvvdp_jod)
    merged['d_cvvdp'] = np.maximum(0.0, 10.0 - merged['cvvdp_jod'])

    is_val = merged['scene_name'].isin(args.val_scenes)
    train_df = merged[~is_val].copy()
    val_df = merged[is_val].copy()
    print(f"Train Pairs: {len(train_df)} | Val Pairs: {len(val_df)}")

    # Tensors
    train_d_c = torch.tensor(train_df['d_cvvdp'].values, dtype=torch.float32, device=device)
    train_d_v = torch.tensor(train_df['d_v1mt'].values, dtype=torch.float32, device=device)
    train_gt = torch.tensor(train_df['gt_jod'].values, dtype=torch.float32, device=device)

    val_d_c = torch.tensor(val_df['d_cvvdp'].values, dtype=torch.float32, device=device)
    val_d_v = torch.tensor(val_df['d_v1mt'].values, dtype=torch.float32, device=device)
    val_gt = torch.tensor(val_df['gt_jod'].values, dtype=torch.float32, device=device)

    all_d_c = torch.tensor(merged['d_cvvdp'].values, dtype=torch.float32, device=device)
    all_d_v = torch.tensor(merged['d_v1mt'].values, dtype=torch.float32, device=device)
    all_gt = torch.tensor(merged['gt_jod'].values, dtype=torch.float32, device=device)

    # 2. Instantiate and Optimize Hybrid Model
    hybrid_model = CanonicalHybridColorVideoVDP(
        fps=240.0,
        init_w_cvvdp=1.0,
        init_w_v1mt=1.0,
        jod_intercept=10.0,
        jod_slope=2.0,
        device=device
    ).to(device)

    optimizer = optim.AdamW([
        {'params': [hybrid_model.raw_w_cvvdp, hybrid_model.raw_w_v1mt], 'lr': args.lr, 'weight_decay': 1e-5},
        {'params': [hybrid_model.jod_intercept, hybrid_model.raw_jod_slope], 'lr': args.lr * 2.0, 'weight_decay': 0.0}
    ])

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        hybrid_model.train()
        optimizer.zero_grad()

        pred_train, _ = hybrid_model.forward_from_features(train_d_c, train_d_v)
        p_loss, r_train = compute_pearson_loss(pred_train, train_gt)
        mse_loss = F.mse_loss(pred_train, train_gt)
        total_loss = p_loss + 0.1 * mse_loss

        total_loss.backward()
        optimizer.step()

        hybrid_model.eval()
        with torch.no_grad():
            pred_val, _ = hybrid_model.forward_from_features(val_d_c, val_d_v)
            val_r_np, _ = pearsonr(pred_val.cpu().numpy(), val_gt.cpu().numpy())

            if val_r_np > best_val_r:
                best_val_r = val_r_np
                best_state = {k: v.cpu().clone() for k, v in hybrid_model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1 or epoch == args.epochs:
            train_r_np, _ = pearsonr(pred_train.detach().cpu().numpy(), train_gt.cpu().numpy())
            print(f"Epoch {epoch:03d}/{args.epochs} | Train PLCC: {train_r_np:+.4f} | Val PLCC: {val_r_np:+.4f} | "
                  f"w_cvvdp: {hybrid_model.w_cvvdp.item():.4f}, w_v1mt: {hybrid_model.w_v1mt.item():.4f}")

    if best_state is not None:
        hybrid_model.load_state_dict(best_state)
    hybrid_model.eval()

    # 3. Final Predictions & Benchmarking
    with torch.no_grad():
        final_preds, final_d_hybrid = hybrid_model.forward_from_features(all_d_c, all_d_v)
        merged['hybrid_pred_jod'] = final_preds.cpu().numpy()
        merged['d_hybrid'] = final_d_hybrid.cpu().numpy()

    # Calibrate V1-MT standalone JOD linearly for RMSE calculation
    p_v1 = np.polyfit(merged['d_v1mt'].values, merged['gt_jod'].values, deg=1)
    merged['v1mt_pred_jod'] = np.polyval(p_v1, merged['d_v1mt'].values)

    # Save detailed evaluation results CSV
    results_csv = os.path.join(args.output_dir, "canonical_hybrid_evaluation_results.csv")
    merged.to_csv(results_csv, index=False)
    print(f"\nSaved detailed evaluation results to:\n  --> {results_csv}")

    # Calculate splits performance
    m_train = merged[~is_val]
    m_val = merged[is_val]

    benchmarks = {
        "cvvdp": {
            "all": evaluate_metric(merged['cvvdp_jod'].values, merged['gt_jod'].values),
            "train": evaluate_metric(m_train['cvvdp_jod'].values, m_train['gt_jod'].values),
            "val": evaluate_metric(m_val['cvvdp_jod'].values, m_val['gt_jod'].values),
        },
        "v1mt_standalone": {
            "all": evaluate_metric(merged['v1mt_pred_jod'].values, merged['gt_jod'].values),
            "train": evaluate_metric(m_train['v1mt_pred_jod'].values, m_train['gt_jod'].values),
            "val": evaluate_metric(m_val['v1mt_pred_jod'].values, m_val['gt_jod'].values),
        },
        "canonical_hybrid": {
            "all": evaluate_metric(merged['hybrid_pred_jod'].values, merged['gt_jod'].values),
            "train": evaluate_metric(m_train['hybrid_pred_jod'].values, m_train['gt_jod'].values),
            "val": evaluate_metric(m_val['hybrid_pred_jod'].values, m_val['gt_jod'].values),
        }
    }

    # Per-distortion breakdown across all distortion types
    distortions = sorted(list(merged['distortion_type'].unique()))
    per_distortion = {}
    for d in distortions:
        sub = merged[merged['distortion_type'] == d]
        per_distortion[d] = {
            "cvvdp": evaluate_metric(sub['cvvdp_jod'].values, sub['gt_jod'].values),
            "v1mt_standalone": evaluate_metric(sub['v1mt_pred_jod'].values, sub['gt_jod'].values),
            "canonical_hybrid": evaluate_metric(sub['hybrid_pred_jod'].values, sub['gt_jod'].values),
        }

    opt_params = {
        "w_cvvdp": float(hybrid_model.w_cvvdp.item()),
        "w_v1mt": float(hybrid_model.w_v1mt.item()),
        "jod_intercept": float(hybrid_model.jod_intercept.item()),
        "jod_slope": float(hybrid_model.jod_slope.item()),
    }

    summary = {
        "benchmarks": benchmarks,
        "per_distortion": per_distortion,
        "optimal_parameters": opt_params,
    }

    metrics_json = os.path.join(args.output_dir, "canonical_hybrid_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to:\n  --> {metrics_json}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("GAIM-240 BENCHMARK EVALUATION RESULTS (720p @ 240Hz, 1200 frames)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'All PLCC':<10} | {'All SRCC':<10} | {'Train PLCC':<10} | {'Val PLCC':<10}")
    print("-" * 80)
    print(f"{'ColorVideoVDP Baseline':<28} | {benchmarks['cvvdp']['all']['plcc']:+.4f}     | {benchmarks['cvvdp']['all']['srcc']:+.4f}     | {benchmarks['cvvdp']['train']['plcc']:+.4f}     | {benchmarks['cvvdp']['val']['plcc']:+.4f}")
    print(f"{'Canonical V1-MT Standalone':<28} | {benchmarks['v1mt_standalone']['all']['plcc']:+.4f}     | {benchmarks['v1mt_standalone']['all']['srcc']:+.4f}     | {benchmarks['v1mt_standalone']['train']['plcc']:+.4f}     | {benchmarks['v1mt_standalone']['val']['plcc']:+.4f}")
    print(f"{'Canonical Hybrid (Exp 08)':<28} | {benchmarks['canonical_hybrid']['all']['plcc']:+.4f}     | {benchmarks['canonical_hybrid']['all']['srcc']:+.4f}     | {benchmarks['canonical_hybrid']['train']['plcc']:+.4f}     | {benchmarks['canonical_hybrid']['val']['plcc']:+.4f}")
    print("=" * 80)

    print("\nPER-DISTORTION PLCC BREAKDOWN:")
    print(f"{'Distortion':<35} | {'CVVDP':<10} | {'V1-MT':<10} | {'Canonical Hybrid':<16}")
    print("-" * 78)
    for d in distortions:
        cv_r = per_distortion[d]['cvvdp']['plcc']
        v1_r = per_distortion[d]['v1mt_standalone']['plcc']
        hy_r = per_distortion[d]['canonical_hybrid']['plcc']
        print(f"{d:<35} | {cv_r:+.4f}     | {v1_r:+.4f}     | {hy_r:+.4f}")
    print("=" * 78)

    # 4. Generate Visualizations
    # 4.1 Scatter Plot
    plt.figure(figsize=(8, 7), dpi=300)
    plt.scatter(merged['gt_jod'], merged['hybrid_pred_jod'], alpha=0.75, edgecolors='none', c='#1f77b4', s=48, label='GAIM-240 Pairs (N=216)')
    min_v = min(merged['gt_jod'].min(), merged['hybrid_pred_jod'].min()) - 0.5
    max_v = max(merged['gt_jod'].max(), merged['hybrid_pred_jod'].max()) + 0.5
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Unity Line (y = x)')
    plt.title(f"Canonical 6-Channel V1-MT + CVVDP Hybrid Model (GAIM-240 720p @ 240Hz)\n"
              f"All PLCC = {benchmarks['canonical_hybrid']['all']['plcc']:+.4f} | SRCC = {benchmarks['canonical_hybrid']['all']['srcc']:+.4f} | Val PLCC = {benchmarks['canonical_hybrid']['val']['plcc']:+.4f}", fontsize=11)
    plt.xlabel("Ground Truth Subjective Quality (JOD)", fontsize=11)
    plt.ylabel("Predicted Quality (JOD)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, "canonical_hybrid_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # 4.2 Overall Comparison Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    names = ['ColorVideoVDP\nBaseline', 'Canonical V1-MT\nStandalone', 'Canonical Hybrid\n(Exp 08)']
    plcc_vals = [benchmarks['cvvdp']['all']['plcc'], benchmarks['v1mt_standalone']['all']['plcc'], benchmarks['canonical_hybrid']['all']['plcc']]
    srcc_vals = [benchmarks['cvvdp']['all']['srcc'], benchmarks['v1mt_standalone']['all']['srcc'], benchmarks['canonical_hybrid']['all']['srcc']]

    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, plcc_vals, width, label='PLCC (Linear)', color='#4C72B0', alpha=0.9)
    plt.bar(x + width/2, srcc_vals, width, label='SRCC (Rank)', color='#55A868', alpha=0.9)

    plt.ylabel('Correlation with Human JOD', fontsize=11)
    plt.title('Overall Metric Benchmark Comparison on GAIM-240 (Full 1200 Frames)', fontsize=12)
    plt.xticks(x, names, fontsize=10)
    plt.ylim(0.0, 0.8)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper left')

    for i in range(len(names)):
        plt.text(i - width/2, plcc_vals[i] + 0.015, f"{plcc_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width/2, srcc_vals[i] + 0.015, f"{srcc_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    bar_path = os.path.join(plots_dir, "benchmark_comparison_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # 4.3 Per-Distortion Breakdown Bar Chart
    plt.figure(figsize=(14, 5.5), dpi=300)
    x = np.arange(len(distortions))
    width = 0.26
    cv_bars = [per_distortion[d]['cvvdp']['plcc'] for d in distortions]
    v1_bars = [per_distortion[d]['v1mt_standalone']['plcc'] for d in distortions]
    hy_bars = [per_distortion[d]['canonical_hybrid']['plcc'] for d in distortions]

    label_map = {
        'dlss_rr': 'DLSS RR',
        'judder': 'Judder',
        'motion_noise': 'Motion Noise',
        'motion_resolution': 'Motion Res',
        'noise_colors': 'Noise Colors',
        'restir': 'ReSTIR',
        'stutter': 'Stutter',
        'temporal-resolution-multiplexing': 'TRM'
    }
    dist_labels = [label_map.get(d, d) for d in distortions]

    plt.bar(x - width, cv_bars, width, label='ColorVideoVDP Baseline', color='#4C72B0', alpha=0.9)
    plt.bar(x, v1_bars, width, label='Canonical V1-MT Standalone', color='#C44E52', alpha=0.9)
    plt.bar(x + width, hy_bars, width, label='Canonical Hybrid (Exp 08)', color='#55A868', alpha=0.9)

    plt.ylabel('PLCC Correlation with JOD', fontsize=11)
    plt.title('Per-Distortion Performance Breakdown on GAIM-240 (All 8 Distortions)', fontsize=12)
    plt.xticks(x, dist_labels, fontsize=10)
    plt.ylim(-0.1, 0.95)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper left')

    for i in range(len(distortions)):
        plt.text(i - width, max(0.01, cv_bars[i]) + 0.015, f"{cv_bars[i]:.2f}", ha='center', fontsize=8)
        plt.text(i, max(0.01, v1_bars[i]) + 0.015, f"{v1_bars[i]:.2f}", ha='center', fontsize=8)
        plt.text(i + width, max(0.01, hy_bars[i]) + 0.015, f"{hy_bars[i]:.2f}", ha='center', fontsize=8, fontweight='bold')


    plt.tight_layout()
    dist_bar_path = os.path.join(plots_dir, "distortion_breakdown_bar.png")
    plt.savefig(dist_bar_path, dpi=300)
    plt.close()

    print(f"\nGenerated plots in {plots_dir}:")
    print(f"  - {scatter_path}")
    print(f"  - {bar_path}")
    print(f"  - {dist_bar_path}")


if __name__ == "__main__":
    main()
