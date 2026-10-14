"""Experiment 09: MT Speed Tuning Ablation Study.

Assesses whether the performance increase comes from:
1. Multi-speed velocity tuning in MT (1.0 vs 4.0 vs 16.0 deg/s vs combined 3-speed)
2. Opponent color form pathway (Achromatic, Red-Green, Blue-Yellow)
3. Synergistic hybrid fusion with the ColorVideoVDP / Isotropic backbone.
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir = os.path.join(repo_root, "src")
sys.path.insert(0, repo_root)
sys.path.insert(0, src_dir)

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES


def compute_pearson_loss(pred_jod: torch.Tensor, gt_jod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Pearson loss: L = 1 - r."""
    pred_m = pred_jod - torch.mean(pred_jod)
    gt_m = gt_jod - torch.mean(gt_jod)

    S_pp = torch.sum(pred_m**2) + 1e-8
    S_yy = torch.sum(gt_m**2) + 1e-8
    S_py = torch.sum(pred_m * gt_m)

    r = S_py / torch.sqrt(S_pp * S_yy)
    loss = 1.0 - r
    return loss, r


def evaluate_metric(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Computes PLCC, SRCC, RMSE."""
    plcc, _ = pearsonr(y_pred, y_true)
    srcc, _ = spearmanr(y_pred, y_true)
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {
        "plcc": float(plcc),
        "srcc": float(srcc),
        "rmse": float(rmse),
    }


class AblationLinearFusionModel(nn.Module):
    """Linear fusion model for arbitrary subsets of feature channels."""

    def __init__(self, num_features: int, init_w: float = 1.0):
        super().__init__()
        # Initialize positive weights via inverse softplus
        init_raw = math.log(math.exp(init_w) - 1.0) if init_w < 20.0 else init_w
        self.raw_weights = nn.Parameter(torch.full((num_features,), init_raw, dtype=torch.float32))
        self.jod_intercept = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.raw_jod_slope = nn.Parameter(torch.tensor(math.log(math.exp(2.0) - 1.0), dtype=torch.float32))

    @property
    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_weights)

    @property
    def jod_slope(self) -> torch.Tensor:
        return F.softplus(self.raw_jod_slope)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: shape (N, num_features)"""
        w = self.weights.unsqueeze(0)  # (1, num_features)
        d_combined = torch.sum(w * features, dim=1)  # (N,)
        pred_jod = self.jod_intercept - self.jod_slope * d_combined
        return pred_jod, d_combined


def fit_and_evaluate_configuration(
    name: str,
    feature_cols: List[str],
    df: pd.DataFrame,
    val_scenes: List[str],
    epochs: int = 150,
    lr: float = 0.01,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, Any]:
    """Fits and evaluates a specific combination of features."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    is_val = df['scene_name'].isin(val_scenes)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=dev)
    y_train = torch.tensor(train_df['gt_jod'].values, dtype=torch.float32, device=dev)

    X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=dev)
    y_val = torch.tensor(val_df['gt_jod'].values, dtype=torch.float32, device=dev)

    X_all = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=dev)
    y_all = torch.tensor(df['gt_jod'].values, dtype=torch.float32, device=dev)

    model = AblationLinearFusionModel(num_features=len(feature_cols)).to(dev)
    optimizer = optim.AdamW([
        {'params': [model.raw_weights], 'lr': lr, 'weight_decay': 1e-5},
        {'params': [model.jod_intercept, model.raw_jod_slope], 'lr': lr * 2.0, 'weight_decay': 0.0}
    ])

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train, _ = model(X_train)
        p_loss, _ = compute_pearson_loss(pred_train, y_train)
        mse_loss = F.mse_loss(pred_train, y_train)
        loss = p_loss + 0.1 * mse_loss

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, _ = model(X_val)
            val_r, _ = pearsonr(pred_val.cpu().numpy(), y_val.cpu().numpy())
            if val_r > best_val_r:
                best_val_r = val_r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds_all, d_comb = model(X_all)
        preds_train, _ = model(X_train)
        preds_val, _ = model(X_val)

    preds_all_np = preds_all.cpu().numpy()
    preds_train_np = preds_train.cpu().numpy()
    preds_val_np = preds_val.cpu().numpy()

    # Per distortion breakdown
    distortions = sorted(list(df['distortion_type'].unique()))
    per_dist = {}
    for d in distortions:
        mask = df['distortion_type'] == d
        per_dist[d] = evaluate_metric(preds_all_np[mask], df.loc[mask, 'gt_jod'].values)

    return {
        "name": name,
        "features": feature_cols,
        "all": evaluate_metric(preds_all_np, df['gt_jod'].values),
        "train": evaluate_metric(preds_train_np, train_df['gt_jod'].values),
        "val": evaluate_metric(preds_val_np, val_df['gt_jod'].values),
        "per_distortion": per_dist,
        "weights": [float(w) for w in model.weights.detach().cpu().numpy()],
        "preds": preds_all_np,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablate MT Speed Tuning on GAIM-240")
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--val_scenes", nargs="+", default=DEFAULT_VAL_SCENES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load Data
    exp08_dir = os.path.join(repo_root, "experiments", "exp08_canonical_v1mt_full_resolution")
    v1mt_csv = os.path.join(exp08_dir, "v1mt_gaim240_features_720x1280_1200f_240hz.csv")
    cvvdp_csv = os.path.join(exp08_dir, "cvvdp_gaim240_features_720x1280_1200f_240hz.csv")

    v1_df = pd.read_csv(v1mt_csv)
    cv_df = pd.read_csv(cvvdp_csv)

    merged = pd.merge(
        cv_df[['ref_vid_path', 'dist_vid_path', 'scene_name', 'distortion_type', 'distortion_level', 'gt_jod', 'cvvdp_jod']],
        v1_df[['ref_vid_path', 'dist_vid_path', 'd_v1mt', 'mae_slow', 'mae_med', 'mae_fast', 'mae_fa', 'mae_frg', 'mae_fby']],
        on=['ref_vid_path', 'dist_vid_path']
    )
    merged['d_cvvdp'] = np.maximum(0.0, 10.0 - merged['cvvdp_jod'])

    # 2. Define Ablation Configurations
    configurations = [
        # --- Baseline ---
        ("CVVDP Baseline", ["d_cvvdp"]),
        
        # --- MT Standalone Speed Tuning Ablations ---
        ("MT Slow Only (1.0 deg/s)", ["mae_slow"]),
        ("MT Med Only (4.0 deg/s)", ["mae_med"]),
        ("MT Fast Only (16.0 deg/s)", ["mae_fast"]),
        ("MT Slow + Med (1.0 + 4.0 deg/s)", ["mae_slow", "mae_med"]),
        ("MT Slow + Fast (1.0 + 16.0 deg/s)", ["mae_slow", "mae_fast"]),
        ("MT Med + Fast (4.0 + 16.0 deg/s)", ["mae_med", "mae_fast"]),
        ("MT 3-Speeds (1.0 + 4.0 + 16.0 deg/s)", ["mae_slow", "mae_med", "mae_fast"]),
        
        # --- Form Standalone ---
        ("DKL Form Only (A + RG + BY)", ["mae_fa", "mae_frg", "mae_fby"]),
        
        # --- Full Cortical V1-MT Standalone ---
        ("Canonical V1-MT (3-Speed MT + Form)", ["mae_slow", "mae_med", "mae_fast", "mae_fa", "mae_frg", "mae_fby"]),
        
        # --- Hybrid Combinations with CVVDP ---
        ("CVVDP + MT Slow Only", ["d_cvvdp", "mae_slow"]),
        ("CVVDP + MT Med Only", ["d_cvvdp", "mae_med"]),
        ("CVVDP + MT Fast Only", ["d_cvvdp", "mae_fast"]),
        ("CVVDP + MT 3-Speeds (Motion Only)", ["d_cvvdp", "mae_slow", "mae_med", "mae_fast"]),
        ("CVVDP + DKL Form Only", ["d_cvvdp", "mae_fa", "mae_frg", "mae_fby"]),
        ("CVVDP + MT 3-Speeds + Form (Full Canonical Hybrid)", ["d_cvvdp", "mae_slow", "mae_med", "mae_fast", "mae_fa", "mae_frg", "mae_fby"]),
    ]

    print("=" * 90)
    print("EXPERIMENT 09: MT SPEED TUNING & PATHWAY ABLATION STUDY")
    print("=" * 90)

    results = []
    for name, cols in configurations:
        res = fit_and_evaluate_configuration(name, cols, merged, args.val_scenes)
        results.append(res)
        print(f"{name:<52} | All r: {res['all']['plcc']:+.4f} | Val r: {res['val']['plcc']:+.4f} | Train r: {res['train']['plcc']:+.4f}")

    # 3. Save Summary Table
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Configuration": r["name"],
            "Features": ", ".join(r["features"]),
            "All PLCC": r["all"]["plcc"],
            "All SRCC": r["all"]["srcc"],
            "All RMSE": r["all"]["rmse"],
            "Train PLCC": r["train"]["plcc"],
            "Val PLCC": r["val"]["plcc"],
            "dlss_rr": r["per_distortion"]["dlss_rr"]["plcc"],
            "judder": r["per_distortion"]["judder"]["plcc"],
            "motion_noise": r["per_distortion"]["motion_noise"]["plcc"],
            "motion_resolution": r["per_distortion"]["motion_resolution"]["plcc"],
            "noise_colors": r["per_distortion"]["noise_colors"]["plcc"],
            "restir": r["per_distortion"]["restir"]["plcc"],
            "stutter": r["per_distortion"]["stutter"]["plcc"],
            "trm": r["per_distortion"]["temporal-resolution-multiplexing"]["plcc"],
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(args.output_dir, "speed_ablation_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved speed ablation summary to:\n  --> {out_csv}")

    # 4. Generate Visualization Charts
    # Chart 1: Standalone MT Speeds vs Form vs Combined
    plt.figure(figsize=(12, 5.5), dpi=300)
    mt_subset = [
        "MT Slow Only (1.0 deg/s)",
        "MT Med Only (4.0 deg/s)",
        "MT Fast Only (16.0 deg/s)",
        "MT Slow + Med (1.0 + 4.0 deg/s)",
        "MT Slow + Fast (1.0 + 16.0 deg/s)",
        "MT Med + Fast (4.0 + 16.0 deg/s)",
        "MT 3-Speeds (1.0 + 4.0 + 16.0 deg/s)",
        "DKL Form Only (A + RG + BY)",
        "Canonical V1-MT (3-Speed MT + Form)"
    ]
    sub_df = summary_df[summary_df["Configuration"].isin(mt_subset)].copy()
    x = np.arange(len(sub_df))
    width = 0.35

    plt.bar(x - width/2, sub_df["All PLCC"], width, label="All PLCC", color="#4C72B0", alpha=0.9)
    plt.bar(x + width/2, sub_df["Val PLCC"], width, label="Val PLCC", color="#55A868", alpha=0.9)
    plt.ylabel("Pearson Correlation (r)", fontsize=11)
    plt.title("Impact of MT Speed Tuning and Form Channels (Standalone Cortical Models)", fontsize=12)
    plt.xticks(x, [c.replace(" (", "\n(") for c in sub_df["Configuration"]], fontsize=8, rotation=25, ha="right")
    plt.ylim(0.0, 0.75)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper left")

    for i, (_, row) in enumerate(sub_df.iterrows()):
        plt.text(i - width/2, row["All PLCC"] + 0.015, f"{row['All PLCC']:.3f}", ha="center", fontsize=7.5, fontweight="bold")
        plt.text(i + width/2, row["Val PLCC"] + 0.015, f"{row['Val PLCC']:.3f}", ha="center", fontsize=7.5)

    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "mt_speed_standalone_ablation.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Chart 2: Hybrid CVVDP + MT Speeds vs Form vs Full
    plt.figure(figsize=(12, 5.5), dpi=300)
    hy_subset = [
        "CVVDP Baseline",
        "CVVDP + MT Slow Only",
        "CVVDP + MT Med Only",
        "CVVDP + MT Fast Only",
        "CVVDP + MT 3-Speeds (Motion Only)",
        "CVVDP + DKL Form Only",
        "CVVDP + MT 3-Speeds + Form (Full Canonical Hybrid)"
    ]
    hy_df = summary_df[summary_df["Configuration"].isin(hy_subset)].copy()
    x = np.arange(len(hy_df))

    plt.bar(x - width/2, hy_df["All PLCC"], width, label="All PLCC", color="#4C72B0", alpha=0.9)
    plt.bar(x + width/2, hy_df["Val PLCC"], width, label="Val PLCC", color="#C44E52", alpha=0.9)
    plt.ylabel("Pearson Correlation (r)", fontsize=11)
    plt.title("Progressive Gains in Hybrid Models: Adding MT Speeds vs Form", fontsize=12)
    plt.xticks(x, [c.replace(" (", "\n(") for c in hy_df["Configuration"]], fontsize=8.5, rotation=20, ha="right")
    plt.ylim(0.4, 0.72)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper left")

    for i, (_, row) in enumerate(hy_df.iterrows()):
        plt.text(i - width/2, row["All PLCC"] + 0.01, f"{row['All PLCC']:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(i + width/2, row["Val PLCC"] + 0.01, f"{row['Val PLCC']:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "hybrid_speed_ablation.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Chart 3: Judder & DLSS RR sensitivity across MT speeds
    plt.figure(figsize=(10, 5), dpi=300)
    speeds_names = ["Slow (1.0 deg/s)", "Med (4.0 deg/s)", "Fast (16.0 deg/s)", "3-Speeds Combined"]
    judder_vals = [
        summary_df.loc[summary_df["Configuration"] == "MT Slow Only (1.0 deg/s)", "judder"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Med Only (4.0 deg/s)", "judder"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Fast Only (16.0 deg/s)", "judder"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT 3-Speeds (1.0 + 4.0 + 16.0 deg/s)", "judder"].values[0],
    ]
    dlss_vals = [
        summary_df.loc[summary_df["Configuration"] == "MT Slow Only (1.0 deg/s)", "dlss_rr"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Med Only (4.0 deg/s)", "dlss_rr"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Fast Only (16.0 deg/s)", "dlss_rr"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT 3-Speeds (1.0 + 4.0 + 16.0 deg/s)", "dlss_rr"].values[0],
    ]
    restir_vals = [
        summary_df.loc[summary_df["Configuration"] == "MT Slow Only (1.0 deg/s)", "restir"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Med Only (4.0 deg/s)", "restir"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT Fast Only (16.0 deg/s)", "restir"].values[0],
        summary_df.loc[summary_df["Configuration"] == "MT 3-Speeds (1.0 + 4.0 + 16.0 deg/s)", "restir"].values[0],
    ]

    x = np.arange(len(speeds_names))
    w_bar = 0.25
    plt.bar(x - w_bar, judder_vals, w_bar, label="Judder (Temporal Pacing)", color="#4C72B0", alpha=0.9)
    plt.bar(x, dlss_vals, w_bar, label="DLSS RR (Temporal Reconstruction)", color="#55A868", alpha=0.9)
    plt.bar(x + w_bar, restir_vals, w_bar, label="ReSTIR (Reservoir Jitter)", color="#C44E52", alpha=0.9)

    plt.ylabel("PLCC Correlation (r)", fontsize=11)
    plt.title("Per-Distortion Sensitivity across MT Speed Channels", fontsize=12)
    plt.xticks(x, speeds_names, fontsize=10)
    plt.ylim(0.0, 0.8)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper left")

    for i in range(len(speeds_names)):
        plt.text(i - w_bar, judder_vals[i] + 0.015, f"{judder_vals[i]:.2f}", ha="center", fontsize=8)
        plt.text(i, dlss_vals[i] + 0.015, f"{dlss_vals[i]:.2f}", ha="center", fontsize=8)
        plt.text(i + w_bar, restir_vals[i] + 0.015, f"{restir_vals[i]:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "distortion_speed_sensitivity.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    print(f"\nGenerated ablation plots:")
    print(f"  - {plot1_path}")
    print(f"  - {plot2_path}")
    print(f"  - {plot3_path}")


if __name__ == "__main__":
    main()
