#!/usr/bin/env python3
"""
Experiment 05: Calibration of Oriented Form Model on GAIM-240.

Performs parameter search and calibration on the 7 training scenes (168 video pairs)
and evaluates generalization on the 2 holdout validation scenes (48 video pairs).

Memory-safe chunked extraction caches spatial-orientation energy tensors (1, C, T, K)
per subband, enabling instantaneous orientation pooling sweeps and joint parameter optimization.
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
from tqdm import tqdm

from simplercolorvideovdp.datasets import (
    GAIM240TorchDataset,
    DEFAULT_VAL_SCENES,
    DEFAULT_CSV_PATH,
    DEFAULT_DATA_DIR,
)
from simplercolorvideovdp.metric import OrientedColorVideoVDP
from simplercolorvideovdp.pooling import MetricPooling, lp_norm


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD power-law mapping."""
    return 10.0 - a * (np.maximum(q, 1e-8) ** b)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive visual quality assessment evaluation metrics."""
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


def extract_spatial_ori_tensor_chunked(
    model: OrientedColorVideoVDP,
    test_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    fps: float = 240.0,
    chunk_size: int = 15,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Extracts per-band (1, C, T, K) spatial pooled energy tensors safely in chunks."""
    if test_tensor.dim() == 4:
        if test_tensor.shape[1] == 3:
            test_t = test_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            ref_t = ref_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            test_t = test_tensor.permute(3, 0, 1, 2).unsqueeze(0)
            ref_t = ref_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    elif test_tensor.dim() == 5:
        test_t, ref_t = test_tensor, ref_tensor
    else:
        raise ValueError(f"Unsupported tensor shape: {test_tensor.shape}")

    total_frames = test_t.shape[2]
    pad_len = min(15, max(4, int(np.ceil(0.0625 * fps))))

    num_chunks = int(np.ceil(total_frames / chunk_size))
    band_chunks: List[List[torch.Tensor]] = []

    for c_idx in range(num_chunks):
        start_f = c_idx * chunk_size
        end_f = min((c_idx + 1) * chunk_size, total_frames)

        pad_start = max(0, start_f - pad_len)
        test_sub = test_t[:, :, pad_start:end_f, :, :].to(device)
        ref_sub = ref_t[:, :, pad_start:end_f, :, :].to(device)

        with torch.no_grad():
            _, stats = model(test_sub, ref_sub, fps=fps, return_stats=True)
            q_ori_chunk = stats["Q_per_ori"]  # List of (1, C, chunk_T, K)

            offset = start_f - pad_start
            valid_len = end_f - start_f

            if len(band_chunks) == 0:
                band_chunks = [[] for _ in range(len(q_ori_chunk))]

            for bb, b_tensor in enumerate(q_ori_chunk):
                b_valid = b_tensor[:, :, offset : offset + valid_len, :].clone().cpu()
                band_chunks[bb].append(b_valid)

        del test_sub, ref_sub, stats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Concatenate chunks along time dimension for each band
    q_ori_full = [torch.cat(chunks, dim=2) for chunks in band_chunks]
    return q_ori_full


def main():
    parser = argparse.ArgumentParser(description="Calibrate Oriented Form Model on GAIM-240")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to evaluate (None for full 1200 frames)")
    parser.add_argument("--chunk_size", type=int, default=15, help="Temporal chunk size in frames")
    parser.add_argument("--output_dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--cache_path", type=str, default=os.path.join(os.path.dirname(__file__), "spatial_ori_cache.pt"))
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of cache")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # 1. Load dataset
    dataset = GAIM240TorchDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        mode="all",
        num_frames=args.max_frames,
        start_frame=0,
        scale=(720, 1280),
        fps=240.0,
        as_torch_tensor=True,
        tensor_layout="TCHW",
    )
    num_scenes = len(dataset.table["scene_name"].unique())
    print(f"Loaded dataset: {len(dataset)} video pairs across {num_scenes} scenes.")

    val_scenes = set(DEFAULT_VAL_SCENES)
    all_eval_indices = [i for i, row in dataset.table.iterrows() if str(row["distortion_level"]) != "Ref"]
    train_indices = [i for i in all_eval_indices if str(dataset.table.iloc[i]["scene_name"]) not in val_scenes]
    val_indices = [i for i in all_eval_indices if str(dataset.table.iloc[i]["scene_name"]) in val_scenes]

    print(f"Train pairs: {len(train_indices)}, Holdout Val pairs: {len(val_indices)}, Total eval pairs: {len(all_eval_indices)}")

    # 2. Extract or load spatial orientation energy cache
    if os.path.exists(args.cache_path) and not args.force_extract:
        print(f"Loading cached spatial orientation stats from: {args.cache_path}")
        cache_data = torch.load(args.cache_path, map_location="cpu")
        extracted_stats = cache_data["extracted_stats"]
        metadata_list = cache_data["metadata"]
    else:
        print("Extracting spatial orientation energy tensors across dataset...")
        metric = OrientedColorVideoVDP(
            num_orientations=4,
            cross_orientation_weight=0.25,
            enable_oblique_effect=True,
            oblique_weight=0.15,
            beta_ori=2.0,
        ).to(device)

        extracted_stats = []
        metadata_list = []

        start_t = time.time()
        for idx in tqdm(all_eval_indices, desc="Extracting Form Stats"):
            item = dataset[idx]
            dist_t = item["dist"]
            ref_t = item["ref"]
            meta = item["meta"]

            q_ori_full = extract_spatial_ori_tensor_chunked(
                metric,
                dist_t,
                ref_t,
                fps=240.0,
                chunk_size=args.chunk_size,
                device=device,
            )

            extracted_stats.append(q_ori_full)
            metadata_list.append({
                "index": idx,
                "scene_name": meta["scene_name"],
                "distortion_type": meta["distortion_type"],
                "distortion_level": meta["distortion_level"],
                "jod_gt": float(meta["jod"]),
            })

        print(f"Extraction completed in {time.time() - start_t:.1f}s.")
        torch.save({"extracted_stats": extracted_stats, "metadata": metadata_list}, args.cache_path)
        print(f"Saved cache to: {args.cache_path}")

    # 3. Parameter Sweep over beta_ori
    beta_ori_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]
    j_gt = np.array([m["jod_gt"] for m in metadata_list], dtype=np.float64)
    scenes = np.array([m["scene_name"] for m in metadata_list])
    train_mask = ~np.isin(scenes, list(val_scenes))
    val_mask = np.isin(scenes, list(val_scenes))

    pooling = MetricPooling()
    sweep_results = []
    print("\n" + "=" * 110)
    print("ORIENTATION POOLING EXPONENT (beta_ori) SWEEP")
    print("=" * 110)

    for beta in beta_ori_candidates:
        q_raw_list = []
        for q_ori in extracted_stats:
            num_bands = len(q_ori)
            b, c, t = q_ori[0].shape[0], q_ori[0].shape[1], q_ori[0].shape[2]
            q_per_ch = torch.empty((b, c, t, num_bands), dtype=torch.float32)

            for bb in range(num_bands):
                b_tensor = q_ori[bb]  # (1, C, T, K)
                k_dim = b_tensor.shape[-1]
                if k_dim > 1:
                    q_band = lp_norm(b_tensor, beta, dim=-1, normalize=True, keepdim=False)
                else:
                    q_band = b_tensor.squeeze(-1)
                q_per_ch[:, :, :, bb] = q_band

            raw_q = float(pooling.pool_raw_q(q_per_ch, is_image=False).squeeze().item())
            q_raw_list.append(raw_q)

        q_arr = np.array(q_raw_list, dtype=np.float64)

        # Fit calibration on training split
        q_tr, j_tr = q_arr[train_mask], j_gt[train_mask]
        q_val, j_val = q_arr[val_mask], j_gt[val_mask]

        a_fit, b_fit = fit_power_law(q_tr, j_tr)

        p_tr = power_law_jod(q_tr, a_fit, b_fit)
        p_val = power_law_jod(q_val, a_fit, b_fit)
        p_all = power_law_jod(q_arr, a_fit, b_fit)

        m_tr = compute_metrics(j_tr, p_tr)
        m_val = compute_metrics(j_val, p_val)
        m_all = compute_metrics(j_gt, p_all)

        sweep_results.append({
            "beta_ori": beta,
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

    df_sweep = pd.DataFrame(sweep_results)
    csv_out = os.path.join(args.output_dir, "beta_ori_sweep_results.csv")
    df_sweep.to_csv(csv_out, index=False)
    print("\n" + df_sweep.to_string(index=False))
    print(f"\nSaved sweep results to: {csv_out}")

    # Plot sweep curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_sweep["beta_ori"], df_sweep["train_r"], "b-o", linewidth=2, label="Train Pearson r")
    ax1.plot(df_sweep["beta_ori"], df_sweep["val_r"], "r-s", linewidth=2, label="Holdout Val Pearson r")
    ax1.plot(df_sweep["beta_ori"], df_sweep["all_r"], "k--^", linewidth=2, label="All Pearson r")
    ax1.set_xlabel(r"Orientation Pooling Exponent ($\beta_{\mathrm{ori}}$)", fontsize=12)
    ax1.set_ylabel("Pearson Correlation (r)", fontsize=12)
    ax1.set_title(r"GAIM-240 Calibration vs Orientation Exponent $\beta_{\mathrm{ori}}$", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plot_out = os.path.join(args.output_dir, "beta_ori_sweep_plot.png")
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"Saved sweep plot to: {plot_out}")


if __name__ == "__main__":
    main()
