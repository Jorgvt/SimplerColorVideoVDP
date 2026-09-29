#!/usr/bin/env python3
"""
Experiment 04: Extract Model Quality Predictions on GAIM-240.

Extracts predictions and uncalibrated contrast distortion metrics Q
for a specified model architecture across all video pairs in GAIM-240.

Supported models:
- baseline : ColorVideoVDP (Isotropic Laplacian Weber Pyramid)
- oriented : OrientedColorVideoVDP (Steerable Weber Pyramid, K=4)
- v1mt     : V1MTColorVideoVDP (Bio-inspired V1/MT spatiotemporal stage)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from simplercolorvideovdp.datasets import (
    DEFAULT_CSV_PATH,
    DEFAULT_DATA_DIR,
    GAIM240TorchDataset,
)
from simplercolorvideovdp.metric import (
    ColorVideoVDP,
    OrientedColorVideoVDP,
    V1MTColorVideoVDP,
)
from simplercolorvideovdp.pooling import lp_norm, metric_to_jod


def create_model(
    model_name: str,
    fps: float = 240.0,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Instantiates the requested model architecture."""
    name = model_name.lower().strip()
    if name in ("baseline", "colorvideovdp", "cvvdp"):
        model = ColorVideoVDP(display_name="standard_4k")
    elif name in ("oriented", "oriented_cvvdp", "steerable"):
        model = OrientedColorVideoVDP(
            display_name="standard_4k",
            num_orientations=4,
            cross_orientation_weight=0.25,
            enable_oblique_effect=True,
        )
    elif name in ("v1mt", "v1_mt", "v1mt_cvvdp"):
        model = V1MTColorVideoVDP(
            display_name="standard_4k",
            fps=fps,
            num_orientations=4,
            motion_weight=0.5,
        )
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: 'baseline', 'oriented', 'v1mt'."
        )

    model.to(device)
    model.eval()
    return model


def compute_model_quality_in_chunks(
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    fps: float = 240.0,
    chunk_size: int = 60,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Computes Q distortion metric and default JOD with sliding temporal chunking.

    Processes long sequences (e.g. 1200 frames @ 720p) in memory-safe chunks
    to avoid GPU out-of-memory errors, then pools across all frames.

    Args:
        model: Perceptual quality model instance.
        test_tensor: (T, C, H, W) or (1, C, T, H, W) test video tensor.
        ref_tensor: (T, C, H, W) or (1, C, T, H, W) reference video tensor.
        fps: Video native frame rate.
        chunk_size: Temporal chunk size in frames.
        device: Torch compute device.

    Returns:
        (raw_q_metric, default_jod)
    """
    # Normalize input layout to (1, C, T, H, W)
    if test_tensor.dim() == 4:
        if test_tensor.shape[1] == 3:  # (T, 3, H, W)
            test_t = test_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)
            ref_t = ref_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        else:  # (T, H, W, 3)
            test_t = test_tensor.permute(3, 0, 1, 2).unsqueeze(0)
            ref_t = ref_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    elif test_tensor.dim() == 5:
        test_t = test_tensor
        ref_t = ref_tensor
    else:
        raise ValueError(f"Unsupported tensor shape: {test_tensor.shape}")

    total_frames = test_t.shape[2]

    # Filter impulse response length (0.25s)
    pad_len = int(np.ceil(0.250 * fps / 2.0) * 2)

    # If sequence fits in single chunk without extreme memory, run direct forward pass
    if total_frames <= chunk_size:
        test_chunk = test_t.to(device)
        ref_chunk = ref_t.to(device)
        q_jod, stats = model(test_chunk, ref_chunk, fps=fps, return_stats=True)
        q_per_ch = torch.tensor(stats["Q_per_ch"], device=device)
        
        # Calculate pooled uncalibrated Q
        per_ch_w = model.pooling.ch_weights[:q_per_ch.shape[1]].view(1, -1, 1, 1)
        per_sband_w = torch.ones((1, q_per_ch.shape[1], 1, q_per_ch.shape[3]), device=device)
        per_sband_w[:, :, 0, -1] = model.pooling.baseband_weight[:q_per_ch.shape[1]]
        
        q_sc = lp_norm(q_per_ch * per_ch_w * per_sband_w, model.pooling.beta_sch, dim=3, normalize=False)
        q_tc = lp_norm(q_sc, model.pooling.beta_tch, dim=1, normalize=False)
        raw_q = lp_norm(q_tc, model.pooling.beta_t, dim=1, normalize=True).squeeze().item()
        jod_val = q_jod.squeeze().item() if isinstance(q_jod, torch.Tensor) else float(q_jod)
        return float(raw_q), float(jod_val)

    # Chunked temporal processing for long videos
    q_per_ch_list = []
    num_chunks = int(np.ceil(total_frames / chunk_size))

    for c_idx in range(num_chunks):
        start_f = c_idx * chunk_size
        end_f = min((c_idx + 1) * chunk_size, total_frames)

        # Include temporal padding history for continuous IIR/FIR filter state
        pad_start = max(0, start_f - pad_len)
        test_sub = test_t[:, :, pad_start:end_f, :, :].to(device)
        ref_sub = ref_t[:, :, pad_start:end_f, :, :].to(device)

        with torch.no_grad():
            _, stats = model(test_sub, ref_sub, fps=fps, return_stats=True)
            # Q_per_ch has shape (B, channels, sub_T, num_bands)
            q_chunk = torch.tensor(stats["Q_per_ch"], device=device)
            # Slice out the valid frames (excluding the prepend padding)
            offset = start_f - pad_start
            valid_len = end_f - start_f
            q_valid = q_chunk[:, :, offset : offset + valid_len, :]
            q_per_ch_list.append(q_valid)

    # Concatenate across time dimension
    q_per_ch_full = torch.cat(q_per_ch_list, dim=2)  # (1, C, total_frames, bands)

    # Pool across channels, bands, and time
    num_ch = q_per_ch_full.shape[1]
    num_bands = q_per_ch_full.shape[3]
    per_ch_w = model.pooling.ch_weights[:num_ch].view(1, num_ch, 1, 1)
    per_sband_w = torch.ones((1, num_ch, 1, num_bands), device=device)
    per_sband_w[:, :, 0, -1] = model.pooling.baseband_weight[:num_ch]

    q_sc = lp_norm(q_per_ch_full * per_ch_w * per_sband_w, model.pooling.beta_sch, dim=3, normalize=False)
    q_tc = lp_norm(q_sc, model.pooling.beta_tch, dim=1, normalize=False)
    raw_q = lp_norm(q_tc, model.pooling.beta_t, dim=1, normalize=True).squeeze().item()

    jod_tensor = metric_to_jod(torch.tensor([raw_q], device=device), model.pooling.jod_a.item(), model.pooling.jod_exp.item())
    jod_val = jod_tensor.item()

    return float(raw_q), float(jod_val)


def main():
    parser = argparse.ArgumentParser(description="Extract Model Predictions on GAIM-240")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "oriented", "v1mt"],
        help="Model architecture: baseline, oriented, or v1mt.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to GAIM-240 jod.csv metadata.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to GAIM-240 MP4 video files directory.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of 240Hz video frames to decode (default: None = full video, 1200 frames).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Spatial resolution height (default: 720 for native full resolution).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Spatial resolution width (default: 1280 for native full resolution).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=240.0,
        help="Native video frame rate (default: 240.0).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=60,
        help="Temporal chunk size in frames for memory-efficient forward passes.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="PyTorch DataLoader worker count for parallel FFmpeg decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda:0', 'cuda:1', 'cpu'). Auto-selects GPU if available.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file path to save extracted predictions.",
    )
    args = parser.parse_args()

    # Determine compute device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set default output CSV filename
    if args.output_csv is None:
        num_f_str = f"{args.num_frames}f" if args.num_frames is not None and args.num_frames > 0 else "full"
        res_str = f"{args.height}x{args.width}"
        output_filename = f"predictions_{args.model}_{res_str}_{num_f_str}.csv"
        args.output_csv = os.path.join(os.path.dirname(__file__), output_filename)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    print("=" * 80)
    print("GAIM-240 Perceptual Quality Model Prediction Extraction")
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Dataset CSV        : {args.csv_path}")
    print(f"Data Directory     : {args.data_dir}")
    print(f"Resolution         : {args.height}x{args.width}")
    print(f"Frames per sample  : {args.num_frames if args.num_frames else 'Full Video (1200 frames)'}")
    print(f"Native FPS         : {args.fps}")
    print(f"Chunk Size         : {args.chunk_size} frames")
    print(f"Workers            : {args.num_workers}")
    print(f"Device             : {device}")
    print(f"Output CSV         : {args.output_csv}")
    print("=" * 80)

    # 1. Initialize Model
    print(f"\nInitializing model '{args.model}' on {device}...")
    model = create_model(model_name=args.model, fps=args.fps, device=device)

    # 2. Load Dataset
    scale = (args.height, args.width) if (args.height > 0 and args.width > 0) else None
    dataset = GAIM240TorchDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        mode="all",  # All 216 comparisons
        num_frames=args.num_frames,
        start_frame=0,
        scale=scale,
        fps=args.fps,
        as_torch_tensor=True,
        tensor_layout="TCHW",
    )
    print(f"Loaded dataset: {len(dataset)} video pairs to evaluate.\n")

    # 3. Check for existing partial predictions to support safe resumption
    results = []
    completed_indices = set()
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            if len(existing_df) > 0 and "index" in existing_df.columns:
                results = existing_df.to_dict("records")
                completed_indices = set(existing_df["index"].tolist())
                print(f"Resuming from existing output: {len(completed_indices)}/{len(dataset)} already evaluated.")
        except Exception as e:
            print(f"Warning: Could not read existing output CSV ({e}). Starting fresh.")

    # 4. Evaluation Loop
    t_start_total = time.perf_counter()
    pbar = tqdm(range(len(dataset)), desc=f"Evaluating {args.model}")

    for idx in pbar:
        if idx in completed_indices:
            continue

        sample = dataset[idx]
        ref_tensor = sample["ref"]   # (T, 3, H, W)
        dist_tensor = sample["dist"]  # (T, 3, H, W)
        jod_gt = float(sample["jod"])
        meta = sample["meta"]

        t0 = time.perf_counter()
        with torch.no_grad():
            raw_q, default_jod = compute_model_quality_in_chunks(
                model=model,
                test_tensor=dist_tensor,
                ref_tensor=ref_tensor,
                fps=args.fps,
                chunk_size=args.chunk_size,
                device=device,
            )
        dt = time.perf_counter() - t0

        record = {
            "index": idx,
            "scene_name": meta["scene_name"],
            "distortion_type": meta["distortion_type"],
            "distortion_level": meta["distortion_level"],
            "dist_vid_path": meta["dist_vid_path"],
            "ref_vid_path": meta["ref_vid_path"],
            "jod_gt": jod_gt,
            "raw_q": raw_q,
            "default_jod": default_jod,
            "eval_time_sec": round(dt, 3),
        }
        results.append(record)

        # Periodically save incremental progress
        if len(results) % 5 == 0 or len(results) == len(dataset):
            df_out = pd.DataFrame(results).sort_values("index").reset_index(drop=True)
            df_out.to_csv(args.output_csv, index=False)

        pbar.set_postfix({
            "scene": meta["scene_name"][:6],
            "dist": meta["distortion_type"][:8],
            "Q": f"{raw_q:.3f}",
            "JOD": f"{default_jod:.2f}",
        })

    # Final Save
    df_out = pd.DataFrame(results).sort_values("index").reset_index(drop=True)
    df_out.to_csv(args.output_csv, index=False)

    total_time = time.perf_counter() - t_start_total
    print("\n" + "=" * 80)
    print(f"Extraction Complete for {args.model.upper()}!")
    print(f"Total samples processed : {len(df_out)}")
    print(f"Total duration           : {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Mean time per sample     : {df_out['eval_time_sec'].mean():.2f} s")
    print(f"Saved predictions to     : {args.output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
