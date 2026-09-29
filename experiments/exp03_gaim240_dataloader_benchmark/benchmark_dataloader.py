#!/usr/bin/env python3
"""
Experiment 03: GAIM-240 Streaming DataLoader Benchmark.

Measures the throughput, latency, memory footprint, and full-epoch processing time
for loading GAIM-240 video pairs across different configurations:
- Multi-worker scaling (0, 2, 4, 8 workers)
- Resolutions (360x640 vs native 720x1280)
- Temporal chunk horizons (60 frames / 0.25s, 120 frames / 0.5s, 1200 frames / full 5s)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from tqdm import tqdm

from simplercolorvideovdp.datasets import (
    DEFAULT_CSV_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_VAL_SCENES,
    create_gaim240_dataloader,
)


def get_current_rss_mb() -> float:
    """Returns current process RSS memory in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb() -> float:
    """Returns peak process RSS memory in Megabytes across process lifetime."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / 1024.0


def benchmark_epoch(
    csv_path: str = DEFAULT_CSV_PATH,
    data_dir: str = DEFAULT_DATA_DIR,
    mode: str = "train",
    val_scenes: Optional[List[str]] = None,
    num_frames: Optional[int] = 60,
    start_frame: int = 0,
    height: int = 360,
    width: int = 640,
    fps: float = 240.0,
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    tensor_layout: str = "TCHW",
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Runs one full epoch over the GAIM-240 dataset and logs timing/memory metrics."""
    scale = (height, width) if (height > 0 and width > 0) else None

    initial_rss = get_current_rss_mb()

    loader = create_gaim240_dataloader(
        csv_path=csv_path,
        data_dir=data_dir,
        mode=mode,
        val_scenes=val_scenes if val_scenes is not None else DEFAULT_VAL_SCENES,
        num_frames=num_frames,
        start_frame=start_frame,
        scale=scale,
        fps=fps,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
        tensor_layout=tensor_layout,
    )

    num_samples = len(loader.dataset)
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    batch_latencies = []
    total_frames_processed = 0
    total_bytes_processed = 0

    t_start = time.perf_counter()

    iterator = enumerate(loader)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=f"Workers={num_workers} | {height}x{width} | T={num_frames}",
        )

    for batch_idx, batch in iterator:
        if max_batches is not None and batch_idx >= max_batches:
            break

        t_batch_start = time.perf_counter()

        ref_tensor: torch.Tensor = batch["ref"]
        dist_tensor: torch.Tensor = batch["dist"]
        jod_tensor: torch.Tensor = batch["jod"]

        # Ensure tensors are in memory
        bs = ref_tensor.shape[0]
        if tensor_layout == "TCHW":
            t_frames = ref_tensor.shape[1]
            c_dim = ref_tensor.shape[2]
            h_dim = ref_tensor.shape[3]
            w_dim = ref_tensor.shape[4]
        else:
            t_frames = ref_tensor.shape[1]
            h_dim = ref_tensor.shape[2]
            w_dim = ref_tensor.shape[3]
            c_dim = ref_tensor.shape[4]

        frames_in_batch = bs * t_frames * 2  # 2 videos per sample (ref and dist)
        bytes_in_batch = ref_tensor.element_size() * ref_tensor.nelement() + \
                         dist_tensor.element_size() * dist_tensor.nelement() + \
                         jod_tensor.element_size() * jod_tensor.nelement()

        total_frames_processed += frames_in_batch
        total_bytes_processed += bytes_in_batch

        t_batch_end = time.perf_counter()
        batch_latencies.append(t_batch_end - t_batch_start)

    t_end = time.perf_counter()
    total_duration = t_end - t_start
    peak_rss = get_peak_rss_mb()
    final_rss = get_current_rss_mb()

    samples_processed = len(batch_latencies) * batch_size
    samples_per_sec = samples_processed / total_duration if total_duration > 0 else 0
    frames_per_sec = total_frames_processed / total_duration if total_duration > 0 else 0
    mb_per_sec = (total_bytes_processed / (1024 * 1024)) / total_duration if total_duration > 0 else 0

    # Theoretical size if full dataset was kept in RAM at once
    pair_bytes = 2 * (t_frames * 3 * h_dim * w_dim * 4)  # 2 videos * float32
    theoretical_preloaded_mb = (num_samples * pair_bytes) / (1024 * 1024)

    # Projected full epoch time if max_batches was truncated
    projected_full_epoch_sec = (num_samples / samples_per_sec) if samples_per_sec > 0 else 0

    return {
        "mode": mode,
        "num_samples": num_samples,
        "samples_evaluated": samples_processed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "num_frames": t_frames,
        "resolution": f"{h_dim}x{w_dim}",
        "tensor_layout": tensor_layout,
        "total_duration_sec": round(total_duration, 3),
        "projected_full_epoch_sec": round(projected_full_epoch_sec, 2),
        "samples_per_sec": round(samples_per_sec, 2),
        "frames_per_sec": round(frames_per_sec, 1),
        "mb_per_sec": round(mb_per_sec, 2),
        "mean_batch_latency_ms": round(float(np.mean(batch_latencies) * 1000), 2) if batch_latencies else 0,
        "median_batch_latency_ms": round(float(np.median(batch_latencies) * 1000), 2) if batch_latencies else 0,
        "p95_batch_latency_ms": round(float(np.percentile(batch_latencies, 95) * 1000), 2) if batch_latencies else 0,
        "initial_rss_mb": round(initial_rss, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "final_rss_mb": round(final_rss, 2),
        "theoretical_preloaded_mb": round(theoretical_preloaded_mb, 2),
        "memory_saving_ratio": round(theoretical_preloaded_mb / max(peak_rss, 1.0), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="GAIM-240 PyTorch DataLoader Streaming Benchmark")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--num_frames", type=int, default=60, help="Temporal window size in frames")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers_list", nargs="+", type=int, default=[0, 2, 4, 8],
                        help="List of worker counts to benchmark")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer samples for smoke test")
    parser.add_argument("--output_json", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    max_batches = 10 if args.quick else None

    print("=" * 80)
    print("GAIM-240 Streaming DataLoader Performance Benchmark")
    print(f"Dataset CSV : {args.csv_path}")
    print(f"Data Dir    : {args.data_dir}")
    print(f"Mode        : {args.mode}")
    print(f"Resolution  : {args.height}x{args.width}")
    print(f"Frames/clip : {args.num_frames} frames (native 240 fps)")
    print(f"Batch Size  : {args.batch_size}")
    print("=" * 80)

    results = []
    for w in args.workers_list:
        print(f"\n--- Testing num_workers = {w} ---")
        res = benchmark_epoch(
            csv_path=args.csv_path,
            data_dir=args.data_dir,
            mode=args.mode,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            num_workers=w,
            prefetch_factor=args.prefetch_factor,
            max_batches=max_batches,
            show_progress=True,
        )
        results.append(res)
        print(f"Result for num_workers={w}:")
        print(f"  Total Epoch Time  : {res['total_duration_sec']:.2f} s")
        print(f"  Throughput        : {res['samples_per_sec']:.2f} video pairs/s ({res['frames_per_sec']:.1f} frames/s, {res['mb_per_sec']:.1f} MB/s)")
        print(f"  Mean Batch Latency: {res['mean_batch_latency_ms']:.2f} ms (p95: {res['p95_batch_latency_ms']:.2f} ms)")
        print(f"  Peak RAM (RSS)    : {res['peak_rss_mb']:.1f} MB (vs ~{res['theoretical_preloaded_mb']:.1f} MB monolithic)")

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), args.output_json)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
