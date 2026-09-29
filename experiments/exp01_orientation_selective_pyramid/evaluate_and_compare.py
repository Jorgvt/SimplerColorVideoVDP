"""Comparison Benchmark: Isotropic SimplerColorVideoVDP vs. OrientedColorVideoVDP."""

import math
import time
import torch

from simplercolorvideovdp import ColorVideoVDP, OrientedColorVideoVDP


def run_grating_masking_experiment():
    print("=" * 75)
    print("EXPERIMENT 1: Cross-Orientation vs Parallel Masking Evaluation")
    print("=" * 75)

    h, w = 256, 256
    device = torch.device("cpu")

    iso_metric = ColorVideoVDP(ppd=60.0).to(device)
    ori_metric = OrientedColorVideoVDP(ppd=60.0, num_orientations=4, cross_orientation_weight=0.25).to(device)

    y_coords = torch.linspace(0, 16 * 2 * math.pi, h).unsqueeze(1).repeat(1, w)
    x_coords = torch.linspace(0, 16 * 2 * math.pi, w).unsqueeze(0).repeat(h, 1)

    # Reference carrier: Horizontal grating (variation along y)
    carrier = 0.5 + 0.25 * torch.sin(y_coords)
    ref_img = carrier.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    # Distortion 1: Parallel phase modulation (masked by carrier)
    test_parallel = (0.5 + 0.25 * torch.sin(y_coords + 0.2)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    # Distortion 2: Orthogonal added ripple (unmasked by carrier)
    test_ortho = (0.5 + 0.25 * torch.sin(y_coords) + 0.05 * torch.sin(x_coords)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    # Evaluate with Isotropic Model
    jod_iso_parallel = iso_metric(test_parallel, ref_img).item()
    jod_iso_ortho = iso_metric(test_ortho, ref_img).item()

    # Evaluate with Oriented Model
    jod_ori_parallel = ori_metric(test_parallel, ref_img).item()
    jod_ori_ortho = ori_metric(test_ortho, ref_img).item()

    print(f"Carrier: Horizontal grating (90° orientation)")
    print(f"\n--- Isotropic Model (Weber Laplacian Pyramid) ---")
    print(f"Parallel Perturbation JOD   : {jod_iso_parallel:.4f}")
    print(f"Orthogonal Perturbation JOD : {jod_iso_ortho:.4f}")
    print(f"Delta JOD (|Iso Diff|)      : {abs(jod_iso_parallel - jod_iso_ortho):.4f}")

    print(f"\n--- Orientation-Selective Model (Steerable Pyramid, K=4) ---")
    print(f"Parallel Perturbation JOD   : {jod_ori_parallel:.4f} (Masked -> higher JOD)")
    print(f"Orthogonal Perturbation JOD : {jod_ori_ortho:.4f} (Unmasked -> lower JOD / higher visibility)")
    print(f"Perceptual Separation       : {(jod_ori_parallel - jod_ori_ortho):.4f} JOD")
    print("=" * 75)


def run_benchmark_speed():
    print("\nEXPERIMENT 2: Throughput & Latency Benchmark (256x256 image pairs)")
    print("=" * 75)

    device = torch.device("cpu")
    iso_metric = ColorVideoVDP(ppd=60.0).to(device)
    ori_metric = OrientedColorVideoVDP(ppd=60.0, num_orientations=4).to(device)

    img_a = torch.rand((1, 3, 256, 256), dtype=torch.float32)
    img_b = torch.rand((1, 3, 256, 256), dtype=torch.float32)

    # Warmup
    iso_metric(img_a, img_b)
    ori_metric(img_a, img_b)

    iters = 10

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = iso_metric(img_a, img_b)
    t_iso = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = ori_metric(img_a, img_b)
    t_ori = (time.perf_counter() - t0) / iters

    print(f"Isotropic SimplerColorVideoVDP : {t_iso * 1000.0:.2f} ms / pair")
    print(f"OrientedColorVideoVDP (4 ori) : {t_ori * 1000.0:.2f} ms / pair")
    print("=" * 75)


if __name__ == "__main__":
    run_grating_masking_experiment()
    run_benchmark_speed()
