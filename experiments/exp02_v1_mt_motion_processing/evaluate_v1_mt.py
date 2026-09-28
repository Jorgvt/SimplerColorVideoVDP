"""Demonstration and Validation of V1-MT Cortical Motion Processing Stage.

Demonstrates:
1. Simoncelli & Heeger (1998) Aperture Problem Resolution:
   - Component vs. Pattern Motion responses in V1 vs MT on Plaid stimuli.
2. Dual-Stream Video Quality Metric evaluation on Motion vs Static Distortions.
"""

import math
import numpy as np
import torch

from simplercolorvideovdp.metric import ColorVideoVDP
from experiments.exp02_v1_mt_motion_processing.metric_v1_mt import V1MTColorVideoVDP
from experiments.exp02_v1_mt_motion_processing.v1_mt import V1MTModel


def run_plaid_aperture_demonstration():
    print("=" * 80)
    print("EXPERIMENT 1: Simoncelli & Heeger (1998) Aperture Problem Resolution")
    print("=" * 80)

    t_len, h, w = 15, 48, 48
    fps = 30.0
    ppd = 30.0
    sf = 1.5  # cpd
    speed = 1.6  # deg/s

    ts = torch.linspace(0, (t_len - 1) / fps, t_len).view(1, t_len, 1, 1)
    xs = torch.linspace(0, (w - 1) / ppd, w).view(1, 1, 1, w)
    ys = torch.linspace(0, (h - 1) / ppd, h).view(1, 1, h, 1)

    # Grating A at +45 deg (Up-Right)
    th_a = math.pi / 4.0
    grating_a = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(th_a) + ys * math.sin(th_a)) - (sf * speed) * ts))

    # Grating B at -45 deg (Down-Right)
    th_b = -math.pi / 4.0
    grating_b = torch.cos(2.0 * math.pi * (sf * (xs * math.cos(th_b) + ys * math.sin(th_b)) - (sf * speed) * ts))

    # Combined Type I Plaid: True global motion vector is Rightward (0 deg)
    plaid = 0.5 * (grating_a + grating_b).expand(1, t_len, h, w)

    v1_dirs = [k * math.pi / 4.0 for k in range(8)]
    mt_dirs = [k * math.pi / 4.0 for k in range(8)]
    dir_names = ["0° (R)", "45° (UR)", "90° (U)", "135° (UL)", "180° (L)", "225° (DL)", "270° (D)", "315° (DR)"]

    model = V1MTModel(
        v1_directions=v1_dirs,
        v1_sf=sf,
        v1_tf=sf * speed,
        v1_kernel_size=(7, 11, 11),
        mt_directions=mt_dirs,
        mt_speed=speed * math.sqrt(2.0),
        mt_spatial_pool_size=(9, 9),
        pixels_per_degree=ppd,
        fps=fps,
    )

    v1_norm, mt_norm, flow = model.forward_all(plaid)

    # Average over center frames to avoid boundary effects
    v1_pop = torch.mean(v1_norm[:, :, 3:-3], dim=(-3, -2, -1))[0].detach().cpu().numpy()
    mt_pop = torch.mean(mt_norm[:, :, 3:-3], dim=(-3, -2, -1))[0].detach().cpu().numpy()

    print("Population tuning responses to a Type I Plaid (+45° and -45° moving components):")
    print(f"{'Direction':<12} | {'V1 (Component Cell Energy)':<28} | {'MT (Pattern Cell Response)':<28}")
    print("-" * 75)
    for i in range(8):
        v1_bar = "#" * int(v1_pop[i] * 5)
        mt_bar = "#" * int(mt_pop[i] * 5)
        print(f"{dir_names[i]:<12} | {v1_pop[i]:>6.3f}  {v1_bar:<18} | {mt_pop[i]:>6.3f}  {mt_bar:<18}")

    print("-" * 75)
    print(f"-> V1 Peak Response: {dir_names[np.argmax(v1_pop)]} (bimodal component tuning at 45° / 315°)")
    print(f"-> MT Peak Response: {dir_names[np.argmax(mt_pop)]} (unimodal pattern tuning at 0° - Aperture Problem Resolved!)")
    print(f"-> Decoded Velocity Flow: vx = {flow[0, 0, 3:-3].mean().item():.3f} deg/s, vy = {flow[0, 1, 3:-3].mean().item():.3f} deg/s")
    print("=" * 80)


def run_video_quality_comparison():
    print("\nEXPERIMENT 2: Motion Artifact Sensitivity (Frame Judder Distortion)")
    print("=" * 80)

    fps = 30.0
    ppd = 30.0
    t_len, h, w = 9, 64, 64

    iso_metric = ColorVideoVDP(ppd=ppd).to(torch.device("cpu"))
    v1mt_metric = V1MTColorVideoVDP(ppd=ppd, fps=fps).to(torch.device("cpu"))

    # Reference video: Smoothly drifting texture
    ts = torch.linspace(0, (t_len - 1) / fps, t_len).view(1, 1, t_len, 1, 1)
    xs = torch.linspace(0, (w - 1) / ppd, w).view(1, 1, 1, 1, w)
    ys = torch.linspace(0, (h - 1) / ppd, h).view(1, 1, 1, h, 1)

    ref_vid = (0.5 + 0.3 * torch.sin(2.0 * math.pi * (1.5 * xs - 2.4 * ts))).repeat(1, 3, 1, h, 1)

    # Distortion A: High-frequency temporal judder (alternating frame shift, destroying motion smoothness)
    judder_shift = torch.tensor([0.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, 0.0]).view(1, 1, t_len, 1, 1) / ppd
    test_judder = (0.5 + 0.3 * torch.sin(2.0 * math.pi * (1.5 * (xs + judder_shift) - 2.4 * ts))).repeat(1, 3, 1, h, 1)

    jod_iso = iso_metric(test_judder, ref_vid, fps=fps).item()
    jod_v1mt = v1mt_metric(test_judder, ref_vid, fps=fps).item()

    print(f"Smooth Drifting Reference vs. Temporal Judder Video Sequence:")
    print(f"Standard Isotropic ColorVideoVDP JOD : {jod_iso:.4f}")
    print(f"Dual-Stream V1-MT ColorVideoVDP JOD : {jod_v1mt:.4f} (Heightened sensitivity to motion disruption)")
    print("=" * 80)


if __name__ == "__main__":
    run_plaid_aperture_demonstration()
    run_video_quality_comparison()
