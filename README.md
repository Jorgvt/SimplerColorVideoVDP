# SimplerColorVideoVDP

A streamlined, modular, and boilerplate-free PyTorch re-implementation of **ColorVideoVDP** (cvvdp).

`SimplerColorVideoVDP` strips out legacy research dead code, training hooks, custom I/O wrappers, and multiple redundant branches while maintaining **100% exact numerical parity** with the calibrated production ColorVideoVDP model.

---

## Key Features

- **Exact Numerical Parity**: Tested against the original `ColorVideoVDP` across image, video, batch, and display configurations ($|Q_{\text{new}} - Q_{\text{orig}}| < 10^{-5}$ JOD).
- **Pure PyTorch Architecture**: Standard `nn.Module` classes accepting native PyTorch tensors `(B, C, H, W)` or `(B, C, T, H, W)`.
- **Zero Boilerplate**: No ad-hoc file wrappers, NVML GPU heuristics, or obsolete masking/pyramid branches.
- **Fast & Differentiable**: Supports end-to-end backpropagation via `.loss(test, ref)`.
- **Managed with `uv`**: Fast, modern package management.

---

## Installation

Using `uv`:
```bash
uv add torch torchvision numpy scipy
# Install simplercolorvideovdp in editable mode:
uv pip install -e .
```

---

## Quickstart

### 1. Image Quality Assessment

```python
import torch
from simplercolorvideovdp import ColorVideoVDP

# Initialize metric with standard 4K monitor viewing conditions (default)
metric = ColorVideoVDP(display_name="standard_4k")

# Input tensors: shape (B, 3, H, W) or (3, H, W), float32 in [0, 1]
ref_img = torch.rand((1, 3, 512, 512))
test_img = (ref_img + torch.randn_like(ref_img) * 0.02).clamp(0.0, 1.0)

# Predict quality score in JOD (Just Objectionable Differences)
# 10 JOD = identical / imperceptible difference
# 9-10 JOD = threshold difference
# <6 JOD = severe distortion
q_jod = metric(test_img, ref_img)
print(f"Predicted Quality: {q_jod.item():.3f} JOD")
```

### 2. Video Quality Assessment

```python
# Video tensors: shape (B, 3, T, H, W) where T is frame count
ref_video = torch.rand((1, 3, 30, 256, 256))
test_video = (ref_video + torch.randn_like(ref_video) * 0.02).clamp(0.0, 1.0)

# Specify frame rate (fps) for video temporal filtering
q_jod = metric(test_video, ref_video, fps=30.0)
print(f"Video Quality: {q_jod.item():.3f} JOD")
```

### 3. Batched Evaluation

```python
# Batched image pairs: shape (B, 3, H, W)
ref_batch = torch.rand((4, 3, 128, 128))
test_batch = (ref_batch + torch.randn_like(ref_batch) * 0.03).clamp(0.0, 1.0)

q_batch = metric(test_batch, ref_batch)
print(f"Batch Scores: {q_batch.tolist()}")
```

### 4. Differentiable Optimization Loss

```python
test_param = torch.rand((1, 3, 128, 128), requires_grad=True)
ref_target = torch.rand((1, 3, 128, 128))

# Loss is (10.0 - Q_jod)
loss = metric.loss(test_param, ref_target)
loss.backward()
print(f"Gradient shape: {test_param.grad.shape}")
```

### 5. Custom Display Configurations

```python
from simplercolorvideovdp import ColorVideoVDP, DisplayPhotometry, DisplayGeometry

# Custom photometry (e.g. 1000-nit HDR display in dark room)
photo = DisplayPhotometry(
    peak_luminance=1000.0,
    contrast=100000.0,
    eotf="PQ",
    e_ambient=0.0,
)

# Custom geometry (e.g. explicit 60 pixels-per-degree)
geom = DisplayGeometry(ppd=60.0)

custom_metric = ColorVideoVDP(photometry=photo, geometry=geom)
```

---

## Architecture Overview

```
src/simplercolorvideovdp/
├── metric.py         # Main ColorVideoVDP nn.Module
├── display.py        # Display photometry and geometry models
├── colorspace.py     # Linear RGB to DKLd65 transformations
├── temporal.py       # 1D temporal filter bank & convolution
├── pyramid.py        # Weber contrast Laplacian pyramid decomposition
├── csf.py            # Vectorized CastleCSF lookup table module
├── masking.py        # Mult-mutual cross-channel masking & phase uncertainty
├── pooling.py        # Minkowski spatial, band, channel, and temporal pooling
└── data/
    └── csf_lut_weber_fixed_size.json   # Calibrated CastleCSF LUT
```

For full codebase audit and design rationale:
- [FINDINGS.md](FINDINGS.md) - Analysis of original repo bloat and mathematical pipeline.
- [PLAN.md](PLAN.md) - Re-implementation and verification roadmap.

---

## Running Tests

```bash
uv run pytest -v
```
