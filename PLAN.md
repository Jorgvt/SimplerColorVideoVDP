# Re-Implementation Plan for SimplerColorVideoVDP

## 1. Objectives

1. **Remove Boilerplate & Over-Generalizations**: Strip out all unused experimental masking models, unused contrast metrics, training gradient hacks, ad-hoc file format wrappers, NVML GPU heuristics, and redundant pyramid implementations.
2. **Exact Processing & Numerical Equivalence**: Ensure bit-for-bit / high-precision floating point equivalence ($|Q_{new} - Q_{orig}| < 10^{-5}$) against the original `ColorVideoVDP` across images, videos, batches, SDR/HDR configurations, and custom viewing geometries.
3. **Idiomatic, Clean PyTorch**: Implement modular, typed, well-documented `torch.nn.Module` components with clear forward passes.
4. **Project Management with `uv`**: Initialize modern `pyproject.toml`, lockfile, and virtual environment using `uv`.
5. **Comprehensive Differential Testing**: Implement automated test suites comparing all intermediate stages (color transform, temporal filter, Laplacian bands, CSF sensitivity, masked differences, pooled JODs) against original `ColorVideoVDP`.

---

## 2. Target Architecture

```
SimplerColorVideoVDP/
├── pyproject.toml              # Modern packaging managed with uv
├── uv.lock                     # uv lockfile
├── FINDINGS.md                 # Detailed audit of original repo
├── PLAN.md                     # This plan
├── simpler_cvvdp/
│   ├── __init__.py             # Clean public API
│   ├── metric.py               # Main SimplerColorVideoVDP nn.Module (clean forward pass)
│   ├── display.py              # Photometry & Geometry models (sRGB, PQ, custom display)
│   ├── colorspace.py           # sRGB -> Linear -> LMS2006 -> DKLd65 transformations
│   ├── temporal.py             # Temporal filter bank & 1D convolution
│   ├── pyramid.py              # Clean Weber Laplacian pyramid decomposition
│   ├── csf.py                  # CastleCSF interpolation module with embedded LUT
│   ├── masking.py              # Mult-mutual cross-channel masking & phase uncertainty
│   ├── pooling.py              # Minkowski lp-norms & JOD mapping
│   └── data/
│       └── csf_lut_weber_fixed_size.json   # Calibrated CSF LUT
└── tests/
    ├── test_numerical_parity.py  # End-to-end and component parity tests against ColorVideoVDP
    ├── test_image_parity.py      # Image evaluation parity
    ├── test_video_parity.py      # Video evaluation parity
    └── test_batch_parity.py      # Batched evaluation parity
```

---

## 3. Implementation Phases

### Phase 1: Environment Setup with `uv`
- Initialize `uv` environment in `./SimplerColorVideoVDP`.
- Define `pyproject.toml` with pinned dependencies (`torch`, `torchvision`, `numpy`, `scipy`, `pytest`).
- Set up development environment linking both `ColorVideoVDP` and `SimplerColorVideoVDP` for direct differential testing.

### Phase 2: Core Components Implementation
1. **`colorspace.py` & `display.py`**:
   - Vectorized photometric conversions (sRGB, PQ, linear, gamma).
   - Clean matrix multiplication for $XYZ \to LMS_{2006} \to DKL_{d65}$.
   - Geometry helper for calculating pixels per degree (ppd) from distance/diagonal/resolution.
2. **`pyramid.py`**:
   - Streamlined Weber contrast Laplacian pyramid with separable 5-tap Gaussian reduction and expansion.
   - Clean handling of background luminance $L_{bkg}$.
3. **`csf.py`**:
   - Fast, vectorized 2D interpolation for CastleCSF over $\log_{10}(\rho)$ and $\log_{10}(L_{bkg})$.
4. **`temporal.py`**:
   - Frequency-domain filter synthesis via real iFFT.
   - Vectorized 1D temporal convolution across frames with replicate and symmetric padding.
5. **`masking.py` & `pooling.py`**:
   - Perceptual normalization, mutual masking, Gaussian phase uncertainty blur.
   - Cross-channel weighting matrix multiplication.
   - Hierarchical Minkowski pooling ($L_2$ spatial, $L_4$ band, $L_4$ channel, $L_2$ temporal) and JOD transfer function.

### Phase 3: Main Metric Class & Top-Level API
- Implement `ColorVideoVDP` (`simpler_cvvdp.ColorVideoVDP` / `simpler_cvvdp.cvvdp` alias).
- Accept standard `torch.Tensor` inputs: `(B, C, H, W)` for images and `(B, C, T, H, W)` for videos.
- Support both direct quality prediction (`metric(test, ref, fps=...)`) and loss function computation (`metric.loss(test, ref)`).

### Phase 4: Rigorous Differential Testing & Parity Verification
- Create automated test suite comparing outputs of `SimplerColorVideoVDP` against original `pycvvdp`:
  1. **Component-level unit tests**: Color transforms, pyramid bands, CSF values, masked differences.
  2. **End-to-end image tests**: Standard SDR images, HDR images, multiple resolutions and ppds.
  3. **End-to-end video tests**: Video clips with various frame rates, durations, and temporal dynamics.
  4. **Batch mode tests**: Multi-image and multi-video batches.
  5. **Device tests**: CPU and CUDA/MPS device verification.
- Enforce max absolute difference threshold $\le 10^{-5}$ JOD across all test cases.

### Phase 5: Documentation & Benchmark Report
- Update `README.md` with clean API documentation and usage examples.
- Document performance and memory benchmarks comparing `SimplerColorVideoVDP` vs `ColorVideoVDP`.
