# ColorVideoVDP Codebase Analysis & Findings

## 1. Executive Summary

`ColorVideoVDP` (cvvdp) is a full-reference visual quality metric predicting perceptual differences (in Just Objectionable Differences, JOD) between pairs of images or videos. It models human visual system (HVS) mechanisms including:
1. Photometric display calibration (EOTF, peak luminance, black level, ambient reflection).
2. Color space transformation to opponent cone spaces ($XYZ \to LMS_{2006} \to DKL_{d65}$).
3. Spatio-temporal decomposition (temporal filter bank + Weber contrast Laplacian pyramid).
4. Contrast Sensitivity Function (CastleCSF interpolation across spatial/temporal frequency and background luminance).
5. Cross-channel contrast masking with phase uncertainty.
6. Multi-stage Minkowski pooling (spatial, band, channel, and temporal).
7. Non-linear mapping to JOD perceptual scores.

However, the original codebase in `./ColorVideoVDP` is burdened with significant boilerplate, legacy dead code, research ablations, monolithic file structures, and ad-hoc I/O abstractions.

---

## 2. Bloatware & Over-Generalization Analysis

### A. Monolithic and Entangled Files
- **`pycvvdp/cvvdp_metric.py` (1,264 lines)**: Combines model configuration loading, training/checkpoint code, complex sliding-window frame buffers with low-level tensor slicing and rolling, GPU memory estimation heuristics with NVML, distogram plotting with matplotlib, diff map export, debug dumping, and multiple alternative masking models.
- **`pycvvdp/display_model.py` (628 lines)**: Intermixes VR headset geometry, eccentricity calculations, angular resolution magnification, and display photometry with disk JSON lookups.
- **`pycvvdp/lpyr_dec.py` (540 lines)**: Contains 4 separate pyramid implementations (`lpyr_dec`, `lpyr_dec_2`, `weber_contrast_pyr`, `log_contrast_pyr`) with redundant code, legacy debug code, and commented-out profiling blocks.
- **`pycvvdp/cvvdp_ml_metric.py` (1,900 lines)**: Experimental ML variants (`cvvdp_ml_saliency`, `cvvdp_ml_transformer`) and HuggingFace Hub downloads that are completely separate from the core metric.

### B. Unused & Legacy Code Paths (Calibrated Model v0.5.7)
In the calibrated production model (`cvvdp_parameters.json` v0.5.7):
- **Masking model**: Uses solely `"mult-mutual"`. Dead code includes `"mult-none"`, `"add-transducer"`, `"mult-transducer"`, `"add-mutual"`, `"mult-mutual-old"`, `"add-similarity"`, `"mult-similarity"`, `"mult-transducer-texture"`, `"add-transducer-texture"`, `ce_overconstancy`, `transd_overconstancy`, `smooth_clamp_cont`.
- **Contrast type**: Uses solely `"weber_g1"`. Dead code includes `"weber_g0_ref"`, `"weber_g1_ref"`, `"log"`.
- **CSF version**: Uses solely `"weber_fixed_size"`. 6 unused JSON lookup tables exist in `vvdp_data`.
- **Temporal filter**: Uses solely `"default"`. Dead code includes `"hp_trans"`, `"grad_trans"`, and `"Bloch_int"`.
- **Ablation hooks**: `block_channels` and other research switches.
- **External utilities**: MATLAB loaders (`loadmat.py`), custom CPU info parser (`cpuinfo.py`), FOV dots experiments.

### C. Over-Engineered Video/Array Source Abstraction
- Complex hierarchy: `video_source` -> `video_source_filter` -> `video_source_dm` -> `video_source_array` / `video_source_packed_array` / `video_source_file` / `video_source_yuv_file`.
- Manual frame-by-frame loading and sliding window buffer maintenance (`cvvdp_frame_buffers`, `sw_buf`, `ra_buf`) with complex padding edge cases.
- For pure PyTorch tensor workflows, this creates excessive overhead, fragmented memory copies, and hard-to-read code.

---

## 3. Mathematical Processing Pipeline (Exact Details)

To ensure 100% numerical identity with the original model, the re-implementation must strictly preserve the following pipeline:

### 1. Photometry & Color Transform
1. Convert input pixel values $V \in [0, 1]$ to absolute linear luminance $L$ (cd/m²):
   $$L = (Y_{peak} - Y_{black}) \cdot \text{srgb2lin}(V) + Y_{black} + Y_{refl}$$
   where $Y_{black} = Y_{peak}/\text{contrast}$ and $Y_{refl} = E_{ambient} \cdot k_{refl} / \pi$.
2. Convert linear RGB to DKL color space via $XYZ$ and $LMS_{2006}$:
   $$M_{DKL} = M_{LMS\to DKL} \cdot M_{XYZ\to LMS} \cdot M_{RGB\to XYZ}$$
   producing sustained channels: $Y$ (achromatic), $RG$, $YV$.

### 2. Temporal Filtering (for Videos, $T > 1$)
1. Construct 4 temporal frequency filters $R(\omega)$ via real FFT impulse responses for:
   - Sustained channels (Ach, RG, YV): $R_{0..2}(\omega) = \exp(-\omega^{\beta_{tf}} / \sigma_{tf})$
   - Transient channel (Ach): $R_3(\omega) = \exp(-(\omega^{\beta_{tf}} - 5^{\beta_{tf}})^2 / \sigma_{tf})$
2. Apply 1D temporal convolution with replicate/symmetric padding to obtain 4 spatio-temporal channels for Test and Reference.
3. For single images ($T = 1$), temporal channels = 1 (sustained only, 3 channels total: Ach, RG, YV).

### 3. Spatial Decomposition (Weber Laplacian Pyramid)
1. Decompose into $K$ spatial frequency bands using 5-tap Gaussian kernel ($a = 0.4$):
   $$K = \text{clip}(\text{first index where } \rho \le 0.2, 0, \text{max\_levels}) + 1$$
2. Bandpass contrast at level $i$:
   $$C_i = \text{clamp}\left(\frac{G_i - \text{expand}(G_{i+1})}{L_{bkg, i}}, \max=1000\right)$$
   where $L_{bkg, i} = \text{clamp}(\text{expand}(G_{i+1})_{ref, sustained}, \min=0.01)$.
3. Baseband:
   $$C_{base} = \frac{G_{base}}{L_{bkg, base}}, \quad L_{bkg, base} = \text{mean}(G_{base, ref}, \text{dim}=[-2, -1])$$
4. Frequency per band: $\rho_0 = \text{ppd}/2$, $\rho_i = 0.3228 \cdot 2^{-(i-1)} \cdot \text{ppd}/2$, baseband = $0.1$.

### 4. Contrast Sensitivity Function (CastleCSF)
1. Interpolate CSF LUT over $\log_{10}(\rho)$ and background luminance $\log_{10}(L_{bkg})$ for temporal frequencies $\omega \in \{0, 5\}$ Hz.
2. Apply sensitivity correction: $S = S_{LUT} \cdot 10^{\text{sens\_corr}/20}$.

### 5. Contrast Masking (`mult-mutual`)
1. Perceptually normalized contrasts:
   $$T_p = T \cdot S \cdot \text{gain}_{ch}, \quad R_p = R \cdot S \cdot \text{gain}_{ch}$$
   $$\text{gain}_{ch} = [1.0, 1.45, 1.0, 1.0]$$
2. Mutual component: $M_{mm} = \min(|T_p|, |R_p|)$.
3. Phase uncertainty: Gaussian blur ($\sigma = 3$) on $M_{mm}$:
   $$M_{pu} = \text{GaussianBlur}(M_{mm}, \sigma=3) \cdot 10^{\text{mask\_c}}$$
4. Cross-channel pooling:
   $$M_i = \sum_{j} 2^{W_{ij}} \cdot (M_{pu, j})^{q_i}$$
5. Difference per band:
   $$D_u = \frac{|T_p - R_p|^p}{1 + M}$$
6. Soft clamping:
   $$D = \frac{10^{d_{max}} \cdot D_u}{10^{d_{max}} + D_u}$$
7. Baseband difference: $D_{base} = |T_{base} - R_{base}| \cdot S_{base}$.

### 6. Multi-Stage Pooling & JOD Mapping
1. Spatial pooling ($L_2$ norm over height $\times$ width):
   $$Q_{b} = \left(\frac{1}{HW} \sum_{x, y} D(x, y)^\beta\right)^{1/\beta} \quad (\beta=2)$$
2. Spatial band pooling ($L_4$ norm over bands, with baseband weights):
   $$Q_{sc} = \left(\sum_b (Q_b \cdot w_{sband, b})^{\beta_{sch}}\right)^{1/\beta_{sch}} \quad (\beta_{sch}=4)$$
3. Chromatic / temporal channel pooling ($L_4$ norm with channel weights):
   $$Q_{tc} = \left(\sum_{ch} (Q_{sc, ch} \cdot w_{ch})^{\beta_{tch}}\right)^{1/\beta_{tch}} \quad (\beta_{tch}=4)$$
4. Temporal pooling ($L_2$ norm over frames for video):
   $$Q = \left(\frac{1}{T} \sum_t Q_{tc}(t)^{\beta_t}\right)^{1/\beta_t} \quad (\beta_t=2)$$
   (For images: $Q = Q_{tc} \cdot t_{int}$ with $t_{int} = 0.5779$).
5. JOD transformation:
   $$Q_{JOD} = \begin{cases} 10 - a \cdot Q_t^{e-1} \cdot Q & \text{if } Q \le Q_t \\ 10 - a \cdot Q^e & \text{if } Q > Q_t \end{cases} \quad (Q_t=0.1, a=0.043957, e=0.930204)$$

---

## 4. Key Improvements for `SimplerColorVideoVDP`

1. **Pure PyTorch Architecture**: Clean `nn.Module` classes with vectorized batched execution.
2. **Eliminate All Dead Paths**: Hardcode calibrated parameters as crisp defaults while keeping flexibility for display configurations.
3. **Streamlined Data Flow**: Direct tensor input `(B, C, T, H, W)` or `(B, C, H, W)` without nested wrapper classes.
4. **Clean Project Management**: Modern `pyproject.toml` with `uv` lockfile and minimal clean dependencies (`torch`, `torchvision`, `numpy`, `scipy`).
5. **Exact Numerical Parity**: Automated differential testing suite against original `ColorVideoVDP` verifying tolerance $< 10^{-5}$.
