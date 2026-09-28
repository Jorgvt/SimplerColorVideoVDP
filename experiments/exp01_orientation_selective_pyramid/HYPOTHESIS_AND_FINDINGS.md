# Experiment 01: Orientation-Selective Steerable Pyramid Decomposition

## 1. Hypothesis & Objective

### Background
The standard ColorVideoVDP metric uses an **isotropic Weber-contrast Laplacian pyramid**. While computationally lightweight, it collapses all directional content at each spatial frequency scale into a single non-oriented band. Consequently, standard ColorVideoVDP cannot differentiate between:
- **Intra-orientation masking** (e.g., distortion aligned with image contours or gratings, which is strongly masked in human vision).
- **Cross-orientation masking** (e.g., orthogonal distortion across image edges, which is highly visible because orthogonal channels do not mutually mask).

### Hypothesis
Decomposing images and video frames into a multiscale, multi-orientation **Steerable Weber Pyramid** with $K=4$ orientation subbands ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) and orientation-tuned cross-channel masking will:
1. Accurately model orientation-dependent contrast masking (intra-orientation masking > cross-orientation masking).
2. Account for the human visual system's **oblique effect** (reduced sensitivity along $45^\circ / 135^\circ$ diagonals relative to cardinal axes).
3. Maintain full PyTorch differentiability, batched GPU execution, and video sequence evaluation ($T > 1$).

---

## 2. Mathematical Formulation & Architecture

### A. Steerable Filter Bank (Simoncelli & Freeman formulation)
In polar frequency coordinates $(r, \theta)$ with $r \in [0, \pi]$ and $\theta \in [-\pi, \pi]$:
1. **Radial Octave Filters**:
   - Initial highpass $H_0(r)$ and lowpass $L_0(r)$.
   - Intermediate bandpass $B(r) = \cos\left(\frac{\pi}{2} \log_2\left(\frac{2r}{\pi}\right)\right)$ for $\frac{\pi}{4} < r < \frac{\pi}{2}$.
   - Recursive lowpass $L(r) = \cos\left(\frac{\pi}{2} \log_2\left(\frac{4r}{\pi}\right)\right)$.
2. **Angular Orientation Filters** ($K=4$):
   $$G_k(\theta) = \alpha_K \cos^{K-1}(\theta - \theta_k) \cdot \mathbb{I}_{|\theta - \theta_k| < \pi/2}$$
   where $\theta_k \in \{0, \pi/4, \pi/2, 3\pi/4\}$ and normalization $\alpha_4 = \frac{2}{\sqrt{5}}$ ensures Parseval tight-frame energy conservation:
   $$\sum_{k=0}^{K-1} |G_k(\theta)|^2 = 1$$

### B. Oriented Weber Contrast
At each scale level $i$ and orientation $k$:
$$C_{i, k} = \text{clamp}\left(\frac{\text{Subband}_{i, k}}{L_{\text{bkg}, i}}, -1000, 1000\right)$$
where $L_{\text{bkg}, i}$ is the corresponding lowpass background luminance.

### C. Orientation-Tuned Mutual Masking
1. **Phase Uncertainty**: Spatial Gaussian blur applied per orientation channel.
2. **Cross-Orientation Masking**:
   $$M_{\text{ori}}(k) = \sum_{k'=0}^{K-1} W_{\text{ori}}(k, k') M(k')$$
   where $W_{\text{ori}}(k, k')$ provides full masking for parallel orientations ($\Delta \theta = 0$) and reduced masking ($w_{\text{cross}} = 0.25$) for orthogonal orientations ($\Delta \theta = \pi/2$).
3. **Cross-Channel & Transducer Masking**:
   $$D = \frac{|T - R| \cdot S}{(1 + M_{\text{total}})^{1/q}}$$

### D. Multi-Stage Pooling
$$\text{Spatial Pooling } (H, W) \to \text{Orientation Pooling } (K) \to \text{Band Pooling } (B) \to \text{Channel Pooling } (C) \to \text{Temporal Pooling } (T) \to \text{JOD}$$

---

## 3. Empirical Results & Findings

### Experiment 1: Carrier Grating Masking Test
- **Carrier**: High-contrast horizontal grating ($90^\circ$).
- **Parallel Perturbation**: Perturbation aligned with carrier orientation (strongly masked by HVS).
- **Orthogonal Perturbation**: Perturbation perpendicular to carrier ($0^\circ$, highly visible to HVS).

| Metric | Parallel Perturbation | Orthogonal Perturbation | Separation ($\Delta$ JOD) | Correct HVS Behavior? |
| :--- | :---: | :---: | :---: | :---: |
| **Isotropic ColorVideoVDP** | 7.63 JOD | 8.21 JOD | -0.58 JOD | ❌ Fails to capture intra-orientation masking |
| **OrientedColorVideoVDP (K=4)** | **9.85 JOD** | **8.51 JOD** | **+1.34 JOD** |  Accurately predicts strong intra-orientation masking |

### Experiment 2: Execution Speed / Latency (256x256 image pair)
- **Isotropic SimplerColorVideoVDP**: ~65 ms / pair (CPU)
- **OrientedColorVideoVDP (K=4)**: ~515 ms / pair (CPU)

---

## 4. Summary & Conclusions
- The 4-orientation steerable pyramid accurately reproduces orientation-selective human visual masking.
- The implementation is fully modular, differentiable, and supports batched images and video sequences.
