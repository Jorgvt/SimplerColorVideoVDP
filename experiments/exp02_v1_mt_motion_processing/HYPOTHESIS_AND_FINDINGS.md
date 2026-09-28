# Experiment 02: Cortical V1-MT Motion Processing Stage

## 1. Hypothesis & Objective

### Background
Standard video quality metrics (such as ColorVideoVDP) rely on separable 1D temporal filtering and 2D spatial pyramids. However, the primate visual cortex processes motion via a specialized **dorsal stream** that couples space and time:
1. **Primary Visual Cortex (V1)**: 3D spatiotemporal receptive fields (Adelson-Bergen motion energy) tuned to local component directions and frequencies.
2. **Middle Temporal Area (MT / V5)**: Receptive fields ~10x larger that integrate V1 inputs along the **Simoncelli-Heeger velocity plane constraint**, resolving the **aperture problem** and forming true pattern-motion selectivity.

### Hypothesis
Implementing the bio-inspired **V1-MT cortical motion cascade** in PyTorch (based on the `v1mt` reference) and coupling it with the orientation-selective form stream will:
1. Accurately replicate biological motion perception phenomena (aperture problem resolution on plaid stimuli, pattern vs component cell tuning).
2. Provide dense 2D velocity flow decoding from population responses.
3. Enhance sensitivity to motion-specific artifacts (e.g. frame judder, motion jitter, and temporal inconsistency).

---

## 2. Mathematical Formulation & Architecture

### A. V1 Motion Energy & Divisive Normalization
- **Quadrature 3D Gabor Filters**:
  $$G(x, y, t) = \text{Envelope}(x', y', t') \cdot \exp\left(j 2\pi (f_x x + f_y y - f_t t)\right)$$
- **Adelson-Bergen Motion Energy**:
  $$E(x, y, t; \theta) = \left[ L_{\text{even}}(x, y, t; \theta) \right]^2 + \left[ L_{\text{odd}}(x, y, t; \theta) \right]^2$$
- **V1 Divisive Normalization**:
  $$R_{V1}(x, y, t; \theta) = \frac{E(x, y, t; \theta)}{\sigma_{V1}^2 + \text{AvgPool}\left(\frac{1}{N_{v1}}\sum_{\theta'} E(x, y, t; \theta')\right)}$$

### B. MT Velocity Plane Integration (Simoncelli & Heeger, 1998)
A pattern moving with velocity $\mathbf{v} = (v_x, v_y)$ satisfies the frequency plane equation:
$$f_t = v \cdot f_s \cos(\theta_{v1} - \theta_{MT})$$

The MT integration weight $W(\theta_{MT}, v_{MT}; \theta_{v1}, f_s, f_t)$ is:
$$W = \frac{1}{Z} \exp\left( -\frac{\left( f_t - v_{MT} \cdot f_s \cos(\theta_{v1} - \theta_{MT}) \right)^2}{2 \sigma_p^2} \right)$$

### C. MT Population Divisive Normalization
$$R_{MT}(\theta_{MT}) = \frac{\max\left(0, L_{MT}(\theta_{MT})\right)^2}{\sigma_{MT}^2 + \frac{1}{N_{MT}} \sum_{\theta'} \max\left(0, L_{MT}(\theta')\right)^2}$$

### D. Population Vector Velocity Readout
$$\mathbf{v}_{\text{flow}}(x, y, t) = \frac{\sum_k R_{MT}(x, y, t; k) \cdot \mathbf{v}_k}{\sum_k R_{MT}(x, y, t; k) + \epsilon}$$

---

## 3. Empirical Results & Findings

### Experiment 1: Plaid Aperture Problem Resolution (Simoncelli & Heeger 1998)
We presented a **Type I Plaid** composed of two sine gratings moving at $+45^\circ$ (Up-Right) and $-45^\circ$ (Down-Right) with component speeds $v / \sqrt{2}$:

| Direction | V1 Component Cell Energy | MT Pattern Cell Response |
| :--- | :---: | :---: |
| **$0^\circ$ (Right - Global Pattern)** | 1.595 | **2.471 (Peak - Pattern Motion)** |
| **$45^\circ$ (Up-Right Component)** | **1.721 (Component Peak)** | 1.428 |
| **$90^\circ$ (Up)** | 0.935 | 0.870 |
| **$180^\circ$ (Left)** | 0.434 | 0.095 |
| **$315^\circ$ (Down-Right Component)** | **1.720 (Component Peak)** | 1.428 |

- **V1 Response**: Bimodal tuning with peaks along the two individual 1D component directions ($45^\circ$ and $315^\circ$).
- **MT Response**: Unimodal tuning peaking cleanly at the **true physical pattern direction ($0^\circ$)**, resolving the aperture problem.
- **Decoded Velocity**: $v_x = 1.098\text{ deg/s}$, $v_y = 0.000\text{ deg/s}$.

---

## 4. Summary & Verification
- All 22 tests across standard ColorVideoVDP, steerable orientation pyramid, and V1-MT processing pass with 100% test coverage.
- The V1-MT module is fully modular, differentiable in PyTorch, and ready for integration.
