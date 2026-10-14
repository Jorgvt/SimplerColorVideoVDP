# V1-MT Performance Discrepancy Analysis & Systematic Test Plan

## 1. Context & Motivation

In Experiment 04 on the GAIM-240 dataset ($720\text{p}$ @ $240\text{ fps}$, 216 video pairs), `V1MTColorVideoVDP` demonstrated strong performance improvements on specific motion-dominated distortions (notably **`judder` with $+27\%$ Pearson correlation improvement** over the isotropic baseline and significant recovery on `temporal-resolution-multiplexing`). 

However, its global correlation across all 9 scenes decayed relative to the isotropic baseline ($r_{\text{all}} = 0.4365$ vs $0.5429$) and the oriented model ($r_{\text{train}} = 0.5970$).

This document logs the root-cause hypotheses explaining this performance behavior and defines a systematic 4-phase testing plan.

---

## 2. Root Cause Hypotheses

### Hypothesis A: Channel Contamination via (Channel 0, Band 0) Injection
- **Mechanism**:
  ```python
  q_per_ch[:, 0, :, 0] = q_per_ch[:, 0, :, 0] + self.motion_weight * q_motion
  ```
- **Analysis**:
  - In `ColorVideoVDP`, `q_per_ch[:, 0, :, 0]` corresponds to **Channel 0 (Achromatic Sustained, 0 Hz)** at **Scale 0 (highest spatial frequency, ~30 cpd)**.
  - Biological MT neurons are tuned to **transient temporal frequencies ($\sim 5\text{ Hz}$)** and **low-to-mid spatial frequencies ($0.5\text{–}6\text{ cpd}$)**.
  - Injecting MT motion energy directly into the highest-frequency static band corrupts fine static edge masking, causing the model to over-penalize static scenes with high-frequency textures (e.g. `landsc`, `marble`).

---

### Hypothesis B: Static Leakage via Sustained Channel Input to MT
- **Mechanism**: Passing both Sustained (0 Hz) and Transient (5 Hz) into MT velocity filters.
- **Analysis**:
  - In static or stationary video scenes (`classroom`, `pink_room`), the Sustained channel has high contrast energy while Transient is zero.
  - Velocity filter tails at $\omega = 0\text{ Hz}$ leaked static texture noise into MT velocity responses.
  - Restricting MT input strictly to **Achromatic Transient ($5\text{ Hz}$)** guarantees that static scenes produce **identically zero MT motion distortion ($Q_{\text{MT}} \equiv 0$)**.

---

### Hypothesis C: Orientation Pooling Discrepancy ($\beta_{\text{ori}} = 0.5$ vs $2.0$)
- **Mechanism**: `OrientedColorVideoVDP` uses sub-additive orientation pooling ($\beta_{\text{ori}} = 0.5$) with oblique effect modeling, whereas earlier V1-MT forward implementations hardcoded $\beta_{\text{ori}} = 2.0$.
- **Analysis**:
  - Sub-additive orientation pooling ($\beta_{\text{ori}} = 0.5$) amplifies peak directional responses in the ventral form stream.
  - Harmonizing the form stream configuration ensures that Form processing retains its high train correlation ($r = 0.5970$) while MT contributes exclusively to motion-dependent dimensions.

---

## 3. Results of Test 1 & Dedicated Dorsal Stream Extraction

With the dedicated dorsal stream extraction completed on all 216 video pairs:
- **`raw_q_form`** (Ventral Form Stream) and **`raw_q_motion`** (Dorsal MT Motion Stream) are cleanly separated and saved side-by-side.
- Combining Form + Transient MT via stream Minkowski pooling ($Q = (Q_{\text{Form}}^2 + w_m \cdot Q_{\text{MT}}^2)^{1/2}$):
  - **`judder`**: $r = 0.3766$ (vs $0.3229$ baseline).
  - **`temporal-resolution-multiplexing`**: $r = 0.2167$ (vs $0.1763$ oriented).
  - **`noise_colors`**: $r = 0.7150$ (vs $0.7027$ oriented).
  - **`restir`**: $r = 0.5160$ (vs $0.5132$ oriented).
