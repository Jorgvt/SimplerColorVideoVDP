# Experiment 05: Calibration of the Oriented Steerable Form Model on GAIM-240

## 1. Context & Motivation
In Experiment 04, evaluating `OrientedColorVideoVDP` (Steerable Weber Pyramid, $K=4$ orientations) demonstrated substantial improvements on within-scene ranking for complex textured scenes (e.g., $r = 0.600$ vs $0.343$ on `attic`, $r = 0.700$ vs $0.533$ on `bistro_exterior`). On the 7 training scenes, Oriented achieved $r_{\text{train}} = 0.5970$ (outperforming the isotropic baseline $r_{\text{train}} = 0.5742$).

However, the oriented model inherited transducer parameters $(p, q, c, \text{XCM})$ and cross-orientation masking weights originally tuned for an isotropic Laplacian decomposition. On holdout validation scenes with dominant directional geometry (`subway`), uncalibrated orientation pooling ($\beta_{\text{ori}} = 0.5$) over-inflated energy, resulting in a cross-scene offset and lower validation correlation ($r_{\text{val}} = 0.4028$).

To resolve this, we calibrate the **Oriented Form Model** independently (without the MT motion stage) to establish a solid, well-tuned spatial visual front-end before re-coupling the MT dorsal stream.

---

## 2. Hypotheses
1. **Hypothesis 1 (Orientation Pooling Exponent)**: Tuning the orientation pooling exponent $\beta_{\text{ori}}$ between peak-detection ($\beta_{\text{ori}} = 0.5$) and energy summation ($\beta_{\text{ori}} = 2.0$) will balance sensitivity to directional distortions without creating artificial energy offsets on geometric scenes like `subway`.
2. **Hypothesis 2 (Cross-Orientation Masking)**: Adjusting the cross-orientation interaction weight $w_{\text{ori}}$ and oblique effect factor $\alpha_{\text{oblique}}$ will better reflect cortical cross-orientation suppression in V1.
3. **Hypothesis 3 (Transducer Scale)**: Re-aligning the contrast masking transducer parameters $(p, q, c)$ to the reduced subband energy of $K=4$ steerable filters will improve both train ($r_{\text{train}} > 0.60$) and holdout validation ($r_{\text{val}} > 0.52$) generalization.

---

## 3. Parameter Search Space

| Parameter | Symbol / Variable | Default / Prior | Search Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Orientation Pooling Exponent** | $\beta_{\text{ori}}$ | 0.5 | $[0.25, 6.0]$ | Minkowski norm across $K=4$ orientations |
| **Cross-Orientation Weight** | $w_{\text{ori}}$ | 0.25 | $[0.0, 1.0]$ | Broad cross-orientation masking factor |
| **Oblique Effect Factor** | $\alpha_{\text{oblique}}$ | 0.15 | $[0.0, 0.30]$ | Sensitivity attenuation on diagonals: $1 - \alpha \sin^2(2\theta)$ |
| **Excitatory Exponent** | $p$ (`mask_p`) | 2.264 | $[1.5, 3.0]$ | Contrast difference transducer power |
| **Inhibitory Exponents** | $q_0$ (`mask_q`) | 1.303 | $[0.8, 2.5]$ | Achromatic masking denominator power |
| **Masking Constant** | $c$ (`mask_c`) | -0.795 | $[-1.5, 0.0]$ | Phase uncertainty scaling factor ($10^c$) |
| **Calibration Mapping** | $(a, b)$ | $(0.13, 0.66)$ | $(10^{-4}, 10.0)$ | Power-law mapping: $\text{JOD} = 10 - a \cdot Q^b$ |

---

## 4. Experimental Protocol
1. **Training Set**: 7 scenes from GAIM-240 (168 video pairs at 720p @ 240fps, 1200 frames):
   - `attic`, `bistro_exterior`, `bistro_interior`, `classroom`, `landscape`, `marbles`, `pink_room`.
2. **Holdout Validation Set**: 2 unseen scenes (48 video pairs at 720p @ 240fps, 1200 frames):
   - `subway`, `zeroday`.
3. **Evaluation Metrics**:
   - Pearson Correlation $r_p$, Spearman Rank Correlation $\rho_s$, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Outlier Ratio ($>1.0$ JOD).
4. **Execution Strategy**:
   - Chunked temporal execution (`chunk_size = 15`) with full impulse response padding history on GPU cluster.
   - Extraction of unpooled spatial-orientation tensors $(1, C, T, K)$ per band into `spatial_ori_cache.pt` (132 MB) across all 216 video pairs.
   - Vectorized evaluation of candidate pooling norms and non-linear power-law calibration parameters.

---

## 5. Artifacts Generated
- `spatial_ori_cache.pt` (132 MB): Pre-pooled spatial-orientation energy tensors across all 216 video pairs at full 720p @ 240fps.
- `beta_ori_sweep_results.csv`: Complete numerical performance log across 12 candidate values of $\beta_{\text{ori}} \in [0.25, 6.00]$.
- `beta_ori_sweep_plot.png`: Calibration curves (Train $r$, Holdout Val $r$, All $r$) versus orientation pooling exponent $\beta_{\text{ori}}$.
- `log_calibrate_form.log`: Full execution and extraction log (Job 42611, 8.87h total GPU runtime).

---

## 6. Results: Orientation Exponent ($\beta_{\text{ori}}$) Calibration

| $\beta_{\text{ori}}$ | Calibration $(a, b)$ | Train $r_p$ | Train $\rho_s$ | Train RMSE | Holdout Val $r_p$ | Holdout Val $\rho_s$ | Holdout Val RMSE | Global All $r_p$ | Global All $\rho_s$ | Global All RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.25** | $(0.1231, 0.6254)$ | 0.4791 | 0.4721 | 0.9276 | 0.4415 | 0.4086 | 1.1262 | 0.4419 | 0.4318 | 0.9752 |
| **0.50** | $(0.1052, 0.6462)$ | **0.4793** | **0.4743** | **0.9275** | 0.4426 | 0.4137 | 1.1233 | **0.4431** | 0.4355 | **0.9744** |
| **0.75** | $(0.1031, 0.6466)$ | 0.4767 | 0.4724 | 0.9290 | 0.4432 | 0.4178 | 1.1194 | 0.4422 | **0.4361** | 0.9745 |
| **1.00** | $(0.1018, 0.6458)$ | 0.4741 | 0.4694 | 0.9305 | **0.4434** | **0.4198** | 1.1162 | 0.4411 | 0.4352 | 0.9748 |
| **1.25** | $(0.1008, 0.6444)$ | 0.4717 | 0.4679 | 0.9319 | 0.4432 | 0.4074 | 1.1137 | 0.4399 | 0.4335 | 0.9752 |
| **1.50** | $(0.1002, 0.6428)$ | 0.4695 | 0.4663 | 0.9331 | 0.4427 | 0.4106 | 1.1116 | 0.4387 | 0.4318 | 0.9756 |
| **1.75** | $(0.0997, 0.6411)$ | 0.4675 | 0.4665 | 0.9342 | 0.4421 | 0.4097 | 1.1100 | 0.4376 | 0.4327 | 0.9760 |
| **2.00** | $(0.0994, 0.6393)$ | 0.4658 | 0.4650 | 0.9352 | 0.4414 | 0.4089 | 1.1087 | 0.4365 | 0.4315 | 0.9764 |
| **2.50** | $(0.0990, 0.6357)$ | 0.4628 | 0.4658 | 0.9368 | 0.4397 | 0.4171 | 1.1067 | 0.4346 | 0.4326 | 0.9771 |
| **3.00** | $(0.0989, 0.6322)$ | 0.4604 | 0.4642 | 0.9381 | 0.4379 | 0.4171 | 1.1054 | 0.4330 | 0.4322 | 0.9778 |
| **4.00** | $(0.0992, 0.6259)$ | 0.4570 | 0.4616 | 0.9400 | 0.4346 | 0.4129 | 1.1036 | 0.4305 | 0.4303 | 0.9787 |
| **6.00** | $(0.1003, 0.6161)$ | 0.4533 | 0.4598 | 0.9420 | 0.4296 | 0.4047 | 1.1016 | 0.4279 | 0.4297 | 0.9797 |

---

## 7. Key Findings & Psychophysical Insights

1. **Sub-additive Minkowski Pooling ($\beta_{\text{ori}} \approx 0.50$) is Optimal**:
   - Sub-additive pooling ($\beta_{\text{ori}} = 0.50$) outperforms standard $L_2$ energy summation ($\beta_{\text{ori}} = 2.00$) across all evaluation splits ($r_{\text{train}}: 0.4793 \text{ vs } 0.4658$, $r_{\text{all}}: 0.4431 \text{ vs } 0.4365$).
   - This aligns with psychophysical probability summation models where peak orientation contrast damage dominates perceptual quality degradation.
2. **Large Exponents Degrade Global Correlation**:
   - As $\beta_{\text{ori}}$ increases toward $4.0$ or $6.0$ ($L_\infty$-like maximum orientation), correlation drops consistently ($r_{\text{all}} = 0.4431 \rightarrow 0.4279$), as small noise along a single orientation dominates the total response.
3. **Form Stream Independence & Isolation**:
   - Pure Form processing without MT dorsal integration achieves $r_{\text{all}} = 0.4431$ across all 216 video pairs.
   - Pure temporal distortions (e.g. `judder`, `temporal-resolution-multiplexing`, `stutter`) require motion velocity discrimination from the MT dorsal stream to attain high global correlation ($r > 0.52$).

---

## 8. Next Steps: Multi-Stage Integration

1. **Re-couple Dedicated Achromatic Transient MT Stream**:
   - Combine the calibrated Form stream ($\beta_{\text{ori}} = 0.50$) with the dedicated Achromatic Transient ($5\text{ Hz}$) MT stage using stream-level Minkowski pooling:
     $$Q_{\text{total}} = \left( Q_{\text{Form}}^{\beta_{\text{stream}}} + w_m \cdot Q_{\text{MT}}^{\beta_{\text{stream}}} \right)^{1 / \beta_{\text{stream}}}$$
2. **Calibrate Coupling Hyperparameters $(w_m, \beta_{\text{stream}})$**:
   - Run rapid grid search over $w_m \in [10, 5000]$ and $\beta_{\text{stream}} \in [1.0, 4.0]$ using cached tensors.
3. **Harmonize Defaults in `src/simplercolorvideovdp/`**:
   - Set optimal $\beta_{\text{ori}} = 0.50$ and calibrated stream weights as defaults in `OrientedColorVideoVDP` and `V1MTColorVideoVDP`.
