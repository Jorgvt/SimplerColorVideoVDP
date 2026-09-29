# Experiment 04: GAIM-240 Full-Resolution Quality Calibration & Evaluation

## 1. Hypothesis & Objective

### Objective
Extract quality predictions across all 216 video pairs in the GAIM-240 dataset at full native resolution ($1280\times 720$, 240 fps, full video sequence) for three distinct cortical visual quality model architectures:
1. **Baseline `ColorVideoVDP`**: Isotropic Laplacian Weber pyramid with temporal impulse response filtering and contrast masking.
2. **`OrientedColorVideoVDP`**: Orientation-selective Steerable Weber pyramid ($K=4$ orientations: $0^\circ, 45^\circ, 90^\circ, 135^\circ$) with cross-orientation masking and oblique effect modeling.
3. **`V1MTColorVideoVDP`**: Unified cortical architecture coupling V1 spatiotemporal steerable filtering with MT speed- and direction-tuned motion integration.

### Hypotheses
1. **Orientation Selectivity Hypothesis**: Steerable orientation decomposition will better distinguish directional high-frequency artifacts (such as temporal multiplexing and motion resolution scaling) compared to isotropic Laplacian filters.
2. **Bio-inspired Motion Integration Hypothesis**: Incorporating MT speed-tuned spatio-temporal pooling will improve prediction correlation on motion-specific distortions (judder, stutter, motion noise) that disrupt smooth cortical motion energy integration.
3. **Generalization Hypothesis**: After calibrating the mapping $JOD = f(Q; a, b) = 10.0 - a \cdot Q^b$ on the 7 training scenes (168 samples), the oriented and V1-MT models will achieve higher cross-scene generalization ($r_p$, $r_s$, and lower RMSE) on the unseen holdout validation scenes (`subway`, `zeroday`).

---

## 2. Experimental Setup

- **Dataset**: GAIM-240 (`/media/disk/vista/BBDD_video_image/GAIM240/`)
  - Full native resolution: $1280\times 720$
  - Full video duration: 1200 frames ($5.0\text{ s}$ at $240\text{ fps}$)
  - 216 pairs total (168 train pairs across 7 scenes, 48 validation pairs across 2 holdout scenes).
- **Scripts**:
  - [`extract_model_predictions.py`](file:///media/disk/users/vitojor/SimplerColorVideoVDP/experiments/exp04_gaim240_model_calibration/extract_model_predictions.py): Parameterized prediction extraction.
  - [`calibrate_and_evaluate.py`](file:///media/disk/users/vitojor/SimplerColorVideoVDP/experiments/exp04_gaim240_model_calibration/calibrate_and_evaluate.py): Calibration curve fitting and cross-validation analysis.
  - [`run_full_calibration.sh`](file:///media/disk/users/vitojor/SimplerColorVideoVDP/experiments/exp04_gaim240_model_calibration/run_full_calibration.sh): Automated pipeline runner.

---

## 3. Results & Comparative Analysis

*(Will be populated with calibration parameter fits, Pearson $r$, Spearman $\rho$, RMSE, and per-distortion breakdowns upon completion of model extractions)*

### Overall Model Comparison Table

| Model Architecture | Fitted Parameters $(a, b)$ | Validation $r_p$ | Validation $r_s$ | Validation RMSE | All Scenes $r_p$ | All Scenes $r_s$ | All Scenes RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ColorVideoVDP** | *pending* | | | | | | |
| **Oriented ColorVideoVDP** | *pending* | | | | | | |
| **V1-MT ColorVideoVDP** | *pending* | | | | | | |

---

## 4. Per-Distortion Breakdown

| Distortion Type | Baseline $r_p$ | Oriented $r_p$ | V1-MT $r_p$ |
| :--- | :---: | :---: | :---: |
| `dlss_rr` | | | |
| `judder` | | | |
| `motion_noise` | | | |
| `motion_resolution` | | | |
| `noise_colors` | | | |
| `restir` | | | |
| `stutter` | | | |
| `temporal-resolution-multiplexing` | | | |
