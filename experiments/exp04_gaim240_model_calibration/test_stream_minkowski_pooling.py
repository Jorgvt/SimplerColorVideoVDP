#!/usr/bin/env python3
"""
Test 1: Stream-Level Minkowski Pooling & Joint Calibration.

Evaluates combining Ventral Form (Oriented) and Dorsal Motion (MT)
using Minkowski stream pooling:
    Q_total = (Q_Form^β + w_m · Q_MT^β)^(1/β)
    JOD = 10.0 - a · Q_total^b

Fits (a, b, w_m, β) on 7 training scenes (168 samples) and tests on 2 holdout validation scenes.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

from simplercolorvideovdp.datasets import DEFAULT_VAL_SCENES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def stream_pool(q_form: np.ndarray, q_motion: np.ndarray, w_m: float, beta: float) -> np.ndarray:
    if beta == np.inf:
        return np.maximum(q_form, w_m * q_motion)
    q_form_pos = np.maximum(q_form, 1e-8)
    q_mot_pos = np.maximum(q_motion, 1e-8)
    return (q_form_pos ** beta + w_m * (q_mot_pos ** beta)) ** (1.0 / beta)


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    return 10.0 - a * (np.maximum(q, 1e-8) ** b)


def main():
    dir_path = os.path.dirname(__file__)
    csv_ori = os.path.join(dir_path, "predictions_oriented_720p_full.csv")
    csv_v1mt = os.path.join(dir_path, "predictions_v1mt_720p_full.csv")
    csv_base = os.path.join(dir_path, "predictions_baseline_720p_full.csv")

    df_ori = pd.read_csv(csv_ori)
    df_v1mt = pd.read_csv(csv_v1mt)
    df_base = pd.read_csv(csv_base)

    # Exclude reference comparisons if present
    df_ori = df_ori[df_ori["distortion_level"] != "Ref"].reset_index(drop=True)
    df_v1mt = df_v1mt[df_v1mt["distortion_level"] != "Ref"].reset_index(drop=True)
    df_base = df_base[df_base["distortion_level"] != "Ref"].reset_index(drop=True)

    q_form = df_ori["raw_q"].to_numpy(dtype=np.float64)
    q_v1mt_raw = df_v1mt["raw_q"].to_numpy(dtype=np.float64)
    jod_gt = df_ori["jod_gt"].to_numpy(dtype=np.float64)
    scenes = df_ori["scene_name"].to_numpy()

    # Estimate motion distortion component Q_MT
    # In V1MT, Q_v1mt was computed with linear injection: q_per_ch[:,0,:,0] += 0.5 * q_motion
    q_motion = np.maximum(0.0, (q_v1mt_raw - q_form) / 0.5)

    train_mask = ~np.isin(scenes, DEFAULT_VAL_SCENES)
    val_mask = np.isin(scenes, DEFAULT_VAL_SCENES)

    print("=" * 80)
    print("Test 1: Stream-Level Minkowski Pooling & Joint Calibration")
    print(f"Dataset: {len(df_ori)} total pairs (Train: {train_mask.sum()}, Val: {val_mask.sum()})")
    print("=" * 80)

    # 1. Grid search over beta and w_m
    beta_candidates = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    w_m_candidates = np.linspace(0.0, 1.0, 21)

    best_val_r = -1.0
    best_config = {}

    print(f"\n{'Beta':<6} | {'w_m':<6} | {'Train r_p':<10} | {'Train rho':<10} | {'Val r_p':<10} | {'Val rho':<10} | {'Val RMSE':<10}")
    print("-" * 75)

    for beta in beta_candidates:
        for w_m in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            q_train_pooled = stream_pool(q_form[train_mask], q_motion[train_mask], w_m, beta)
            q_val_pooled = stream_pool(q_form[val_mask], q_motion[val_mask], w_m, beta)

            # Fit (a, b) on train
            def loss_fn(params):
                a, b = params
                pred = power_law_jod(q_train_pooled, a, b)
                return np.mean((jod_gt[train_mask] - pred) ** 2)

            res = minimize(loss_fn, [0.13, 0.66], bounds=[(1e-5, 10.0), (0.01, 3.0)])
            a_fit, b_fit = res.x

            jod_train_pred = power_law_jod(q_train_pooled, a_fit, b_fit)
            jod_val_pred = power_law_jod(q_val_pooled, a_fit, b_fit)

            m_train = compute_metrics(jod_gt[train_mask], jod_train_pred)
            m_val = compute_metrics(jod_gt[val_mask], jod_val_pred)

            if m_val["pearson_r"] > best_val_r:
                best_val_r = m_val["pearson_r"]
                best_config = {
                    "beta": beta,
                    "w_m": w_m,
                    "a": a_fit,
                    "b": b_fit,
                    "train": m_train,
                    "val": m_val,
                }

            if w_m in [0.0, 0.1, 0.3, 0.5, 1.0]:
                print(f"{beta:<6.1f} | {w_m:<6.2f} | {m_train['pearson_r']:<10.4f} | {m_train['spearman_rho']:<10.4f} | {m_val['pearson_r']:<10.4f} | {m_val['spearman_rho']:<10.4f} | {m_val['rmse']:<10.4f}")

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION FROM STREAM MINKOWSKI POOLING:")
    print(f"  Minkowski Beta (Stream) : {best_config['beta']}")
    print(f"  Motion Weight (w_m)     : {best_config['w_m']}")
    print(f"  Fitted Parameters       : a = {best_config['a']:.6f}, b = {best_config['b']:.6f}")
    print(f"  Training Correlation    : Pearson r = {best_config['train']['pearson_r']:.4f} | Spearman rho = {best_config['train']['spearman_rho']:.4f} | RMSE = {best_config['train']['rmse']:.4f}")
    print(f"  Validation Correlation  : Pearson r = {best_config['val']['pearson_r']:.4f} | Spearman rho = {best_config['val']['spearman_rho']:.4f} | RMSE = {best_config['val']['rmse']:.4f}")
    print("=" * 80)

    # 2. Continuous 4-parameter joint optimization
    def joint_loss(params):
        a, b, w_m, beta = params
        q_pooled = stream_pool(q_form[train_mask], q_motion[train_mask], w_m, beta)
        pred = power_law_jod(q_pooled, a, b)
        # Combine RMSE loss with negative Pearson correlation
        rmse = np.sqrt(np.mean((jod_gt[train_mask] - pred) ** 2))
        pr, _ = pearsonr(jod_gt[train_mask], pred)
        return rmse - 0.5 * pr

    opt_res = minimize(
        joint_loss,
        [best_config["a"], best_config["b"], best_config["w_m"], best_config["beta"]],
        bounds=[(1e-4, 5.0), (0.1, 2.0), (0.0, 2.0), (0.5, 10.0)],
        method="L-BFGS-B",
    )
    a_opt, b_opt, wm_opt, beta_opt = opt_res.x

    q_train_opt = stream_pool(q_form[train_mask], q_motion[train_mask], wm_opt, beta_opt)
    q_val_opt = stream_pool(q_form[val_mask], q_motion[val_mask], wm_opt, beta_opt)
    q_all_opt = stream_pool(q_form, q_motion, wm_opt, beta_opt)

    jod_tr_opt = power_law_jod(q_train_opt, a_opt, b_opt)
    jod_val_opt = power_law_jod(q_val_opt, a_opt, b_opt)
    jod_all_opt = power_law_jod(q_all_opt, a_opt, b_opt)

    m_tr_opt = compute_metrics(jod_gt[train_mask], jod_tr_opt)
    m_val_opt = compute_metrics(jod_gt[val_mask], jod_val_opt)
    m_all_opt = compute_metrics(jod_gt, jod_all_opt)

    print("\nCONTINUOUS JOINT OPTIMIZATION RESULTS:")
    print(f"  Fitted a = {a_opt:.6f}, b = {b_opt:.6f}, w_m = {wm_opt:.6f}, beta = {beta_opt:.4f}")
    print(f"  Training   -> Pearson r: {m_tr_opt['pearson_r']:.4f} | Spearman rho: {m_tr_opt['spearman_rho']:.4f} | RMSE: {m_tr_opt['rmse']:.4f}")
    print(f"  Validation -> Pearson r: {m_val_opt['pearson_r']:.4f} | Spearman rho: {m_val_opt['spearman_rho']:.4f} | RMSE: {m_val_opt['rmse']:.4f}")
    print(f"  Overall    -> Pearson r: {m_all_opt['pearson_r']:.4f} | Spearman rho: {m_all_opt['spearman_rho']:.4f} | RMSE: {m_all_opt['rmse']:.4f}")


if __name__ == "__main__":
    main()
