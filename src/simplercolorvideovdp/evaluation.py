"""Evaluation and Cross-Validation utilities for ColorVideoVDP models.

Provides Leave-One-Scene-Out (LOSO), Leave-One-Distortion-Out (LODO),
and Group K-Fold cross-validation routines with confidence intervals and fold distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr, t


def power_law_jod(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Standard ColorVideoVDP JOD conversion power law: JOD = 10 - a * Q^b."""
    q_safe = np.maximum(q, 1e-8)
    return 10.0 - a * (q_safe ** b)


def fit_calibration_curve(
    q_train: np.ndarray,
    jod_train: np.ndarray,
    p0: Tuple[float, float] = (0.04395, 0.9302),
    bounds: Tuple[Sequence[float], Sequence[float]] = ([1e-6, 1e-3], [10.0, 5.0]),
    maxfev: int = 10000,
) -> Tuple[float, float]:
    """Fits parameters (a, b) using non-linear least squares on training samples."""
    try:
        popt, _ = curve_fit(
            power_law_jod,
            q_train,
            jod_train,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return p0[0], p0[1]


def compute_metrics(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
) -> Dict[str, float]:
    """Computes standard visual quality assessment metrics (PLCC, SROCC, KROCC, RMSE, MAE, Outlier Ratio)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) > 0 else float("nan"),
            "mae": float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else float("nan"),
            "outlier_ratio": float(np.mean(np.abs(y_true - y_pred) > 1.0) * 100) if len(y_true) > 0 else float("nan"),
        }

    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    outlier_ratio = float(np.mean(np.abs(y_true - y_pred) > 1.0)) * 100.0

    return {
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(sr), 4),
        "kendall_tau": round(float(kt), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "outlier_ratio": round(outlier_ratio, 2),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Computes non-parametric bootstrap confidence intervals for evaluation metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)

    if n < 5:
        nan_tup = (float("nan"), float("nan"))
        return {"pearson_r": nan_tup, "spearman_rho": nan_tup, "rmse": nan_tup, "mae": nan_tup}

    rng = np.random.default_rng(random_state)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))

    b_true = y_true[boot_indices]
    b_pred = y_pred[boot_indices]

    # 1. Pearson r (vectorized)
    mu_t = np.mean(b_true, axis=1, keepdims=True)
    mu_p = np.mean(b_pred, axis=1, keepdims=True)
    cov = np.mean((b_true - mu_t) * (b_pred - mu_p), axis=1)
    std_t = np.std(b_true, axis=1)
    std_p = np.std(b_pred, axis=1)
    denom = std_t * std_p
    b_plcc = np.where(denom > 1e-8, cov / denom, np.nan)

    # 2. RMSE & MAE (vectorized)
    diff = b_true - b_pred
    b_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
    b_mae = np.mean(np.abs(diff), axis=1)

    # 3. Spearman rho
    b_srocc = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        try:
            r_t = rankdata(b_true[i])
            r_p = rankdata(b_pred[i])
            sr, _ = spearmanr(r_t, r_p)
            b_srocc[i] = sr
        except Exception:
            b_srocc[i] = np.nan

    alpha_low = ((1.0 - ci) / 2.0) * 100.0
    alpha_high = (1.0 - (1.0 - ci) / 2.0) * 100.0

    def get_ci(arr: np.ndarray) -> Tuple[float, float]:
        clean = arr[~np.isnan(arr)]
        if len(clean) == 0:
            return (float("nan"), float("nan"))
        return (
            round(float(np.percentile(clean, alpha_low)), 4),
            round(float(np.percentile(clean, alpha_high)), 4),
        )

    return {
        "pearson_r": get_ci(b_plcc),
        "spearman_rho": get_ci(b_srocc),
        "rmse": get_ci(b_rmse),
        "mae": get_ci(b_mae),
    }


@dataclass
class FoldResult:
    fold_id: Union[int, str]
    test_group: str
    train_size: int
    test_size: int
    calib_a: float
    calib_b: float
    metrics: Dict[str, float]
    test_indices: List[int]
    y_true: List[float]
    y_pred: List[float]


@dataclass
class CrossValidationSummary:
    group_column: str
    n_folds: int
    oof_metrics: Dict[str, float]
    oof_ci_95: Dict[str, Tuple[float, float]]
    fold_mean_metrics: Dict[str, float]
    fold_std_metrics: Dict[str, float]
    fold_ci_95_metrics: Dict[str, Tuple[float, float]]
    fold_range_metrics: Dict[str, Tuple[float, float]]
    fold_results: List[FoldResult]
    oof_dataframe: pd.DataFrame

    def format_oof(self, metric: str = "pearson_r", decimals: int = 3) -> str:
        """Formats OOF metric estimate with 95% bootstrap confidence interval."""
        val = self.oof_metrics.get(metric, float("nan"))
        ci = self.oof_ci_95.get(metric, (float("nan"), float("nan")))
        return f"{val:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"

    def format_fold_mean(self, metric: str = "pearson_r", decimals: int = 3) -> str:
        """Formats fold average as Mean ± Std."""
        mean = self.fold_mean_metrics.get(metric, float("nan"))
        std = self.fold_std_metrics.get(metric, float("nan"))
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_column": self.group_column,
            "n_folds": self.n_folds,
            "oof_metrics": self.oof_metrics,
            "oof_ci_95": self.oof_ci_95,
            "fold_mean_metrics": self.fold_mean_metrics,
            "fold_std_metrics": self.fold_std_metrics,
            "fold_ci_95_metrics": self.fold_ci_95_metrics,
            "fold_range_metrics": self.fold_range_metrics,
            "per_fold": [
                {
                    "fold_id": f.fold_id,
                    "test_group": f.test_group,
                    "calib_a": f.calib_a,
                    "calib_b": f.calib_b,
                    "metrics": f.metrics,
                }
                for f in self.fold_results
            ],
        }


def cross_validate_by_group(
    df: pd.DataFrame,
    group_col: str,
    q_col: str = "raw_q",
    gt_col: str = "jod_gt",
    exclude_ref: bool = True,
    p0: Tuple[float, float] = (0.04395, 0.9302),
    n_bootstraps: int = 1000,
    ci: float = 0.95,
) -> CrossValidationSummary:
    """Performs Leave-One-Group-Out cross-validation across unique values in `group_col`.

    Args:
        df: Input DataFrame containing predictions and ground-truth scores.
        group_col: Column name to group by (e.g. 'scene_name' or 'distortion_type').
        q_col: Column containing raw uncalibrated Q metric values.
        gt_col: Column containing ground truth subjective JOD values.
        exclude_ref: Whether to filter out reference videos with distortion_level == 'Ref'.
        p0: Initial parameters for calibration power law fit.
        n_bootstraps: Number of bootstrap iterations for OOF confidence intervals.
        ci: Confidence interval level (default: 0.95 for 95% CI).

    Returns:
        CrossValidationSummary containing OOF metrics, 95% CIs, per-fold statistics, and OOF DataFrame.
    """
    df_eval = df.copy()
    if exclude_ref and "distortion_level" in df_eval.columns:
        df_eval = df_eval[df_eval["distortion_level"] != "Ref"].reset_index(drop=True)

    if group_col not in df_eval.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame columns: {list(df_eval.columns)}")

    groups = sorted(df_eval[group_col].dropna().unique())
    oof_predictions = np.zeros(len(df_eval), dtype=np.float64)
    fold_results: List[FoldResult] = []

    for fold_idx, test_group_val in enumerate(groups):
        test_mask = df_eval[group_col] == test_group_val
        train_mask = ~test_mask

        df_train = df_eval[train_mask]
        df_test = df_eval[test_mask]

        q_train = df_train[q_col].to_numpy(dtype=np.float64)
        jod_train = df_train[gt_col].to_numpy(dtype=np.float64)

        q_test = df_test[q_col].to_numpy(dtype=np.float64)
        jod_test = df_test[gt_col].to_numpy(dtype=np.float64)

        # Fit calibration mapping on training folds
        calib_a, calib_b = fit_calibration_curve(q_train, jod_train, p0=p0)

        # Predict on held-out test fold
        pred_test = power_law_jod(q_test, calib_a, calib_b)
        test_indices = df_eval.index[test_mask].tolist()
        oof_predictions[test_indices] = pred_test

        fold_m = compute_metrics(jod_test, pred_test)

        fold_results.append(
            FoldResult(
                fold_id=fold_idx + 1,
                test_group=str(test_group_val),
                train_size=int(train_mask.sum()),
                test_size=int(test_mask.sum()),
                calib_a=round(calib_a, 6),
                calib_b=round(calib_b, 6),
                metrics=fold_m,
                test_indices=test_indices,
                y_true=jod_test.tolist(),
                y_pred=pred_test.tolist(),
            )
        )

    # Attach Out-of-Fold predictions
    df_eval["jod_oof_pred"] = oof_predictions

    # Dataset-wide Out-Of-Fold metrics & Bootstrap CI
    y_all_gt = df_eval[gt_col].to_numpy(dtype=np.float64)
    oof_metrics = compute_metrics(y_all_gt, oof_predictions)
    oof_ci_95 = compute_bootstrap_ci(y_all_gt, oof_predictions, n_bootstraps=n_bootstraps, ci=ci)

    # Fold distribution statistics (Mean, Std, Student-t CI, Min/Max)
    metric_keys = ["pearson_r", "spearman_rho", "kendall_tau", "rmse", "mae", "outlier_ratio"]
    fold_mean_metrics: Dict[str, float] = {}
    fold_std_metrics: Dict[str, float] = {}
    fold_ci_95_metrics: Dict[str, Tuple[float, float]] = {}
    fold_range_metrics: Dict[str, Tuple[float, float]] = {}

    k = len(fold_results)
    t_crit = float(t.ppf(1.0 - (1.0 - ci) / 2.0, df=k - 1)) if k > 1 else 1.96

    for mk in metric_keys:
        vals = [f.metrics[mk] for f in fold_results if not np.isnan(f.metrics[mk])]
        if vals:
            m_val = float(np.mean(vals))
            s_val = float(np.std(vals, ddof=1 if len(vals) > 1 else 0))
            sem = s_val / np.sqrt(len(vals)) if len(vals) > 0 else 0.0

            fold_mean_metrics[mk] = round(m_val, 4)
            fold_std_metrics[mk] = round(s_val, 4)
            fold_ci_95_metrics[mk] = (
                round(float(m_val - t_crit * sem), 4),
                round(float(m_val + t_crit * sem), 4),
            )
            fold_range_metrics[mk] = (
                round(float(np.min(vals)), 4),
                round(float(np.max(vals)), 4),
            )
        else:
            fold_mean_metrics[mk] = float("nan")
            fold_std_metrics[mk] = float("nan")
            fold_ci_95_metrics[mk] = (float("nan"), float("nan"))
            fold_range_metrics[mk] = (float("nan"), float("nan"))

    return CrossValidationSummary(
        group_column=group_col,
        n_folds=k,
        oof_metrics=oof_metrics,
        oof_ci_95=oof_ci_95,
        fold_mean_metrics=fold_mean_metrics,
        fold_std_metrics=fold_std_metrics,
        fold_ci_95_metrics=fold_ci_95_metrics,
        fold_range_metrics=fold_range_metrics,
        fold_results=fold_results,
        oof_dataframe=df_eval,
    )


def leave_one_scene_out_cv(
    df: pd.DataFrame,
    scene_col: str = "scene_name",
    q_col: str = "raw_q",
    gt_col: str = "jod_gt",
    exclude_ref: bool = True,
    n_bootstraps: int = 1000,
) -> CrossValidationSummary:
    """Performs Leave-One-Scene-Out (LOSO) cross-validation to assess content generalization."""
    return cross_validate_by_group(
        df=df,
        group_col=scene_col,
        q_col=q_col,
        gt_col=gt_col,
        exclude_ref=exclude_ref,
        n_bootstraps=n_bootstraps,
    )


def leave_one_distortion_out_cv(
    df: pd.DataFrame,
    distortion_col: str = "distortion_type",
    q_col: str = "raw_q",
    gt_col: str = "jod_gt",
    exclude_ref: bool = True,
    n_bootstraps: int = 1000,
) -> CrossValidationSummary:
    """Performs Leave-One-Distortion-Out (LODO) cross-validation to assess distortion generalization."""
    return cross_validate_by_group(
        df=df,
        group_col=distortion_col,
        q_col=q_col,
        gt_col=gt_col,
        exclude_ref=exclude_ref,
        n_bootstraps=n_bootstraps,
    )
