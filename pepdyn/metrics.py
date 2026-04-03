from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Empty arrays passed to regression_metrics.")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }

