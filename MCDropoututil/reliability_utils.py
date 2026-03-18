"""
Reliability analysis utilities.
Computes AUROC/AUPRC, correlations, and risk-coverage curves.
"""
from typing import Dict, Tuple

import numpy as np


def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC.

    Args:
        y_true: Binary labels (1 = failure, 0 = success)
        y_score: Continuous uncertainty score (higher = more likely failure)

    Returns:
        Dictionary with auroc and auprc
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if np.unique(y_true).size < 2:
        return {'auroc': float('nan'), 'auprc': float('nan')}

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return {'auroc': float(auroc), 'auprc': float(auprc)}


def compute_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with pearson and spearman
    """
    from scipy.stats import pearsonr, spearmanr

    if len(x) < 2:
        return {'pearson': float('nan'), 'spearman': float('nan')}

    pearson = pearsonr(x, y)[0]
    spearman = spearmanr(x, y)[0]
    return {'pearson': float(pearson), 'spearman': float(spearman)}


def compute_risk_coverage(uncertainty: np.ndarray, risk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute risk-coverage curve and AURC.

    Args:
        uncertainty: Uncertainty scores (higher = less confident)
        risk: Per-sample risk (e.g., 1 - dice_mean)

    Returns:
        coverage: Array of coverage values
        risk_curve: Array of risk values
        aurc: Area under risk-coverage curve
    """
    order = np.argsort(uncertainty)  # most confident first
    risk_sorted = risk[order]

    n = len(risk_sorted)
    coverage = np.arange(1, n + 1) / n
    risk_curve = np.cumsum(risk_sorted) / np.arange(1, n + 1)

    aurc = float(np.trapz(risk_curve, coverage))
    return coverage, risk_curve, aurc