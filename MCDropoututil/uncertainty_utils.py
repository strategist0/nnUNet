"""
Uncertainty quantification utilities for MC Dropout.
Computes variance, entropy, and other uncertainty metrics.
"""
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from typing import Tuple


def compute_predictive_uncertainty(
    logits: np.ndarray,
    num_classes: int,
    return_probabilities: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predictive uncertainty metrics from MC Dropout samples.
    
    IMPORTANT: This function expects LOGITS as input, not probabilities.
    Uncertainty metrics (especially entropy) must be computed on probabilities.
    
    Args:
        logits: Array of shape (num_samples, C, H, W, D) - LOGITS from network output
        num_classes: Number of segmentation classes
        return_probabilities: If True, return mean probabilities; if False, return mean logits
        
    Returns:
        mean_pred: Mean prediction (C, H, W, D) - probabilities or logits based on flag
        variance: Variance map (H, W, D) - computed on probabilities
        entropy_map: Entropy map (H, W, D) - predictive entropy
    """
    # Convert logits to probabilities (REQUIRED for uncertainty metrics)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    # Compute mean prediction on probabilities
    mean_prob = np.mean(probabilities, axis=0)  # (C, H, W, D)
    
    # Compute variance across samples (on probabilities)
    # Average variance across classes for a single scalar per voxel
    variance = np.var(probabilities, axis=0).mean(axis=0)  # (H, W, D)
    
    # Compute predictive entropy (on mean probability)
    entropy_map = compute_entropy(mean_prob)  # (H, W, D)
    
    # Return mean logits or probabilities as requested
    if return_probabilities:
        mean_pred = mean_prob
    else:
        mean_pred = np.mean(logits, axis=0)
    
    return mean_pred, variance, entropy_map


def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute entropy from probability distribution.
    
    Args:
        probabilities: Array of shape (C, H, W, D) - probability per class
        
    Returns:
        Entropy map of shape (H, W, D)
    """
    # Transpose to (H, W, D, C) for easier computation
    probs = np.transpose(probabilities, (1, 2, 3, 0)).astype(np.float32, copy=False)

    # Compute entropy along class dimension with safe clamping
    epsilon = np.finfo(np.float32).eps
    np.clip(probs, epsilon, 1.0, out=probs)
    entropy_map = -np.sum(probs * np.log(probs), axis=-1)
    
    return entropy_map


def compute_mutual_information(logits: np.ndarray) -> np.ndarray:
    """
    Compute mutual information (epistemic uncertainty).
    MI = Total Entropy - Expected Data Entropy
    
    Args:
        logits: Array of shape (num_samples, C, H, W, D) - LOGITS from network output
        
    Returns:
        Mutual information map (H, W, D)
    """
    # Convert logits to probabilities (REQUIRED)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    # Total entropy (entropy of mean prediction)
    mean_prob = np.mean(probabilities, axis=0)
    total_entropy = compute_entropy(mean_prob)
    
    # Expected data entropy (mean of individual entropies)
    individual_entropies = np.array([compute_entropy(prob) for prob in probabilities])
    expected_entropy = np.mean(individual_entropies, axis=0)
    
    # Mutual information
    mi = total_entropy - expected_entropy
    
    return mi


def compute_coefficient_of_variation(logits: np.ndarray) -> np.ndarray:
    """
    Compute coefficient of variation (CV = std / mean) on probabilities.
    
    Args:
        logits: Array of shape (num_samples, C, H, W, D) - LOGITS from network output
        
    Returns:
        CV map (H, W, D)
    """
    # Convert to probabilities
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    mean = np.mean(probabilities, axis=0).mean(axis=0)  # (H, W, D)
    std = np.std(probabilities, axis=0).mean(axis=0)    # (H, W, D)
    
    epsilon = 1e-10
    cv = std / (mean + epsilon)
    
    return cv


def get_uncertainty_stats(uncertainty_map: np.ndarray) -> dict:
    """
    Get statistics of an uncertainty map.

    Args:
        uncertainty_map: Uncertainty values (H, W, D)

    Returns:
        Dictionary with statistics
    """
    flat = uncertainty_map.ravel()
    finite = flat[np.isfinite(flat)]

    if finite.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'q95': 0.0
        }

    return {
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'median': float(np.median(finite)),
        'q25': float(np.percentile(finite, 25)),
        'q75': float(np.percentile(finite, 75)),
        'q95': float(np.percentile(finite, 95))
    }


def aggregate_uncertainty_in_region(uncertainty_map: np.ndarray, mask: np.ndarray) -> dict:
    """
    Aggregate uncertainty metrics within a specific region.
    
    Args:
        uncertainty_map: Uncertainty values (H, W, D)
        mask: Binary mask defining the region (H, W, D)
        
    Returns:
        Dictionary with regional uncertainty statistics
    """
    masked_uncertainty = uncertainty_map[mask > 0]
    
    if len(masked_uncertainty) == 0:
        return {
            'mean': 0.0,
            'max': 0.0,
            'std': 0.0
        }
    
    return {
        'mean': float(np.mean(masked_uncertainty)),
        'max': float(np.max(masked_uncertainty)),
        'std': float(np.std(masked_uncertainty))
    }


def get_top_k_percent_uncertainty(uncertainty_map: np.ndarray, k: float = 10.0) -> float:
    """
    Get mean uncertainty of the top k% most uncertain voxels.
    
    Args:
        uncertainty_map: Uncertainty values (H, W, D)
        k: Percentage of top uncertain voxels (default: 10.0)
        
    Returns:
        Mean uncertainty of top k% voxels
    """
    flat_uncertainty = uncertainty_map.flatten()
    threshold_idx = int(len(flat_uncertainty) * (1 - k / 100.0))
    sorted_uncertainty = np.sort(flat_uncertainty)
    top_k_values = sorted_uncertainty[threshold_idx:]
    
    if len(top_k_values) == 0:
        return 0.0
    
    return float(np.mean(top_k_values))


def create_tumor_mask(segmentation: np.ndarray) -> np.ndarray:
    """
    Create binary mask for tumor region (nnU-Net re-encoding: labels 1, 2, 3).
    
    Args:
        segmentation: Segmentation array
        
    Returns:
        Binary tumor mask
    """
    return np.isin(segmentation, [1, 2, 3]).astype(np.uint8)

