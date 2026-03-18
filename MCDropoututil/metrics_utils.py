"""
Metrics utilities for segmentation evaluation.
Computes Dice scores and other evaluation metrics.
"""
import numpy as np
import SimpleITK as sitk
from typing import Dict, Optional


def compute_dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """
    Compute Dice score for a specific label.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        label: Label value to compute Dice for
        
    Returns:
        Dice score (0-1)
    """
    pred_mask = (pred == label).astype(np.float32)
    gt_mask = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return float(dice)


def compute_brats_dice_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute BraTS-specific Dice scores.
    
    BraTS regions (using nnU-Net encoding):
    - WT (Whole Tumor): labels 1, 2, 3
    - TC (Tumor Core): labels 1, 3
    - ET (Enhancing Tumor): label 3
    
    Note: nnU-Net remaps original BraTS labels (1,2,4) to (1,2,3)
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        
    Returns:
        Dictionary with Dice scores for WT, TC, ET, and mean
    """
    # Create region masks (using nnU-Net label encoding)
    wt_pred = np.isin(pred, [1, 2, 3]).astype(np.float32)
    wt_gt = np.isin(gt, [1, 2, 3]).astype(np.float32)
    
    tc_pred = np.isin(pred, [1, 3]).astype(np.float32)
    tc_gt = np.isin(gt, [1, 3]).astype(np.float32)
    
    et_pred = (pred == 3).astype(np.float32)
    et_gt = (gt == 3).astype(np.float32)
    
    # Compute Dice for each region
    def dice(pred_mask, gt_mask):
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return (2.0 * intersection) / union
    
    dice_wt = dice(wt_pred, wt_gt)
    dice_tc = dice(tc_pred, tc_gt)
    dice_et = dice(et_pred, et_gt)
    dice_mean = (dice_wt + dice_tc + dice_et) / 3.0
    
    return {
        'dice_wt': float(dice_wt),
        'dice_tc': float(dice_tc),
        'dice_et': float(dice_et),
        'dice_mean': float(dice_mean)
    }


def load_nifti_array(filepath: str) -> np.ndarray:
    """
    Load NIfTI file as numpy array.
    
    Args:
        filepath: Path to NIfTI file
        
    Returns:
        Numpy array
    """
    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    return array


def determine_failure_case(dice_scores: Dict[str, float], threshold: float = 0.5) -> int:
    """
    Determine if a case is a failure based on Dice scores.
    
    Args:
        dice_scores: Dictionary with Dice scores
        threshold: Threshold for failure (default: 0.5)
        
    Returns:
        1 if failure, 0 otherwise
    """
    # Consider a case as failure if mean Dice < threshold
    # or if any region has very low Dice
    if dice_scores['dice_mean'] < threshold:
        return 1
    
    # Additional criterion: if any region has Dice < 0.3
    if min(dice_scores['dice_wt'], dice_scores['dice_tc'], dice_scores['dice_et']) < 0.3:
        return 1
    
    return 0
