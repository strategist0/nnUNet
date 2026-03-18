"""
Compare baseline vs MC Dropout results.
Reads two master_results.csv files and generates comparison plots.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from MCDropoututil.reliability_utils import (
    compute_auroc_auprc,
    compute_correlation,
    compute_risk_coverage
)
from MCDropoututil.plotting_utils import (
    plot_bar_comparison,
    plot_risk_coverage_compare
)


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ['case_id', 'dice_mean', 'failure_label']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def compute_reliability_metrics(df: pd.DataFrame, uncertainty_cols: list) -> list:
    y_true = df['failure_label'].values.astype(int)
    dice = df['dice_mean'].values.astype(float)
    risk = 1.0 - dice

    results = []
    for col in uncertainty_cols:
        if col not in df.columns:
            continue
        scores = df[col].values.astype(float)
        
        # Skip uncertainty metrics if all values are zero or nearly zero (e.g., Baseline with no MC sampling)
        if np.std(scores) < 1e-8:
            continue
        
        au = compute_auroc_auprc(y_true, scores)
        corr = compute_correlation(scores, dice)
        coverage, risk_curve, aurc = compute_risk_coverage(scores, risk)
        results.append({
            'uncertainty_metric': col,
            'auroc': au['auroc'],
            'auprc': au['auprc'],
            'pearson': corr['pearson'],
            'spearman': corr['spearman'],
            'aurc': aurc,
            'coverage': coverage,
            'risk_curve': risk_curve
        })
    return results


def run_comparison(baseline_csv: str, mc_csv: str, output_dir: str, label_a: str, label_b: str) -> None:
    maybe_mkdir_p(output_dir)

    df_a = load_results(baseline_csv)
    df_b = load_results(mc_csv)

    uncertainty_cols = [
        'var_tumor_max', 'var_tumor_p99', 'var_tumor_p95', 'var_tumor_std', 'var_tumor',
        'var_top10', 'var_mean',
        'entropy_tumor_max', 'entropy_tumor_p99', 'entropy_tumor_p95', 'entropy_tumor_std', 'entropy_tumor',
        'entropy_top10', 'entropy_mean'
    ]
    dice_cols = ['dice_mean', 'dice_wt', 'dice_tc', 'dice_et']

    # Dice comparison plot
    dice_means_a = [df_a[col].mean() for col in dice_cols if col in df_a.columns]
    dice_means_b = [df_b[col].mean() for col in dice_cols if col in df_b.columns]
    dice_stds_a = [df_a[col].std() for col in dice_cols if col in df_a.columns]
    dice_stds_b = [df_b[col].std() for col in dice_cols if col in df_b.columns]

    plot_bar_comparison(
        labels=[c.replace('dice_', '').upper() for c in dice_cols],
        values_a=dice_means_a,
        values_b=dice_means_b,
        title='Dice Comparison',
        output_path=join(output_dir, 'dice_comparison.png'),
        label_a=label_a,
        label_b=label_b,
        yerr_a=dice_stds_a,
        yerr_b=dice_stds_b
    )

    # Reliability metrics
    metrics_a = compute_reliability_metrics(df_a, uncertainty_cols)
    metrics_b = compute_reliability_metrics(df_b, uncertainty_cols)

    # Save comparison table
    rows = []
    for m in metrics_a:
        rows.append({
            'model': label_a,
            'uncertainty_metric': m['uncertainty_metric'],
            'auroc': m['auroc'],
            'auprc': m['auprc'],
            'pearson': m['pearson'],
            'spearman': m['spearman'],
            'aurc': m['aurc']
        })
    for m in metrics_b:
        rows.append({
            'model': label_b,
            'uncertainty_metric': m['uncertainty_metric'],
            'auroc': m['auroc'],
            'auprc': m['auprc'],
            'pearson': m['pearson'],
            'spearman': m['spearman'],
            'aurc': m['aurc']
        })

    pd.DataFrame(rows).to_csv(join(output_dir, 'comparison_metrics.csv'), index=False, float_format='%.6f')

    # Risk-coverage plots (overlay)
    metrics_map_a = {m['uncertainty_metric']: m for m in metrics_a}
    metrics_map_b = {m['uncertainty_metric']: m for m in metrics_b}

    for col in uncertainty_cols:
        if col not in metrics_map_a or col not in metrics_map_b:
            continue
        plot_risk_coverage_compare(
            metrics_map_a[col]['coverage'],
            metrics_map_a[col]['risk_curve'],
            metrics_map_b[col]['coverage'],
            metrics_map_b[col]['risk_curve'],
            title=f'Risk-Coverage ({col})',
            output_path=join(output_dir, f'risk_coverage_compare_{col}.png'),
            label_a=label_a,
            label_b=label_b
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare baseline vs MC Dropout results')
    parser.add_argument('--baseline_csv', type=str, required=True, help='Path to baseline master_results.csv')
    parser.add_argument('--mc_csv', type=str, required=True, help='Path to MC master_results.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots and tables')
    parser.add_argument('--label_baseline', type=str, default='Baseline', help='Label for baseline model')
    parser.add_argument('--label_mc', type=str, default='MC', help='Label for MC model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_comparison(args.baseline_csv, args.mc_csv, args.output_dir, args.label_baseline, args.label_mc)
