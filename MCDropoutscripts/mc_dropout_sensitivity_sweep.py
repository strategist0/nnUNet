"""
MC Dropout sensitivity sweep for dropout probability (p) and sampling count (N).

Default sweep grid:
- p in [0.1, 0.3, 0.5]
- N in [10, 20, 30]

Pipeline per setting:
1) Run MC inference (decoder-only injection by default)
2) Build master_results_200.csv
3) Compute compact summary metrics

Outputs:
- <sweep_root>/p{p}_N{N}/ (inference outputs)
- <sweep_root>/sweep_summary.csv
"""

import argparse
import os
import sys
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MCDropoutscripts.mc_inference import run_mc_dropout_inference
from MCDropoutscripts.collect_case_data import collect_all_cases
from MCDropoututil.reliability_utils import compute_auroc_auprc, compute_risk_coverage


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _normalize_case_id(case_id: str) -> str:
    case_id = case_id.strip()
    if case_id.endswith('.nii.gz'):
        case_id = case_id[:-7]
    elif case_id.endswith('.nii'):
        case_id = case_id[:-4]
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    return case_id


def _load_case_ids(case_ids_file: str) -> List[str]:
    with open(case_ids_file, 'r', encoding='utf-8') as f:
        return [_normalize_case_id(line) for line in f if line.strip()]


def _is_case_complete(output_folder: str, case_id: str) -> bool:
    seg = join(output_folder, f'{case_id}.nii.gz')
    ent = join(output_folder, f'{case_id}_entropy.nii.gz')
    var = join(output_folder, f'{case_id}_variance.nii.gz')
    return os.path.isfile(seg) and os.path.isfile(ent) and os.path.isfile(var)


def _get_completed_case_ids(output_folder: str, case_ids: List[str]) -> List[str]:
    return [case_id for case_id in case_ids if _is_case_complete(output_folder, case_id)]


def _compute_quality(df: pd.DataFrame, score_col: str) -> dict:
    if score_col not in df.columns:
        return {'auroc': np.nan, 'auprc': np.nan, 'aurc': np.nan}

    scores = df[score_col].values.astype(float)
    y_true = df['failure_label'].values.astype(int)
    risk = 1.0 - df['dice_mean'].values.astype(float)
    if np.std(scores) < 1e-12:
        return {'auroc': np.nan, 'auprc': np.nan, 'aurc': np.nan}

    au = compute_auroc_auprc(y_true, scores)
    _, _, aurc = compute_risk_coverage(scores, risk)
    return {'auroc': float(au['auroc']), 'auprc': float(au['auprc']), 'aurc': float(aurc)}


def _run_single_setting(args: argparse.Namespace, p: float, n_samples: int, case_ids: List[str]) -> dict:
    setting_name = f"p{p:.2f}_N{n_samples}".replace('.', 'p')
    out_dir = join(args.sweep_root, setting_name)
    master_csv = join(out_dir, 'master_results_200.csv')
    maybe_mkdir_p(out_dir)

    if args.resume and os.path.isfile(master_csv):
        try:
            existing = pd.read_csv(master_csv)
            if len(existing) == len(case_ids):
                print('\n' + '=' * 80)
                print(f'[RESUME-SKIP] {setting_name} already complete ({len(existing)} cases).')
                print('=' * 80)
                q_entropy = _compute_quality(existing, 'entropy_tumor')
                q_var = _compute_quality(existing, 'var_mean')
                return {
                    'setting': setting_name,
                    'dropout_p': p,
                    'num_mc_iterations': n_samples,
                    'num_cases': int(len(existing)),
                    'dice_mean': float(existing['dice_mean'].mean()),
                    'dice_std': float(existing['dice_mean'].std()),
                    'failure_rate_pct': float(existing['failure_label'].mean() * 100.0),
                    'auroc_entropy_tumor': q_entropy['auroc'],
                    'auprc_entropy_tumor': q_entropy['auprc'],
                    'aurc_entropy_tumor': q_entropy['aurc'],
                    'auroc_var_mean': q_var['auroc'],
                    'auprc_var_mean': q_var['auprc'],
                    'aurc_var_mean': q_var['aurc'],
                    'output_folder': out_dir,
                    'master_csv': master_csv,
                }
        except Exception:
            pass

    run_case_ids = case_ids
    if args.resume:
        completed = set(_get_completed_case_ids(out_dir, case_ids))
        run_case_ids = [case_id for case_id in case_ids if case_id not in completed]

        print('\n' + '=' * 80)
        print(f'[RESUME] {setting_name}: completed={len(completed)}, remaining={len(run_case_ids)}')
        print('=' * 80)

    infer_args = Namespace(
        dataset=args.dataset,
        model_folder=args.model_folder,
        input_folder=args.input_folder,
        output_folder=out_dir,
        case_ids=run_case_ids,
        max_cases=None,
        num_mc_iterations=n_samples,
        checkpoint_name=args.checkpoint_name,
        fold=args.fold,
        device=args.device,
        dropout_p=p,
        inject_dropout=True,
        inject_decoder_only=args.inject_decoder_only,
        enable_mc_dropout=True,
    )

    print('\n' + '=' * 80)
    print(f'Running setting: p={p}, N={n_samples}')
    print(f'Output folder: {out_dir}')
    print('=' * 80)

    if len(run_case_ids) > 0:
        run_mc_dropout_inference(infer_args)
    else:
        print('[RESUME] No remaining cases. Skip inference and rebuild summary CSV only.')

    collect_all_cases(
        prediction_folder=out_dir,
        gt_folder=args.gt_folder,
        output_csv=master_csv,
        case_ids=case_ids,
    )

    df = pd.read_csv(master_csv)
    q_entropy = _compute_quality(df, 'entropy_tumor')
    q_var = _compute_quality(df, 'var_mean')

    return {
        'setting': setting_name,
        'dropout_p': p,
        'num_mc_iterations': n_samples,
        'num_cases': int(len(df)),
        'dice_mean': float(df['dice_mean'].mean()),
        'dice_std': float(df['dice_mean'].std()),
        'failure_rate_pct': float(df['failure_label'].mean() * 100.0),
        'auroc_entropy_tumor': q_entropy['auroc'],
        'auprc_entropy_tumor': q_entropy['auprc'],
        'aurc_entropy_tumor': q_entropy['aurc'],
        'auroc_var_mean': q_var['auroc'],
        'auprc_var_mean': q_var['auprc'],
        'aurc_var_mean': q_var['aurc'],
        'output_folder': out_dir,
        'master_csv': master_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MC Dropout sensitivity sweep (p x N)')

    parser.add_argument('--dataset', type=str, default='Dataset001_BraTS2021_Test')
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--gt_folder', type=str, required=True)
    parser.add_argument('--sweep_root', type=str, required=True)
    parser.add_argument('--case_ids_file', type=str, required=True)

    parser.add_argument('--dropout_ps', type=str, default='0.1,0.3,0.5')
    parser.add_argument('--mc_iterations', type=str, default='10,20,30')

    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_cases', type=int, default=None)

    parser.add_argument('--inject_decoder_only', action='store_true', default=True)
    parser.add_argument('--no_inject_decoder_only', dest='inject_decoder_only', action='store_false')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing outputs and skip finished work (default: True)')
    parser.add_argument('--no_resume', dest='resume', action='store_false')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_mkdir_p(args.sweep_root)

    case_ids = _load_case_ids(args.case_ids_file)
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]

    dropout_ps = _parse_float_list(args.dropout_ps)
    mc_iterations = _parse_int_list(args.mc_iterations)

    summary_csv = join(args.sweep_root, 'sweep_summary.csv')
    rows_by_setting = {}
    if args.resume and os.path.isfile(summary_csv):
        try:
            prev = pd.read_csv(summary_csv)
            for _, row in prev.iterrows():
                rows_by_setting[str(row['setting'])] = row.to_dict()
        except Exception:
            pass

    for p in dropout_ps:
        for n_samples in mc_iterations:
            row = _run_single_setting(args, p, n_samples, case_ids)
            rows_by_setting[row['setting']] = row
            pd.DataFrame(list(rows_by_setting.values())).sort_values('setting').to_csv(
                summary_csv, index=False, float_format='%.6f'
            )

    summary_df = pd.DataFrame(list(rows_by_setting.values())).sort_values('setting')
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')

    print('\n' + '=' * 80)
    print(f'Sensitivity sweep complete. Summary: {summary_csv}')
    print('=' * 80)


if __name__ == '__main__':
    main()
