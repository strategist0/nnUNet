"""
Temperature Scaling inference for nnU-Net (no retraining required).

This script:
1) Fits a single temperature T on a small fitting split (optional).
2) Runs deterministic nnU-Net inference and exports temperature-scaled outputs.
3) Saves segmentation + uncertainty maps with names compatible with existing metrics scripts:
   - {case_id}.nii.gz
   - {case_id}_entropy.nii.gz
   - {case_id}_variance.nii.gz

Notes:
- Keeps your original MC Dropout code untouched.
- Uses probability-space scaling: p_T(c|x) = p(c|x)^(1/T) / sum_k p(k|x)^(1/T)
  which is equivalent to softmax(logits / T).
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subfiles,
    load_json,
    save_json,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw, nnUNet_results


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


EXPECTED_TRAINER_NAME = 'nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres'


SPLITS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'splits'))
DEFAULT_FIT_CASE_IDS_FILE = os.path.join(
    SPLITS_DIR,
    'Dataset001_BraTS2021_Test_calibration_50_seed42.txt',
)
DEFAULT_INFER_CASE_IDS_FILE = os.path.join(
    SPLITS_DIR,
    'Dataset001_BraTS2021_Test_final_test_200_seed42.txt',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Temperature Scaling for nnU-Net inference')

    parser.add_argument('--dataset', type=str, default='Dataset002_BraTS2021_Train')
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--gt_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)

    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--mode', type=str, default='both', choices=['fit', 'infer', 'both'])
    parser.add_argument('--temperature', type=float, default=None,
                        help='If provided, skip fitting and use this T directly.')

    parser.add_argument('--fit_cases', type=int, default=20,
                        help='Number of cases used to fit T (from start of discovered case list).')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Max number of cases to run inference on.')
    parser.add_argument('--case_ids', nargs='+', default=None)
    parser.add_argument('--fit_case_ids_file', type=str, default=DEFAULT_FIT_CASE_IDS_FILE,
                        help='Text file with one case id per line used only for temperature fitting.')
    parser.add_argument('--infer_case_ids_file', type=str, default=DEFAULT_INFER_CASE_IDS_FILE,
                        help='Text file with one case id per line used for inference.')

    parser.add_argument('--num_proc_pre', type=int, default=2)
    parser.add_argument('--num_proc_export', type=int, default=2)
    parser.add_argument('--warmup_cases', type=int, default=5,
                        help='Number of initial inference cases excluded from wall-clock summary stats.')

    parser.add_argument('--t_min', type=float, default=0.5)
    parser.add_argument('--t_max', type=float, default=5.0)
    parser.add_argument('--t_steps', type=int, default=60)

    return parser.parse_args()


def get_case_ids(input_folder: str, requested_case_ids: List[str] = None) -> List[str]:
    if requested_case_ids is not None:
        return requested_case_ids
    files = subfiles(input_folder, suffix='_0000.nii.gz', join=False)
    return sorted([f[:-12] for f in files])


def _load_case_ids_from_file(case_ids_file: str) -> List[str]:
    with open(case_ids_file, 'r') as f:
        case_ids = []
        for line in f:
            case_id = line.strip()
            if not case_id:
                continue

            # Normalize possible file-style entries such as:
            # BraTS2021_00006_0000.nii.gz -> BraTS2021_00006
            # BraTS2021_00006_0000.nii -> BraTS2021_00006
            if case_id.endswith('.nii.gz'):
                case_id = case_id[:-7]
            elif case_id.endswith('.nii'):
                case_id = case_id[:-4]

            if case_id.endswith('_0000'):
                case_id = case_id[:-5]

            case_ids.append(case_id)

        return case_ids


def init_predictor(model_folder: str, fold: int, checkpoint_name: str, device: int) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda', device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _write_latency_csv(output_folder: str, rows: List[Dict]) -> str:
    path = join(output_folder, 'latency_wall_clock.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'latency_sec', 'is_warmup'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summarize_latency(rows: List[Dict]) -> Dict[str, float]:
    valid = [float(r['latency_sec']) for r in rows if not bool(r['is_warmup'])]
    if len(valid) == 0:
        return {
            'latency_num_cases': 0,
            'latency_sec_total': 0.0,
            'latency_sec_per_case': float('nan'),
        }
    total = float(sum(valid))
    return {
        'latency_num_cases': int(len(valid)),
        'latency_sec_total': total,
        'latency_sec_per_case': float(total / len(valid)),
    }


def _assert_expected_model_folder(model_folder: str) -> None:
    norm = os.path.normpath(model_folder)
    if EXPECTED_TRAINER_NAME.lower() not in norm.lower():
        raise ValueError(
            f'Unexpected model_folder: {model_folder}. '
            f'Expected decoder-only trainer path containing: {EXPECTED_TRAINER_NAME}'
        )


def _load_npz_probs(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    if 'probabilities' in data:
        probs = data['probabilities']
    else:
        first_key = list(data.keys())[0]
        probs = data[first_key]
    # expected shape: (C, H, W, D)
    return probs.astype(np.float32, copy=False)


def _scale_probabilities(prob: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    # Equivalent to softmax(logits / T) when prob = softmax(logits)
    prob = np.clip(prob, eps, 1.0)
    power = 1.0 / float(temperature)
    scaled = np.power(prob, power, dtype=np.float32)
    denom = np.sum(scaled, axis=0, keepdims=True)
    denom = np.clip(denom, eps, None)
    scaled /= denom
    return scaled


def _entropy_map(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(prob, eps, 1.0)
    return -np.sum(p * np.log(p), axis=0, dtype=np.float32)


def _variance_map(prob: np.ndarray) -> np.ndarray:
    # Single-pass proxy variance across class probabilities.
    return np.var(prob, axis=0, dtype=np.float32)


def _nll_from_prob_and_gt(prob: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> Tuple[float, int]:
    # prob: (C, H, W, D), gt: (H, W, D)
    c, h, w, d = prob.shape
    if gt.shape != (h, w, d):
        raise ValueError(f'Shape mismatch: prob={prob.shape}, gt={gt.shape}')

    gt_int = gt.astype(np.int64, copy=False)
    valid = (gt_int >= 0) & (gt_int < c)
    if not np.any(valid):
        return 0.0, 0

    flat_prob = np.transpose(prob, (1, 2, 3, 0))[valid]  # (N, C)
    flat_gt = gt_int[valid]                              # (N,)

    chosen = flat_prob[np.arange(flat_prob.shape[0]), flat_gt]
    chosen = np.clip(chosen, eps, 1.0)
    nll_sum = float(-np.log(chosen).sum())
    return nll_sum, int(flat_gt.shape[0])


def _predict_with_prob_export(
    predictor: nnUNetPredictor,
    case_id: str,
    input_folder: str,
    output_folder: str,
    num_modalities: int,
    num_proc_pre: int,
    num_proc_export: int,
) -> Tuple[str, str]:
    case_files = [[join(input_folder, f'{case_id}_{i:04d}.nii.gz') for i in range(num_modalities)]]
    predictor.predict_from_files(
        case_files,
        output_folder,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=num_proc_pre,
        num_processes_segmentation_export=num_proc_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    seg_path = join(output_folder, f'{case_id}.nii.gz')
    prob_path = join(output_folder, f'{case_id}.npz')
    return seg_path, prob_path


def fit_temperature(
    predictor: nnUNetPredictor,
    fit_case_ids: List[str],
    input_folder: str,
    gt_folder: str,
    cache_folder: str,
    num_modalities: int,
    num_proc_pre: int,
    num_proc_export: int,
    t_min: float,
    t_max: float,
    t_steps: int,
) -> Dict:
    maybe_mkdir_p(cache_folder)

    # Cache probabilities once to speed up grid search over T.
    cached_pairs = []
    for case_id in fit_case_ids:
        seg_path, prob_path = _predict_with_prob_export(
            predictor,
            case_id,
            input_folder,
            cache_folder,
            num_modalities,
            num_proc_pre,
            num_proc_export,
        )
        gt_path = join(gt_folder, f'{case_id}.nii.gz')
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f'Ground truth not found for fitting case: {gt_path}')
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f'Probability export failed: {prob_path}')
        cached_pairs.append((case_id, prob_path, gt_path, seg_path))

    grid = np.linspace(t_min, t_max, t_steps, dtype=np.float32)
    best_t = None
    best_nll = float('inf')

    for t in grid:
        total_nll = 0.0
        total_vox = 0

        for _, prob_path, gt_path, _ in cached_pairs:
            prob = _load_npz_probs(prob_path)
            scaled = _scale_probabilities(prob, float(t))

            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.int32)
            nll_sum, n_vox = _nll_from_prob_and_gt(scaled, gt)
            total_nll += nll_sum
            total_vox += n_vox

        if total_vox == 0:
            continue

        mean_nll = total_nll / total_vox
        if mean_nll < best_nll:
            best_nll = mean_nll
            best_t = float(t)

    if best_t is None:
        raise RuntimeError('Temperature fitting failed: no valid voxels found.')

    return {
        'temperature': best_t,
        'best_nll': float(best_nll),
        'fit_cases': fit_case_ids,
        'search': {
            't_min': float(t_min),
            't_max': float(t_max),
            't_steps': int(t_steps),
        },
    }


def run_inference_with_temperature(
    predictor: nnUNetPredictor,
    case_ids: List[str],
    input_folder: str,
    output_folder: str,
    num_modalities: int,
    temperature: float,
    num_proc_pre: int,
    num_proc_export: int,
    warmup_cases: int,
) -> Dict:
    maybe_mkdir_p(output_folder)
    stats = []
    latency_rows = []

    for idx, case_id in enumerate(case_ids):
        _sync_if_cuda(predictor.device)
        t0 = time.perf_counter()
        seg_path, prob_path = _predict_with_prob_export(
            predictor,
            case_id,
            input_folder,
            output_folder,
            num_modalities,
            num_proc_pre,
            num_proc_export,
        )

        prob = _load_npz_probs(prob_path)
        scaled = _scale_probabilities(prob, temperature)

        seg_scaled = np.argmax(scaled, axis=0).astype(np.uint8)
        ent = _entropy_map(scaled).astype(np.float32)
        var = _variance_map(scaled).astype(np.float32)

        # Keep spatial metadata from nnU-Net exported segmentation.
        seg_img_ref = sitk.ReadImage(seg_path)

        seg_img = sitk.GetImageFromArray(seg_scaled)
        seg_img.CopyInformation(seg_img_ref)
        sitk.WriteImage(seg_img, seg_path)

        ent_path = join(output_folder, f'{case_id}_entropy.nii.gz')
        ent_img = sitk.GetImageFromArray(ent)
        ent_img.CopyInformation(seg_img_ref)
        sitk.WriteImage(ent_img, ent_path)

        var_path = join(output_folder, f'{case_id}_variance.nii.gz')
        var_img = sitk.GetImageFromArray(var)
        var_img.CopyInformation(seg_img_ref)
        sitk.WriteImage(var_img, var_path)

        stats.append({
            'case_id': case_id,
            'entropy_mean': float(ent.mean()),
            'variance_mean': float(var.mean()),
        })

        _sync_if_cuda(predictor.device)
        latency_rows.append({
            'case_id': case_id,
            'latency_sec': float(time.perf_counter() - t0),
            'is_warmup': idx < warmup_cases,
        })

    return {'num_cases': len(case_ids), 'case_stats': stats, 'latency_rows': latency_rows}


def main() -> None:
    args = parse_args()

    if args.model_folder is None:
        args.model_folder = join(
            nnUNet_results,
            'Dataset002_BraTS2021_Train/nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres',
        )
    _assert_expected_model_folder(args.model_folder)
    if args.input_folder is None:
        args.input_folder = join(nnUNet_raw, f'{args.dataset}/imagesTr')
    if args.gt_folder is None:
        args.gt_folder = join(nnUNet_raw, f'{args.dataset}/labelsTr')
    if args.output_folder is None:
        args.output_folder = join(nnUNet_results, f'{args.dataset}/TS_Inference_Results')

    maybe_mkdir_p(args.output_folder)

    predictor = init_predictor(
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )

    dataset_json = load_json(join(args.model_folder, 'dataset.json'))
    num_modalities = len(dataset_json['channel_names'])

    all_case_ids = get_case_ids(args.input_folder, args.case_ids)
    if args.max_cases is not None:
        all_case_ids = all_case_ids[:args.max_cases]

    fit_case_ids = None
    if args.fit_case_ids_file is not None:
        fit_case_ids = _load_case_ids_from_file(args.fit_case_ids_file)

    infer_case_ids = all_case_ids
    if args.infer_case_ids_file is not None:
        infer_case_ids = _load_case_ids_from_file(args.infer_case_ids_file)

    if len(all_case_ids) == 0:
        raise RuntimeError('No cases found for processing.')

    summary = {
        'dataset': args.dataset,
        'model_folder': args.model_folder,
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'mode': args.mode,
    }

    fit_cache = join(args.output_folder, '_fit_cache')
    temperature = args.temperature

    if args.mode in ('fit', 'both'):
        fit_started = time.perf_counter()
        if temperature is None:
            if fit_case_ids is None:
                fit_case_ids = all_case_ids[: max(1, min(args.fit_cases, len(all_case_ids)))]
            fit_result = fit_temperature(
                predictor=predictor,
                fit_case_ids=fit_case_ids,
                input_folder=args.input_folder,
                gt_folder=args.gt_folder,
                cache_folder=fit_cache,
                num_modalities=num_modalities,
                num_proc_pre=args.num_proc_pre,
                num_proc_export=args.num_proc_export,
                t_min=args.t_min,
                t_max=args.t_max,
                t_steps=args.t_steps,
            )
            temperature = fit_result['temperature']
            summary['fit_result'] = fit_result
        else:
            summary['fit_result'] = {
                'temperature': float(temperature),
                'provided_by_user': True,
            }
        summary['fit_elapsed_seconds'] = float(time.perf_counter() - fit_started)

    if temperature is None:
        raise RuntimeError('Temperature is required for inference. Provide --temperature or run --mode fit/both.')

    summary['temperature'] = float(temperature)

    if args.mode in ('infer', 'both'):
        infer_result = run_inference_with_temperature(
            predictor=predictor,
            case_ids=infer_case_ids,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            num_modalities=num_modalities,
            temperature=float(temperature),
            num_proc_pre=args.num_proc_pre,
            num_proc_export=args.num_proc_export,
            warmup_cases=args.warmup_cases,
        )
        latency_rows = infer_result.pop('latency_rows', [])
        latency_csv = _write_latency_csv(args.output_folder, latency_rows)
        latency_summary = _summarize_latency(latency_rows)
        summary['infer_result'] = infer_result
        summary['warmup_cases'] = int(args.warmup_cases)
        summary['latency_csv'] = latency_csv
        summary.update(latency_summary)

    save_json(summary, join(args.output_folder, 'summary.json'))
    print('Done. Summary saved to:', join(args.output_folder, 'summary.json'))
    print('Temperature used:', float(temperature))


if __name__ == '__main__':
    main()
