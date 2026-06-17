"""
TTA uncertainty inference for nnU-Net (no retraining).

Method:
- Run deterministic inference under several test-time augmentations.
- Invert each augmentation in probability space and average probabilities.
- Use inter-augmentation variance as uncertainty.

Outputs per case (compatible with existing pipeline):
- {case_id}.nii.gz
- {case_id}_entropy.nii.gz
- {case_id}_variance.nii.gz
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subfiles,
    load_json,
    save_json,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


EXPECTED_TRAINER_NAME = 'nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TTA uncertainty inference for nnU-Net')

    parser.add_argument('--dataset', type=str, default='Dataset001_BraTS2021_Test')
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--case_ids', nargs='+', default=None)
    parser.add_argument('--case_ids_file', type=str, default=None,
                        help='Text file with one case id per line. Takes priority over --case_ids.')
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--warmup_cases', type=int, default=5,
                        help='Number of initial cases excluded from wall-clock summary stats.')

    parser.add_argument(
        '--tta_mode',
        type=str,
        default='flip3d',
        choices=['none', 'flip1', 'flip3d'],
        help='none: identity; flip1: identity + one flip; flip3d: identity + 3 axis flips',
    )

    return parser.parse_args()


def _normalize_case_id(case_id: str) -> str:
    case_id = case_id.strip()
    if case_id.endswith('.nii.gz'):
        case_id = case_id[:-7]
    elif case_id.endswith('.nii'):
        case_id = case_id[:-4]
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    return case_id


def get_case_ids(input_folder: str, requested_case_ids: List[str] = None) -> List[str]:
    if requested_case_ids is not None:
        return [_normalize_case_id(cid) for cid in requested_case_ids if cid.strip()]
    return sorted([f[:-12] for f in subfiles(input_folder, suffix='_0000.nii.gz', join=False)])


def get_case_ids_from_file(case_ids_file: str) -> List[str]:
    with open(case_ids_file, 'r', encoding='utf-8') as f:
        return [_normalize_case_id(line) for line in f if line.strip()]


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


def init_predictor(model_folder: str, fold: int, checkpoint_name: str, device: int) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device('cuda', device),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def _entropy_map(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(prob, min=eps, max=1.0)
    return -torch.sum(p * torch.log(p), dim=0)


def _prepare_case_data(predictor: nnUNetPredictor, case_files: List[str]) -> Dict:
    # Single-process preprocessing is more stable on Windows.
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, data_properties = preprocessor.run_case(
        case_files,
        None,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    return {
        'data': torch.from_numpy(data),
        'data_properties': data_properties,
    }


def _get_tta_dims(tta_mode: str) -> List[Tuple[int, ...]]:
    if tta_mode == 'none':
        return [tuple()]
    if tta_mode == 'flip1':
        return [tuple(), (2,)]
    return [tuple(), (2,), (3,), (4,)]


def _apply_aug_patch(patch: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    if len(dims) == 0:
        return patch
    return torch.flip(patch, dims=dims)


def _invert_aug_prob(prob: torch.Tensor, dims_patch: Tuple[int, ...]) -> torch.Tensor:
    if len(dims_patch) == 0:
        return prob
    dim_map = {2: 1, 3: 2, 4: 3}
    prob_dims = tuple(dim_map[d] for d in dims_patch)
    return torch.flip(prob, dims=prob_dims)


def _predict_tta_prob_and_var(
    predictor: nnUNetPredictor,
    data_tensor: torch.Tensor,
    tta_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_prob: (C, H, W, D)
      var_map: (H, W, D)
    """
    device = predictor.device
    network = predictor.network.to(device)
    network.eval()

    aug_dims_list = _get_tta_dims(tta_mode)

    with torch.inference_mode():
        x, slicer_revert_padding = pad_nd_image(
            data_tensor,
            predictor.configuration_manager.patch_size,
            mode='constant',
            kwargs={'value': 0},
            return_slicer=True,
            shape_must_be_divisible_by=None,
        )
        x = x.to(device)

        slicers = predictor._internal_get_sliding_window_slicers(tuple(x.shape[1:]))
        num_classes = predictor.label_manager.num_segmentation_heads

        accum_prob = torch.zeros((num_classes, *x.shape[1:]), dtype=torch.float32, device=device)
        accum_var = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_count = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)

        gaussian = compute_gaussian(
            tuple(predictor.configuration_manager.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10.0,
            device=device,
        ).to(torch.float32)

        for sl in tqdm(slicers, desc='TTA Sliding Window', leave=False):
            patch = x[sl][None].contiguous()

            probs_list = []
            for dims_patch in aug_dims_list:
                patch_aug = _apply_aug_patch(patch, dims_patch)
                logits_aug = predictor._internal_maybe_mirror_and_predict(patch_aug)[0].to(torch.float32)
                prob_aug = torch.softmax(logits_aug, dim=0)
                prob_inv = _invert_aug_prob(prob_aug, dims_patch)
                probs_list.append(prob_inv)

            probs_stack = torch.stack(probs_list, dim=0)
            fused_prob = torch.mean(probs_stack, dim=0)
            inter_aug_var = torch.var(probs_stack, dim=0, unbiased=False).mean(dim=0)

            accum_prob[sl] += fused_prob * gaussian
            accum_var[sl[1:]] += inter_aug_var * gaussian
            accum_count[sl[1:]] += gaussian

        accum_count = torch.clamp(accum_count, min=1e-8)
        mean_prob = accum_prob / accum_count.unsqueeze(0)
        var_map = accum_var / accum_count

        mean_prob = mean_prob[(slice(None), *slicer_revert_padding[1:])]
        var_map = var_map[slicer_revert_padding[1:]]

        empty_cache(device)
        return mean_prob.cpu().numpy(), var_map.cpu().numpy()


def _save_outputs(
    mean_prob: np.ndarray,
    var_map: np.ndarray,
    properties: Dict,
    output_folder: str,
    case_id: str,
) -> Dict[str, float]:
    seg = np.argmax(mean_prob, axis=0).astype(np.uint8)
    ent = _entropy_map(torch.from_numpy(mean_prob).to(torch.float32)).numpy().astype(np.float32)
    var = var_map.astype(np.float32)

    bbox = properties.get('bbox_used_for_cropping')
    original_shape = properties.get('shape_before_cropping')

    if bbox is not None and original_shape is not None:
        slicer = bounding_box_to_slice(bbox)
        seg_full = np.zeros(original_shape, dtype=np.uint8)
        ent_full = np.zeros(original_shape, dtype=np.float32)
        var_full = np.zeros(original_shape, dtype=np.float32)
        seg_full[slicer] = seg
        ent_full[slicer] = ent
        var_full[slicer] = var
        seg, ent, var = seg_full, ent_full, var_full

    sitk.WriteImage(sitk.GetImageFromArray(seg), join(output_folder, f'{case_id}.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(ent), join(output_folder, f'{case_id}_entropy.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(var), join(output_folder, f'{case_id}_variance.nii.gz'))

    return {
        'case_id': case_id,
        'entropy_mean': float(np.mean(ent)),
        'variance_mean': float(np.mean(var)),
    }


def run(args: argparse.Namespace) -> None:
    if args.model_folder is None:
        args.model_folder = join(
            nnUNet_results,
            'Dataset002_BraTS2021_Train/nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres',
        )
    _assert_expected_model_folder(args.model_folder)
    if args.input_folder is None:
        args.input_folder = join(nnUNet_raw, f'{args.dataset}/imagesTr')
    if args.output_folder is None:
        args.output_folder = join(nnUNet_results, f'{args.dataset}/TTA_Inference_Results')

    maybe_mkdir_p(args.output_folder)

    predictor = init_predictor(args.model_folder, args.fold, args.checkpoint_name, args.device)
    n_mod = len(load_json(join(args.model_folder, 'dataset.json'))['channel_names'])

    if args.case_ids_file is not None:
        case_ids = get_case_ids_from_file(args.case_ids_file)
    else:
        case_ids = get_case_ids(args.input_folder, args.case_ids)
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]

    results = []
    latency_rows = []
    for idx, cid in enumerate(case_ids):
        _sync_if_cuda(predictor.device)
        t0 = time.perf_counter()
        case_files = [join(args.input_folder, f'{cid}_{i:04d}.nii.gz') for i in range(n_mod)]
        batch = _prepare_case_data(predictor, case_files)
        mean_prob, var_map = _predict_tta_prob_and_var(predictor, batch['data'], args.tta_mode)
        stats = _save_outputs(mean_prob, var_map, batch['data_properties'], args.output_folder, cid)
        results.append(stats)
        _sync_if_cuda(predictor.device)
        latency_rows.append({
            'case_id': cid,
            'latency_sec': float(time.perf_counter() - t0),
            'is_warmup': idx < args.warmup_cases,
        })

    latency_csv = _write_latency_csv(args.output_folder, latency_rows)
    latency_summary = _summarize_latency(latency_rows)

    summary = {
        'dataset': args.dataset,
        'model_folder': args.model_folder,
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'num_cases': len(case_ids),
        'tta_mode': args.tta_mode,
        'warmup_cases': int(args.warmup_cases),
        'latency_csv': latency_csv,
        **latency_summary,
        'results': results,
    }
    save_json(summary, join(args.output_folder, 'summary.json'))
    print('Done. Summary:', join(args.output_folder, 'summary.json'))


if __name__ == '__main__':
    run(parse_args())
