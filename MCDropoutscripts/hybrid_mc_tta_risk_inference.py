"""
Hybrid MC+TTA uncertainty inference for nnU-Net.

Design goal:
- keep MC Dropout as a source of epistemic uncertainty
- use TTA disagreement as the primary low-noise geometric instability signal
- optimize the uncertainty maps for failure ranking quality rather than Dice

Outputs per case:
- {case_id}.nii.gz
- {case_id}_variance.nii.gz          : hybrid risk-focused variance map
- {case_id}_entropy.nii.gz           : hybrid risk-focused entropy map
- (optional, save_mode=full) {case_id}_variance_tta.nii.gz      : pure TTA disagreement map
- (optional, save_mode=full) {case_id}_variance_mc.nii.gz       : pure MC disagreement map
- (optional, save_mode=full) {case_id}_mutual_info.nii.gz       : predictive mutual information map
- (optional, save_mode=full) {case_id}_boundary_disagree.nii.gz : sample-boundary disagreement map
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
    subfiles,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.helpers import empty_cache
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MCDropoututil.model_utils import (  # noqa: E402
    enable_mc_dropout,
    get_unwrapped_network,
    inject_dropout_layers,
    verify_dropout_injection,
)


EXPECTED_TRAINER_NAME = (
    'nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres'
)

_BOUNDARY_BAND_RADIUS = 2
_BOUNDARY_GATE_THRESHOLD = 0.12
_TAIL_QUANTILE = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Hybrid MC+TTA risk-focused uncertainty inference for nnU-Net'
    )

    parser.add_argument('--dataset', type=str, default='Dataset001_BraTS2021_Test')
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--case_ids', nargs='+', default=None)
    parser.add_argument('--case_ids_file', type=str, default=None)
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--warmup_cases', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing outputs/progress in output_folder')
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    parser.add_argument(
        '--save_mode',
        type=str,
        default='full',
        choices=['full', 'minimal'],
        help='full: write all uncertainty maps; minimal: write only seg/variance/entropy for master_results.csv',
    )

    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--num_mc_samples', type=int, default=8)
    parser.add_argument('--inject_decoder_only', action='store_true', default=True)
    parser.add_argument('--no_inject_decoder_only', action='store_false', dest='inject_decoder_only')

    parser.add_argument(
        '--tta_mode',
        type=str,
        default='flip3d',
        choices=['none', 'flip1', 'flip3d'],
    )

    parser.add_argument(
        '--progress_mode',
        type=str,
        default='plain',
        choices=['plain', 'tqdm'],
    )
    parser.add_argument('--plain_log_every_patches', type=int, default=2)
    return parser.parse_args()


def _normalize_case_id(case_id: str) -> str:
    case_id = case_id.strip()
    for suffix in ('.nii.gz', '.nii'):
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    return case_id


def get_case_ids(input_folder: str, requested_case_ids: List[str] | None = None) -> List[str]:
    if requested_case_ids is not None:
        return [_normalize_case_id(c) for c in requested_case_ids if c.strip()]
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


def _progress_csv_path(output_folder: str) -> str:
    return join(output_folder, 'progress_cases.csv')


def _append_progress_row(output_folder: str, row: Dict) -> None:
    path = _progress_csv_path(output_folder)
    fieldnames = ['case_id', 'variance_mean', 'entropy_mean', 'latency_sec', 'is_warmup']
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def _load_progress_rows(output_folder: str) -> List[Dict]:
    path = _progress_csv_path(output_folder)
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = _normalize_case_id(str(r.get('case_id', '')))
            if not cid:
                continue
            try:
                variance_mean = float(r.get('variance_mean', 'nan'))
            except Exception:
                variance_mean = float('nan')
            try:
                entropy_mean = float(r.get('entropy_mean', 'nan'))
            except Exception:
                entropy_mean = float('nan')
            try:
                latency_sec = float(r.get('latency_sec', 'nan'))
            except Exception:
                latency_sec = float('nan')
            is_warmup = str(r.get('is_warmup', 'False')).strip().lower() in ('1', 'true', 'yes')
            rows.append({
                'case_id': cid,
                'variance_mean': variance_mean,
                'entropy_mean': entropy_mean,
                'latency_sec': latency_sec,
                'is_warmup': is_warmup,
            })
    return rows


def _detect_completed_case_ids_from_outputs(output_folder: str, case_ids: List[str]) -> List[str]:
    completed = []
    for cid in case_ids:
        seg_file = join(output_folder, f'{cid}.nii.gz')
        var_file = join(output_folder, f'{cid}_variance.nii.gz')
        ent_file = join(output_folder, f'{cid}_entropy.nii.gz')
        if os.path.exists(seg_file) and os.path.exists(var_file) and os.path.exists(ent_file):
            completed.append(cid)
    return completed


def _build_stats_from_existing_outputs(output_folder: str, case_ids: List[str]) -> List[Dict]:
    rows = []
    for cid in case_ids:
        var_file = join(output_folder, f'{cid}_variance.nii.gz')
        ent_file = join(output_folder, f'{cid}_entropy.nii.gz')
        if (not os.path.exists(var_file)) or (not os.path.exists(ent_file)):
            continue
        try:
            variance = sitk.GetArrayFromImage(sitk.ReadImage(var_file)).astype(np.float32)
            entropy = sitk.GetArrayFromImage(sitk.ReadImage(ent_file)).astype(np.float32)
            rows.append({
                'case_id': cid,
                'variance_mean': float(np.mean(variance)),
                'entropy_mean': float(np.mean(entropy)),
            })
        except Exception:
            continue
    return rows


def _summarize_latency(rows: List[Dict]) -> Dict[str, float]:
    valid = []
    for r in rows:
        if bool(r.get('is_warmup', False)):
            continue
        try:
            v = float(r.get('latency_sec', float('nan')))
        except Exception:
            v = float('nan')
        if np.isfinite(v):
            valid.append(v)
    if not valid:
        return {
            'latency_num_cases': 0,
            'latency_sec_total': 0.0,
            'latency_sec_per_case': float('nan'),
        }
    total = float(sum(valid))
    return {
        'latency_num_cases': len(valid),
        'latency_sec_total': total,
        'latency_sec_per_case': total / len(valid),
    }


def _assert_expected_model_folder(model_folder: str) -> None:
    norm = os.path.normpath(model_folder)
    if EXPECTED_TRAINER_NAME.lower() not in norm.lower():
        raise ValueError(
            f'Unexpected model_folder: {model_folder}. Expected path containing: {EXPECTED_TRAINER_NAME}'
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


def _prepare_network(
    predictor: nnUNetPredictor,
    dropout_p: float,
    decoder_only: bool,
) -> Dict:
    network = get_unwrapped_network(predictor.network)
    injected = inject_dropout_layers(network, dropout_p=dropout_p, decoder_only=decoder_only)
    enable_mc_dropout(network)
    verify = verify_dropout_injection(network)
    return {
        'dropout_layers_injected': injected,
        'dropout_layers_total': verify['total_dropout_layers'],
        'dropout_p': float(dropout_p),
        'decoder_only': bool(decoder_only),
    }


def _entropy_map(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(prob, min=eps, max=1.0 - eps)
    return -torch.sum(p * torch.log(p), dim=0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min()
    xmax = x.max()
    if (xmax - xmin).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _prepare_case_data(predictor: nnUNetPredictor, case_files: List[str]) -> Dict:
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
    if not dims:
        return patch
    return torch.flip(patch, dims=list(dims))


def _invert_aug_prob(prob: torch.Tensor, dims_patch: Tuple[int, ...]) -> torch.Tensor:
    if not dims_patch:
        return prob
    dim_map = {2: 1, 3: 2, 4: 3}
    return torch.flip(prob, dims=[dim_map[d] for d in dims_patch])


def _erode_mask_3d(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    pooled = F.max_pool3d((1.0 - mask.float())[None, None], kernel_size, stride=1, padding=pad)
    return (1.0 - pooled[0, 0]).bool()


def _dilate_mask_3d(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    kernel_size = radius * 2 + 1
    pooled = F.max_pool3d(mask.float()[None, None], kernel_size, stride=1, padding=radius)
    return pooled[0, 0] > 0


def _extract_boundary_mask_3d(seg: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    fg = seg > 0
    if fg.sum() == 0:
        return torch.zeros_like(fg)
    eroded = _erode_mask_3d(fg, kernel_size=kernel_size)
    return fg & (~eroded)


def _boundary_disagreement(sample_segs: torch.Tensor) -> torch.Tensor:
    count = sample_segs.shape[0]
    strict_sum = torch.zeros_like(sample_segs[0], dtype=torch.float32)
    band_sum = torch.zeros_like(sample_segs[0], dtype=torch.float32)
    for i in range(count):
        core = _extract_boundary_mask_3d(sample_segs[i])
        band = _dilate_mask_3d(core, radius=_BOUNDARY_BAND_RADIUS)
        strict_sum += core.float()
        band_sum += band.float()
    strict_ratio = strict_sum / float(count)
    band_ratio = band_sum / float(count)
    return 0.35 * strict_ratio + 0.65 * band_ratio


def _build_hybrid_maps(
    mean_prob: torch.Tensor,
    tta_var: torch.Tensor,
    mc_var: torch.Tensor,
    mutual_info: torch.Tensor,
    boundary_disagree: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_entropy = _entropy_map(mean_prob)
    tta_norm = _normalize(tta_var)
    mc_norm = _normalize(mc_var)
    mi_norm = _normalize(mutual_info)
    ent_norm = _normalize(pred_entropy)

    top2 = torch.topk(mean_prob, k=min(2, mean_prob.shape[0]), dim=0).values
    if top2.shape[0] < 2:
        ambiguity = torch.zeros_like(top2[0])
    else:
        ambiguity = _normalize(1.0 - (top2[0] - top2[1]))

    boundary_gate = torch.where(
        boundary_disagree > _BOUNDARY_GATE_THRESHOLD,
        torch.pow(torch.clamp(boundary_disagree, 0.0, 1.0), 1.4),
        torch.zeros_like(boundary_disagree),
    )

    foreground = (mean_prob.argmax(dim=0) > 0)
    foreground_band = _dilate_mask_3d(foreground, radius=1).float()

    risk_core = (
        0.48 * tta_norm
        + 0.22 * mc_norm
        + 0.18 * mi_norm
        + 0.12 * ambiguity
    )
    risk_tail = risk_core * (1.0 + 0.85 * boundary_gate * (0.5 + 0.5 * ambiguity))
    q = torch.quantile(risk_tail, _TAIL_QUANTILE)
    top_tail = (risk_tail >= q).float()
    variance_map = risk_tail * (1.0 + 0.10 * top_tail)
    variance_map = variance_map * (0.25 + 0.75 * foreground_band)

    entropy_map = (
        0.55 * tta_norm
        + 0.20 * ent_norm
        + 0.15 * mi_norm
        + 0.10 * boundary_gate
    )
    entropy_map = entropy_map * (1.0 + 0.30 * boundary_gate)
    entropy_map = entropy_map * (0.25 + 0.75 * foreground_band)

    return variance_map.to(torch.float32), entropy_map.to(torch.float32)


def _predict_hybrid_prob_and_uncertainty(
    predictor: nnUNetPredictor,
    data_tensor: torch.Tensor,
    tta_mode: str,
    num_mc_samples: int,
    progress_mode: str,
    plain_log_every_patches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = predictor.device
    network = predictor.network.to(device)
    network.eval()
    enable_mc_dropout(get_unwrapped_network(network))

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
        accum_entropy = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_tta_var = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_mc_var = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_mi = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_boundary = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        accum_count = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)

        gaussian = compute_gaussian(
            tuple(predictor.configuration_manager.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10.0,
            device=device,
        ).to(torch.float32)

        iterator = slicers
        if progress_mode == 'tqdm':
            iterator = tqdm(slicers, desc='Hybrid Sliding Window', leave=False)

        total_patches = len(slicers)
        for patch_idx, sl in enumerate(iterator):
            if progress_mode == 'plain' and (
                patch_idx == 0
                or (patch_idx + 1) % max(1, plain_log_every_patches) == 0
                or (patch_idx + 1) == total_patches
            ):
                print(f'    Patch progress: {patch_idx + 1}/{total_patches}', flush=True)

            patch = x[sl][None].contiguous()
            per_view_mean_probs = []
            per_view_mc_var = []
            all_probs = []
            all_segs = []

            for dims_patch in aug_dims_list:
                mc_probs = []
                patch_aug = _apply_aug_patch(patch, dims_patch)
                for _ in range(num_mc_samples):
                    logits_aug = predictor._internal_maybe_mirror_and_predict(patch_aug)[0].to(torch.float32)
                    prob_aug = torch.softmax(logits_aug, dim=0)
                    prob_inv = _invert_aug_prob(prob_aug, dims_patch)
                    mc_probs.append(prob_inv)
                    all_probs.append(prob_inv)
                    all_segs.append(torch.argmax(prob_inv, dim=0))

                mc_stack = torch.stack(mc_probs, dim=0)
                per_view_mean_probs.append(mc_stack.mean(dim=0))
                per_view_mc_var.append(torch.var(mc_stack, dim=0, unbiased=False).mean(dim=0))

            view_stack = torch.stack(per_view_mean_probs, dim=0)
            sample_stack = torch.stack(all_probs, dim=0)
            seg_stack = torch.stack(all_segs, dim=0)

            mean_prob_patch = sample_stack.mean(dim=0)
            tta_var_patch = torch.var(view_stack, dim=0, unbiased=False).mean(dim=0)
            mc_var_patch = torch.stack(per_view_mc_var, dim=0).mean(dim=0)
            pred_entropy_patch = _entropy_map(mean_prob_patch)
            sample_entropy_patch = -torch.sum(
                torch.clamp(sample_stack, min=1e-8, max=1.0 - 1e-8)
                * torch.log(torch.clamp(sample_stack, min=1e-8, max=1.0 - 1e-8)),
                dim=1,
            )
            exp_entropy_patch = sample_entropy_patch.mean(dim=0)
            mutual_info_patch = torch.clamp(pred_entropy_patch - exp_entropy_patch, min=0.0)
            boundary_patch = _boundary_disagreement(seg_stack)

            hybrid_var_patch, hybrid_entropy_patch = _build_hybrid_maps(
                mean_prob_patch,
                tta_var_patch,
                mc_var_patch,
                mutual_info_patch,
                boundary_patch,
            )

            accum_prob[sl] += mean_prob_patch * gaussian
            accum_var[sl[1:]] += hybrid_var_patch * gaussian
            accum_entropy[sl[1:]] += hybrid_entropy_patch * gaussian
            accum_tta_var[sl[1:]] += tta_var_patch * gaussian
            accum_mc_var[sl[1:]] += mc_var_patch * gaussian
            accum_mi[sl[1:]] += mutual_info_patch * gaussian
            accum_boundary[sl[1:]] += boundary_patch * gaussian
            accum_count[sl[1:]] += gaussian

        accum_count = torch.clamp(accum_count, min=1e-8)
        mean_prob = accum_prob / accum_count.unsqueeze(0)
        hybrid_var = accum_var / accum_count
        hybrid_entropy = accum_entropy / accum_count
        tta_var = accum_tta_var / accum_count
        mc_var = accum_mc_var / accum_count
        mutual_info = accum_mi / accum_count
        boundary = accum_boundary / accum_count

        sl = slicer_revert_padding[1:]
        mean_prob = mean_prob[(slice(None), *sl)]
        hybrid_var = hybrid_var[sl]
        hybrid_entropy = hybrid_entropy[sl]
        tta_var = tta_var[sl]
        mc_var = mc_var[sl]
        mutual_info = mutual_info[sl]
        boundary = boundary[sl]

        empty_cache(device)
        return (
            mean_prob.cpu().numpy(),
            hybrid_var.cpu().numpy(),
            hybrid_entropy.cpu().numpy(),
            tta_var.cpu().numpy(),
            mc_var.cpu().numpy(),
            mutual_info.cpu().numpy(),
            boundary.cpu().numpy(),
        )


def _save_outputs(
    mean_prob: np.ndarray,
    variance_map: np.ndarray,
    entropy_map: np.ndarray,
    tta_var_map: np.ndarray,
    mc_var_map: np.ndarray,
    mutual_info_map: np.ndarray,
    boundary_map: np.ndarray,
    properties: Dict,
    output_folder: str,
    case_id: str,
    save_mode: str,
) -> Dict[str, float]:
    seg = np.argmax(mean_prob, axis=0).astype(np.uint8)
    variance = variance_map.astype(np.float32)
    entropy = entropy_map.astype(np.float32)
    tta_var = tta_var_map.astype(np.float32)
    mc_var = mc_var_map.astype(np.float32)
    mutual_info = mutual_info_map.astype(np.float32)
    boundary = boundary_map.astype(np.float32)

    bbox = properties.get('bbox_used_for_cropping')
    original_shape = properties.get('shape_before_cropping')

    if bbox is not None and original_shape is not None:
        slicer = bounding_box_to_slice(bbox)

        def _restore(arr, dtype):
            full = np.zeros(original_shape, dtype=dtype)
            full[slicer] = arr
            return full

        seg = _restore(seg, np.uint8)
        variance = _restore(variance, np.float32)
        entropy = _restore(entropy, np.float32)
        tta_var = _restore(tta_var, np.float32)
        mc_var = _restore(mc_var, np.float32)
        boundary = _restore(boundary, np.float32)
        mutual_info = _restore(mutual_info, np.float32)

    sitk.WriteImage(sitk.GetImageFromArray(seg), join(output_folder, f'{case_id}.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(variance), join(output_folder, f'{case_id}_variance.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(entropy), join(output_folder, f'{case_id}_entropy.nii.gz'))
    if save_mode == 'full':
        sitk.WriteImage(sitk.GetImageFromArray(tta_var), join(output_folder, f'{case_id}_variance_tta.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(mc_var), join(output_folder, f'{case_id}_variance_mc.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(mutual_info), join(output_folder, f'{case_id}_mutual_info.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(boundary), join(output_folder, f'{case_id}_boundary_disagree.nii.gz'))

    return {
        'case_id': case_id,
        'variance_mean': float(np.mean(variance)),
        'entropy_mean': float(np.mean(entropy)),
    }


def run(args: argparse.Namespace) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.model_folder is None:
        args.model_folder = join(
            nnUNet_results,
            'Dataset002_BraTS2021_Train/nnUNetTrainerV2_MCDropout_DecoderOnly__nnUNetPlans__3d_fullres',
        )
    _assert_expected_model_folder(args.model_folder)

    if args.input_folder is None:
        args.input_folder = join(nnUNet_raw, f'{args.dataset}/imagesTr')
    if args.output_folder is None:
        args.output_folder = join(nnUNet_results, f'{args.dataset}/Hybrid_MC_TTA_Risk')

    maybe_mkdir_p(args.output_folder)

    predictor = init_predictor(args.model_folder, args.fold, args.checkpoint_name, args.device)
    net_info = _prepare_network(
        predictor,
        dropout_p=args.dropout_p,
        decoder_only=args.inject_decoder_only,
    )
    n_mod = len(load_json(join(args.model_folder, 'dataset.json'))['channel_names'])

    if args.case_ids_file is not None:
        case_ids = get_case_ids_from_file(args.case_ids_file)
    else:
        case_ids = get_case_ids(args.input_folder, args.case_ids)
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]

    print(f'Using {len(case_ids)} cases (save_mode={args.save_mode})', flush=True)
    if args.case_ids_file is not None:
        print(f'Case list file: {args.case_ids_file}', flush=True)

    results = []
    latency_rows = []

    completed_case_ids = set()
    resumed_from_progress = 0
    resumed_from_backfill = 0
    if args.resume:
        progress_rows = _load_progress_rows(args.output_folder)
        for r in progress_rows:
            cid = r['case_id']
            if cid in completed_case_ids:
                continue
            completed_case_ids.add(cid)
            results.append({
                'case_id': cid,
                'variance_mean': float(r.get('variance_mean', float('nan'))),
                'entropy_mean': float(r.get('entropy_mean', float('nan'))),
            })
            latency_rows.append({
                'case_id': cid,
                'latency_sec': float(r.get('latency_sec', float('nan'))),
                'is_warmup': bool(r.get('is_warmup', False)),
            })
        resumed_from_progress = len(completed_case_ids)

        completed_from_files = set(_detect_completed_case_ids_from_outputs(args.output_folder, case_ids))
        missing_progress = sorted(list(completed_from_files - completed_case_ids))
        if missing_progress:
            backfill_stats = _build_stats_from_existing_outputs(args.output_folder, missing_progress)
            for s in backfill_stats:
                cid = s['case_id']
                if cid in completed_case_ids:
                    continue
                completed_case_ids.add(cid)
                results.append(s)
                latency_rows.append({
                    'case_id': cid,
                    'latency_sec': float('nan'),
                    'is_warmup': False,
                })
                _append_progress_row(args.output_folder, {
                    'case_id': cid,
                    'variance_mean': s['variance_mean'],
                    'entropy_mean': s['entropy_mean'],
                    'latency_sec': float('nan'),
                    'is_warmup': False,
                })
            resumed_from_backfill = len(backfill_stats)

    pending_case_ids = [c for c in case_ids if c not in completed_case_ids]
    if args.resume:
        print(
            f'Resume enabled: completed={len(completed_case_ids)}, pending={len(pending_case_ids)} '
            f'(from_progress={resumed_from_progress}, from_outputs={resumed_from_backfill})',
            flush=True,
        )

    start_done = len(case_ids) - len(pending_case_ids)

    for run_idx, cid in enumerate(pending_case_ids, start=1):
        overall_idx = start_done + run_idx
        print(f'\n[{overall_idx}/{len(case_ids)}] {cid}', flush=True)
        _sync_if_cuda(predictor.device)
        t0 = time.perf_counter()
        case_files = [join(args.input_folder, f'{cid}_{i:04d}.nii.gz') for i in range(n_mod)]
        batch = _prepare_case_data(predictor, case_files)
        (
            mean_prob,
            variance_map,
            entropy_map,
            tta_var_map,
            mc_var_map,
            mutual_info_map,
            boundary_map,
        ) = _predict_hybrid_prob_and_uncertainty(
            predictor,
            batch['data'],
            tta_mode=args.tta_mode,
            num_mc_samples=args.num_mc_samples,
            progress_mode=args.progress_mode,
            plain_log_every_patches=args.plain_log_every_patches,
        )
        stats = _save_outputs(
            mean_prob,
            variance_map,
            entropy_map,
            tta_var_map,
            mc_var_map,
            mutual_info_map,
            boundary_map,
            batch['data_properties'],
            args.output_folder,
            cid,
            args.save_mode,
        )
        results.append(stats)
        _sync_if_cuda(predictor.device)
        elapsed = float(time.perf_counter() - t0)
        latency_row = {
            'case_id': cid,
            'latency_sec': elapsed,
            'is_warmup': overall_idx <= args.warmup_cases,
        }
        latency_rows.append(latency_row)
        _append_progress_row(args.output_folder, {
            'case_id': cid,
            'variance_mean': stats['variance_mean'],
            'entropy_mean': stats['entropy_mean'],
            'latency_sec': elapsed,
            'is_warmup': overall_idx <= args.warmup_cases,
        })
        _write_latency_csv(args.output_folder, latency_rows)
        print(f'  Latency: {elapsed:.1f}s', flush=True)
        print(f'  variance_mean: {stats["variance_mean"]:.6f}', flush=True)
        print(f'  entropy_mean:  {stats["entropy_mean"]:.6f}', flush=True)

    latency_csv = _write_latency_csv(args.output_folder, latency_rows)
    latency_summary = _summarize_latency(latency_rows)
    summary = {
        'version': 'hybrid_mc_tta_risk_v1',
        'dataset': args.dataset,
        'model_folder': args.model_folder,
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'num_cases': len(case_ids),
        'num_cases_completed': len(results),
        'num_cases_remaining': max(0, len(case_ids) - len(results)),
        'resume_enabled': bool(args.resume),
        'resumed_from_progress': int(resumed_from_progress),
        'resumed_from_outputs': int(resumed_from_backfill),
        'num_mc_samples': int(args.num_mc_samples),
        'tta_mode': args.tta_mode,
        'warmup_cases': int(args.warmup_cases),
        'latency_csv': latency_csv,
        **latency_summary,
        **net_info,
        'results': results,
    }
    save_json(summary, join(args.output_folder, 'summary.json'))
    print('Done. Summary:', join(args.output_folder, 'summary.json'), flush=True)


if __name__ == '__main__':
    run(parse_args())