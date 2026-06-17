"""
Layer Ensemble inference for nnU-Net (no retraining).

Method:
- Enable deep supervision to get multi-scale decoder outputs.
- Upsample each scale to patch resolution.
- Convert to probabilities and fuse with fixed scale weights.
- Build uncertainty from inter-scale disagreement (variance across scales).

Outputs per case (compatible with existing evaluation pipeline):
- {case_id}.nii.gz
- {case_id}_entropy.nii.gz
- {case_id}_variance.nii.gz
"""

import os
import sys
import argparse
import csv
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


def _normalize_case_id(case_id: str) -> str:
    case_id = case_id.strip()
    if case_id.endswith('.nii.gz'):
        case_id = case_id[:-7]
    elif case_id.endswith('.nii'):
        case_id = case_id[:-4]
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    return case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Layer Ensemble inference for nnU-Net')

    parser.add_argument('--dataset', type=str, default='Dataset001_BraTS2021_Test')
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_best.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--case_ids', nargs='+', default=None)
    parser.add_argument('--case_ids_file', type=str, default=None,
                        help='Text file with one case id per line. Takes priority over auto-discovery.')
    parser.add_argument('--max_cases', type=int, default=None)

    parser.add_argument('--num_proc_pre', type=int, default=2)
    parser.add_argument('--num_proc_export', type=int, default=2)
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Optional scale weights, e.g. --weights 0.4 0.25 0.15 0.12 0.08')
    parser.add_argument('--warmup_cases', type=int, default=5,
                        help='Number of initial cases excluded from wall-clock summary stats.')

    return parser.parse_args()


def get_case_ids(input_folder: str, requested_case_ids: List[str] = None) -> List[str]:
    if requested_case_ids is not None:
        return [_normalize_case_id(case_id) for case_id in requested_case_ids if case_id.strip()]
    return sorted([f[:-12] for f in subfiles(input_folder, suffix='_0000.nii.gz', join=False)])


def get_case_ids_from_file(case_ids_file: str) -> List[str]:
    with open(case_ids_file, 'r', encoding='utf-8') as f:
        return [_normalize_case_id(line) for line in f if line.strip()]


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


def _default_scale_weights(n_scales: int) -> torch.Tensor:
    # Favor finer scales while still using coarse context.
    if n_scales == 5:
        w = torch.tensor([0.40, 0.25, 0.15, 0.12, 0.08], dtype=torch.float32)
    else:
        # Geometric decay fallback.
        w = torch.tensor([0.5 ** i for i in range(n_scales)], dtype=torch.float32)
    w = w / torch.sum(w)
    return w


def _entropy_map(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(prob, min=eps, max=1.0)
    return -torch.sum(p * torch.log(p), dim=0)


def _prepare_case_data(
    predictor: nnUNetPredictor,
    case_files: List[str],
) -> Dict:
    # Use single-process preprocessing to avoid multiprocessing Manager issues on Windows.
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


def _predict_layer_ensemble_prob_and_var(
    predictor: nnUNetPredictor,
    data_tensor: torch.Tensor,
    weights: List[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_prob: (C, H, W, D)
      var_map:   (H, W, D) - inter-scale disagreement variance (class-averaged)
    """
    device = predictor.device
    network = predictor.network.to(device)
    network.eval()

    # Enable multi-scale decoder outputs.
    old_ds = network.decoder.deep_supervision
    network.decoder.deep_supervision = True

    try:
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

            scale_w = None

            for sl in tqdm(slicers, desc='Layer Ensemble Sliding Window', leave=False):
                patch = x[sl][None].contiguous()  # (1, C, px, py, pz)
                outputs = predictor._internal_maybe_mirror_and_predict(patch)

                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]

                if scale_w is None:
                    if weights is not None:
                        if len(weights) != len(outputs):
                            raise ValueError(
                                f'Provided weights length {len(weights)} != number of scales {len(outputs)}'
                            )
                        scale_w = torch.tensor(weights, dtype=torch.float32, device=device)
                        scale_w = scale_w / torch.sum(scale_w)
                    else:
                        scale_w = _default_scale_weights(len(outputs)).to(device)

                up_probs = []
                for out in outputs:
                    # out: (1, C, sx, sy, sz)
                    logits = out[0].to(torch.float32)
                    if logits.shape[1:] != tuple(predictor.configuration_manager.patch_size):
                        logits = F.interpolate(
                            logits.unsqueeze(0),
                            size=tuple(predictor.configuration_manager.patch_size),
                            mode='trilinear',
                            align_corners=False,
                        ).squeeze(0)
                    up_probs.append(torch.softmax(logits, dim=0))

                probs_stack = torch.stack(up_probs, dim=0)  # (S, C, px, py, pz)
                fused_prob = torch.sum(
                    probs_stack * scale_w.view(-1, 1, 1, 1, 1),
                    dim=0,
                )

                inter_scale_var = torch.var(probs_stack, dim=0, unbiased=False).mean(dim=0)  # (px,py,pz)

                accum_prob[sl] += fused_prob * gaussian
                accum_var[sl[1:]] += inter_scale_var * gaussian
                accum_count[sl[1:]] += gaussian

            accum_count = torch.clamp(accum_count, min=1e-8)
            mean_prob = accum_prob / accum_count.unsqueeze(0)
            var_map = accum_var / accum_count

            # Revert padding.
            mean_prob = mean_prob[(slice(None), *slicer_revert_padding[1:])]
            var_map = var_map[slicer_revert_padding[1:]]

            return mean_prob.cpu().numpy(), var_map.cpu().numpy()

    finally:
        network.decoder.deep_supervision = old_ds
        empty_cache(device)


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
        args.output_folder = join(nnUNet_results, f'{args.dataset}/LayerEnsemble_Inference_Results')

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
    failed_cases = []
    for idx, cid in enumerate(case_ids):
        seg_out = join(args.output_folder, f'{cid}.nii.gz')
        ent_out = join(args.output_folder, f'{cid}_entropy.nii.gz')
        var_out = join(args.output_folder, f'{cid}_variance.nii.gz')
        if os.path.isfile(seg_out) and os.path.isfile(ent_out) and os.path.isfile(var_out):
            print(f'[SKIP] {cid} already exists')
            continue

        try:
            _sync_if_cuda(predictor.device)
            t0 = time.perf_counter()
            case_files = [join(args.input_folder, f'{cid}_{i:04d}.nii.gz') for i in range(n_mod)]
            batch = _prepare_case_data(predictor, case_files)
            mean_prob, var_map = _predict_layer_ensemble_prob_and_var(
                predictor,
                batch['data'],
                args.weights,
            )
            stats = _save_outputs(
                mean_prob,
                var_map,
                batch['data_properties'],
                args.output_folder,
                cid,
            )
            results.append(stats)
            _sync_if_cuda(predictor.device)
            latency_rows.append({
                'case_id': cid,
                'latency_sec': float(time.perf_counter() - t0),
                'is_warmup': idx < args.warmup_cases,
            })
        except Exception as exc:
            print(f'[FAIL] {cid}: {exc}')
            failed_cases.append(cid)

    latency_csv = _write_latency_csv(args.output_folder, latency_rows)
    latency_summary = _summarize_latency(latency_rows)

    summary = {
        'dataset': args.dataset,
        'model_folder': args.model_folder,
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'num_cases': len(case_ids),
        'num_cases_processed': len(results),
        'num_cases_failed': len(failed_cases),
        'failed_cases': failed_cases,
        'warmup_cases': int(args.warmup_cases),
        'latency_csv': latency_csv,
        **latency_summary,
        'weights': args.weights,
        'results': results,
    }
    save_json(summary, join(args.output_folder, 'summary.json'))
    print('Done. Summary:', join(args.output_folder, 'summary.json'))


if __name__ == '__main__':
    run(parse_args())
