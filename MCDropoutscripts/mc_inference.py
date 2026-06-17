"""
MC Dropout Inference Script for nnU-Net
Performs 30 forward passes with MC Dropout and saves:
- Mean prediction
- Variance map
- Entropy map
- Latency (inference time per case)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import SimpleITK as sitk
import argparse
import time
from typing import List, Optional
from tqdm import tqdm
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, save_json, subfiles, isfile

from MCDropoututil.model_utils import inject_dropout_layers, enable_mc_dropout, get_unwrapped_network, verify_dropout_injection
from MCDropoututil.uncertainty_utils import (
    compute_predictive_uncertainty,
    get_uncertainty_stats,
    aggregate_uncertainty_in_region,
    create_tumor_mask,
    get_top_k_percent_uncertainty
)


def resolve_model_folder(path: str) -> str:
    """
    Accept either a model root folder or a fold subfolder and return the model root.
    """
    base = os.path.basename(path.rstrip(os.sep))
    if base.startswith('fold_'):
        return os.path.dirname(path)
    return path


def perform_mc_dropout_inference(
    predictor: nnUNetPredictor,
    preprocessed_data: dict,
    num_iterations: int = 30,
    enable_dropout: bool = True
) -> dict:
    """
    Perform MC Dropout inference with multiple forward passes.
    Workaround: patch network.eval() to prevent disabling Dropout during inference.
    """
    data_tensor = preprocessed_data['data']
    network = predictor.network
    
    all_predictions = []
    
    # Workaround: Save the original eval method and patch it
    original_eval = network.eval
    
    if enable_dropout:
        # During MC Dropout inference, we patch eval() to not actually disable training mode
        def eval_noop(*args, **kwargs):
            # Keep network in training mode for Dropout
            for module in network.modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                    module.train()
                elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
            return network
        
        network.eval = eval_noop
    
    try:
        for i in tqdm(range(num_iterations), desc="MC Dropout Sampling"):
            # Ensure dropout is in training mode
            if enable_dropout:
                for module in network.modules():
                    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                        module.train()
            
            with torch.no_grad():
                logits = predictor.predict_logits_from_preprocessed_data(data_tensor)
                all_predictions.append(logits.cpu().numpy())
    finally:
        # Restore original eval method
        network.eval = original_eval
        network.eval()
    
    all_predictions = np.stack(all_predictions, axis=0)
    return {
        'predictions': all_predictions,
        'properties': preprocessed_data['data_properties']
    }


def save_uncertainty_maps(
    mean_pred: np.ndarray,
    variance: np.ndarray,
    entropy: np.ndarray,
    properties: dict,
    output_folder: str,
    case_id: str
) -> None:
    """
    Save mean prediction and uncertainty maps as NIfTI files.
    """
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    
    # Convert to segmentation
    segmentation = np.argmax(mean_pred, axis=0).astype(np.uint8)
    
    # Uncrop if needed
    bbox = properties.get('bbox_used_for_cropping')
    original_shape = properties.get('shape_before_cropping')
    
    if bbox is not None and original_shape is not None:
        slicer = bounding_box_to_slice(bbox)
        seg_full = np.zeros(original_shape, dtype=np.uint8)
        var_full = np.zeros(original_shape, dtype=np.float32)
        ent_full = np.zeros(original_shape, dtype=np.float32)
        
        seg_full[slicer] = segmentation
        var_full[slicer] = variance
        ent_full[slicer] = entropy
        segmentation, variance, entropy = seg_full, var_full, ent_full
    
    # Save files
    sitk.WriteImage(sitk.GetImageFromArray(segmentation), 
                   join(output_folder, f"{case_id}.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(variance.astype(np.float32)), 
                   join(output_folder, f"{case_id}_variance.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(entropy.astype(np.float32)), 
                   join(output_folder, f"{case_id}_entropy.nii.gz"))

    print(f"Saved: {case_id}.nii.gz, {case_id}_variance.nii.gz, {case_id}_entropy.nii.gz")


def get_case_list(input_folder: str, case_ids: Optional[List[str]] = None) -> List[str]:
    """
    Get list of case IDs to process.
    """
    if case_ids is not None:
        return case_ids
    
    # Auto-discover cases
    all_files = subfiles(input_folder, suffix='_0000.nii.gz', join=False)
    case_set = set(f[:-12] for f in all_files)  # Remove _0000.nii.gz
    return sorted(list(case_set))


def process_single_case(
    case_id: str,
    predictor: nnUNetPredictor,
    input_folder: str,
    output_folder: str,
    num_modalities: int,
    num_mc_iterations: int,
    num_classes: int,
    enable_dropout: bool
) -> dict:
    """
    Process a single case with MC Dropout inference.
    Records wall-clock time for the entire inference process.
    """
    case_start_time = time.time()
    print(f"\nProcessing: {case_id}")
    
    # Load input files
    image_files = [join(input_folder, f'{case_id}_{i:04d}.nii.gz') for i in range(num_modalities)]
    
    # Preprocess
    from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
    data_iterator = preprocessing_iterator_fromfiles(
        [image_files], None, None,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_processes=1,
        pin_memory=False,
        verbose=False
    )
    preprocessed_data = next(data_iterator)
    
    # MC Dropout inference
    mc_results = perform_mc_dropout_inference(
        predictor, preprocessed_data, num_mc_iterations, enable_dropout
    )
    
    # Compute uncertainty
    mean_pred, variance, entropy = compute_predictive_uncertainty(
        logits=mc_results['predictions'],
        num_classes=num_classes,
        return_probabilities=True
    )
    
    # Save results
    save_uncertainty_maps(
        mean_pred, variance, entropy,
        mc_results['properties'],
        output_folder, case_id
    )
    
    # Get statistics
    variance_stats = get_uncertainty_stats(variance)
    entropy_stats = get_uncertainty_stats(entropy)

    # Region-focused stats (tumor only)
    segmentation = np.argmax(mean_pred, axis=0).astype(np.uint8)
    tumor_mask = create_tumor_mask(segmentation)
    variance_tumor_stats = aggregate_uncertainty_in_region(variance, tumor_mask)
    entropy_tumor_stats = aggregate_uncertainty_in_region(entropy, tumor_mask)

    # Top-k uncertainty (less dominated by background)
    variance_top10 = get_top_k_percent_uncertainty(variance, k=10.0)
    entropy_top10 = get_top_k_percent_uncertainty(entropy, k=10.0)
    
    print(f"  Variance: mean={variance_stats['mean']:.6f}, std={variance_stats['std']:.6f}")
    print(f"  Entropy:  mean={entropy_stats['mean']:.6f}, std={entropy_stats['std']:.6f}")
    print(
        f"  Variance (tumor): mean={variance_tumor_stats['mean']:.6f}, "
        f"max={variance_tumor_stats['max']:.6f}, std={variance_tumor_stats['std']:.6f}"
    )
    print(
        f"  Entropy (tumor):  mean={entropy_tumor_stats['mean']:.6f}, "
        f"max={entropy_tumor_stats['max']:.6f}, std={entropy_tumor_stats['std']:.6f}"
    )
    print(f"  Variance (top10% voxels): {variance_top10:.6f}")
    print(f"  Entropy  (top10% voxels): {entropy_top10:.6f}")
    
    case_elapsed_time = time.time() - case_start_time
    print(f"  Latency: {case_elapsed_time:.2f} seconds")
    
    return {
        'case_id': case_id,
        'variance_stats': variance_stats,
        'entropy_stats': entropy_stats,
        'variance_tumor_stats': variance_tumor_stats,
        'entropy_tumor_stats': entropy_tumor_stats,
        'variance_top10': variance_top10,
        'entropy_top10': entropy_top10,
        'latency_seconds': case_elapsed_time
    }


def run_mc_dropout_inference(args):
    """Main function to run MC Dropout inference."""
    args.model_folder = resolve_model_folder(args.model_folder)
    maybe_mkdir_p(args.output_folder)
    
    print("="*80)
    print("MC Dropout Inference for nnU-Net")
    print(f"Model: {args.model_folder}")
    print(f"Output: {args.output_folder}")
    print(f"MC iterations: {args.num_mc_iterations}")
    print("="*80)
    
    # Initialize predictor
    print("\n[1/4] Initializing predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device('cuda', args.device),
        verbose=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint_name
    )
    
    # Inject Dropout layers
    if args.inject_dropout:
        print("\n[2/4] Injecting Dropout layers...")
        unwrapped_network = get_unwrapped_network(predictor.network)
        num_injected = inject_dropout_layers(
            unwrapped_network,
            dropout_p=args.dropout_p,
            decoder_only=args.inject_decoder_only
        )
        dropout_info = verify_dropout_injection(predictor.network)
        print(f"Dropout layers: {dropout_info['total_dropout_layers']}")
    else:
        print("\n[2/4] Skipping Dropout injection (baseline mode)")
    
    # Get dataset info
    print("\n[3/4] Loading dataset information...")
    dataset_json = load_json(join(args.model_folder, 'dataset.json'))
    num_classes = len(dataset_json['labels'])
    num_modalities = len(dataset_json['channel_names'])
    
    # Get case list
    case_list = None
    if args.case_ids_file:
        with open(args.case_ids_file, 'r') as f:
            case_list = [line.strip() for line in f if line.strip()]
        # Remove file extensions if present (e.g., "BraTS2021_00002_0000.nii" -> "BraTS2021_00002_0000")
        case_list = [c.replace('_0000.nii', '').replace('_0000.nii.gz', '') for c in case_list]
    else:
        case_list = get_case_list(args.input_folder, args.case_ids)
    if args.max_cases:
        case_list = case_list[:args.max_cases]
    print(f"Processing {len(case_list)} cases")
    
    # Process cases
    print("\n[4/4] Processing cases...")
    all_results = []
    
    for case_id in case_list:
        # Check if case already processed (resume capability)
        prediction_file = join(args.output_folder, f'{case_id}.nii.gz')
        variance_file = join(args.output_folder, f'{case_id}_variance.nii.gz')
        entropy_file = join(args.output_folder, f'{case_id}_entropy.nii.gz')
        if isfile(prediction_file) and isfile(variance_file) and isfile(entropy_file):
            print(f"\n[SKIP] {case_id} (already processed)")
            # Load previous result from summary if available
            try:
                existing_summary = load_json(join(args.output_folder, 'summary.json'))
                for result in existing_summary.get('results', []):
                    if result.get('case_id') == case_id:
                        all_results.append(result)
                        break
            except:
                pass
            continue
        
        result = process_single_case(
            case_id, predictor, args.input_folder, args.output_folder,
            num_modalities, args.num_mc_iterations, num_classes,
            enable_dropout=args.enable_mc_dropout
        )
        all_results.append(result)
    
    # Save summary
    latencies = [r.get('latency_seconds', 0) for r in all_results]
    avg_latency = np.mean(latencies) if latencies else 0
    total_latency = np.sum(latencies) if latencies else 0
    
    summary = {
        'num_mc_iterations': args.num_mc_iterations,
        'total_cases': len(case_list),
        'avg_latency_seconds': float(avg_latency),
        'total_latency_seconds': float(total_latency),
        'results': all_results
    }
    save_json(summary, join(args.output_folder, 'summary.json'))
    
    print("\n" + "="*80)
    print(f"Completed! Processed {len(case_list)} cases")
    print(f"Average latency per case: {avg_latency:.2f} seconds")
    print(f"Total latency: {total_latency:.2f} seconds")
    print("="*80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MC Dropout Inference for nnU-Net')
    
    parser.add_argument('--dataset', type=str, default='Dataset001_BraTS2021',
                       help='Dataset name (e.g., Dataset001_BraTS2021_Train for 80/20 split)')
    parser.add_argument('--model_folder', type=str, default=None,
                       help='Model folder (auto-constructed from dataset if not provided)')
    parser.add_argument('--input_folder', type=str, default=None,
                       help='Input folder (auto-constructed from dataset if not provided)')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Output folder (auto-constructed from dataset if not provided)')
    parser.add_argument('--case_ids', type=str, nargs='+', default=None,
                       help='Specific case IDs (default: all)')
    parser.add_argument('--case_ids_file', type=str, default=None,
                       help='File containing case IDs (one per line)')
    parser.add_argument('--max_cases', type=int, default=None,
                       help='Max number of cases')
    parser.add_argument('--num_mc_iterations', type=int, default=30)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_final.pth')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--inject_dropout', action='store_true', default=True,
                       help='Inject Dropout layers (disable for baseline)')
    parser.add_argument('--no_inject_dropout', dest='inject_dropout', action='store_false')
    parser.add_argument('--inject_decoder_only', action='store_true', default=False,
                       help='Only inject Dropout layers into decoder blocks')
    parser.add_argument('--enable_mc_dropout', action='store_true', default=True,
                       help='Enable MC Dropout during inference')
    parser.add_argument('--disable_mc_dropout', dest='enable_mc_dropout', action='store_false')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Auto-construct paths from dataset if not explicitly provided
    if args.model_folder is None:
        args.model_folder = join(nnUNet_results, f'{args.dataset}/nnUNetTrainerV2_MCDropout__nnUNetPlans__3d_fullres')
    if args.input_folder is None:
        args.input_folder = join(nnUNet_raw, f'{args.dataset}/imagesTr')
    if args.output_folder is None:
        args.output_folder = join(nnUNet_results, f'{args.dataset}/MC_Inference_Results')
    
    run_mc_dropout_inference(args)