"""
Boundary-gate ablation for hybrid MC+TTA uncertainty inference.

This wrapper keeps the original inference pipeline intact and only removes the
explicit boundary-gating terms from the hybrid fusion stage.
"""

from __future__ import annotations

import torch

import hybrid_mc_tta_risk_inference as base


def _build_hybrid_maps_no_boundary_gate(
    mean_prob: torch.Tensor,
    tta_var: torch.Tensor,
    mc_var: torch.Tensor,
    mutual_info: torch.Tensor,
    boundary_disagree: torch.Tensor,
):
    pred_entropy = base._entropy_map(mean_prob)
    tta_norm = base._normalize(tta_var)
    mc_norm = base._normalize(mc_var)
    mi_norm = base._normalize(mutual_info)
    ent_norm = base._normalize(pred_entropy)

    top2 = torch.topk(mean_prob, k=min(2, mean_prob.shape[0]), dim=0).values
    if top2.shape[0] < 2:
        ambiguity = torch.zeros_like(top2[0])
    else:
        ambiguity = base._normalize(1.0 - (top2[0] - top2[1]))

    foreground = mean_prob.argmax(dim=0) > 0
    foreground_band = base._dilate_mask_3d(foreground, radius=1).float()

    risk_core = (
        0.48 * tta_norm
        + 0.22 * mc_norm
        + 0.18 * mi_norm
        + 0.12 * ambiguity
    )
    q = torch.quantile(risk_core, base._TAIL_QUANTILE)
    top_tail = (risk_core >= q).float()
    variance_map = risk_core * (1.0 + 0.10 * top_tail)
    variance_map = variance_map * (0.25 + 0.75 * foreground_band)

    entropy_map = (
        0.55 * tta_norm
        + 0.20 * ent_norm
        + 0.15 * mi_norm
    )
    entropy_map = entropy_map * (0.25 + 0.75 * foreground_band)

    return variance_map.to(torch.float32), entropy_map.to(torch.float32)


def main() -> None:
    base._build_hybrid_maps = _build_hybrid_maps_no_boundary_gate
    args = base.parse_args()
    if args.output_folder is None:
        args.output_folder = base.join(
            base.nnUNet_results,
            f'{args.dataset}/Hybrid_MC_TTA_Risk_NoBoundaryGate',
        )
    base.run(args)


if __name__ == '__main__':
    main()