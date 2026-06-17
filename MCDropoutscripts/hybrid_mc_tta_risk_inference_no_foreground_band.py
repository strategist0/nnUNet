"""
Foreground-band ablation for hybrid MC+TTA uncertainty inference.

This wrapper keeps the original inference pipeline intact and only removes the
foreground-band constraint terms from the hybrid fusion stage.
"""

from __future__ import annotations

import torch

import hybrid_mc_tta_risk_inference as base


def _build_hybrid_maps_no_foreground_band(
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

    boundary_gate = torch.where(
        boundary_disagree > base._BOUNDARY_GATE_THRESHOLD,
        torch.pow(torch.clamp(boundary_disagree, 0.0, 1.0), 1.4),
        torch.zeros_like(boundary_disagree),
    )

    risk_core = (
        0.48 * tta_norm
        + 0.22 * mc_norm
        + 0.18 * mi_norm
        + 0.12 * ambiguity
    )
    risk_tail = risk_core * (1.0 + 0.85 * boundary_gate * (0.5 + 0.5 * ambiguity))
    q = torch.quantile(risk_tail, base._TAIL_QUANTILE)
    top_tail = (risk_tail >= q).float()
    variance_map = risk_tail * (1.0 + 0.10 * top_tail)

    entropy_map = (
        0.55 * tta_norm
        + 0.20 * ent_norm
        + 0.15 * mi_norm
        + 0.10 * boundary_gate
    )
    entropy_map = entropy_map * (1.0 + 0.30 * boundary_gate)

    return variance_map.to(torch.float32), entropy_map.to(torch.float32)


def main() -> None:
    base._build_hybrid_maps = _build_hybrid_maps_no_foreground_band
    args = base.parse_args()
    if args.output_folder is None:
        args.output_folder = base.join(
            base.nnUNet_results,
            f'{args.dataset}/Hybrid_MC_TTA_Risk_NoForegroundBand',
        )
    base.run(args)


if __name__ == '__main__':
    main()