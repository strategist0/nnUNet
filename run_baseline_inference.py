import torch
import os
import time
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles


def run_baseline_inference():
    """
    用baseline模型（不带MC Dropout）对训练集做预测
    """

    # ========== 配置参数 ⭐ 可以在这里调整 ==========
    NUM_CASES = 5  # ⭐ 改这里：5=测试, 30=论文, 50=扩展, -1=全部
    USE_CHECKPOINT = 'checkpoint_best.pth'  # 或 'checkpoint_final.pth'

    # ========== 路径配置 ==========
    baseline_model_folder = r'E:\nnUNet\nnUNet_results\Dataset001_BraTS2021\nnUNetTrainer__nnUNetPlans__3d_fullres'
    input_folder = join(nnUNet_raw, 'Dataset001_BraTS2021/imagesTr')
    output_folder = r'E:\nnUNet\nnUNet_results\Dataset001_BraTS2021\baseline_predictions'
    maybe_mkdir_p(output_folder)

    # ========== 初始化预测器 ==========
    print("正在加载Baseline模型...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        baseline_model_folder,
        use_folds=(0,),
        checkpoint_name=USE_CHECKPOINT
    )

    print(f"✅ 模型加载成功！")
    print(f"   使用checkpoint: {USE_CHECKPOINT}")

    # ========== 准备输入文件列表 ==========
    all_cases = subfiles(input_folder, suffix='_0000.nii.gz', join=False)
    case_ids = [c[:-12] for c in all_cases]

    # 根据NUM_CASES选择预测数量
    if NUM_CASES == -1:
        case_ids_to_predict = case_ids  # 全部
    else:
        case_ids_to_predict = case_ids[:NUM_CASES]

    print(f"\n{'=' * 60}")
    print(f"📊 数据集总病例数: {len(case_ids)}")
    print(f"🎯 本次预测病例数: {len(case_ids_to_predict)}")
    print(f"⏱️  预计耗时: ~{len(case_ids_to_predict)}分钟")
    print(f"{'=' * 60}\n")

    # ========== 逐个预测 ==========
    start_time = time.time()

    for i, case_id in enumerate(case_ids_to_predict, 1):
        print(f"[{i}/{len(case_ids_to_predict)}] 正在预测: {case_id}", end=" ")

        # 检查是否已经预测过
        output_file = join(output_folder, f'{case_id}.nii.gz')
        if os.path.exists(output_file):
            print("⏭️  已存在")
            continue

        # 构建输入文件路径
        case_files = [[
            join(input_folder, f'{case_id}_0000.nii.gz'),
            join(input_folder, f'{case_id}_0001.nii.gz'),
            join(input_folder, f'{case_id}_0002.nii.gz'),
            join(input_folder, f'{case_id}_0003.nii.gz')
        ]]

        # 检查文件完整性
        if not all(os.path.exists(f) for f in case_files[0]):
            print("⚠️  文件不完整")
            continue

        # 预测
        try:
            predictor.predict_from_files(
                case_files,
                output_folder,
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ 预测完成！")
    print(f"⏱️  实际耗时: {elapsed / 60:.1f}分钟")
    print(f"📁 结果保存在: {output_folder}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_baseline_inference()