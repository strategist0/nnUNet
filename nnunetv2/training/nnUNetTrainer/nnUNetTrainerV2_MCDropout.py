import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.modules.module import _IncompatibleKeys


class nnUNetTrainerV2_MCDropout(nnUNetTrainer):
    """
    nnU-Net Trainer with MC Dropout for uncertainty estimation
    
    This trainer injects Dropout3d layers into the network for uncertainty estimation.
    Dropout is active during training and can be used during inference for MC Dropout.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # 设置dropout概率 - 只在decoder中注入
        self.dropout_p = 0.5

    def initialize(self):
        if not self.was_initialized:
            super().initialize()

            # 注入MC Dropout层（只在decoder）
            self.print_to_log_file("\n" + "=" * 60)
            self.print_to_log_file("[MC Dropout] Injecting MC Dropout layers (DECODER ONLY)...")
            dropout_count = self._inject_dropout_to_network()
            self.print_to_log_file(f"[SUCCESS] Injected {dropout_count} Dropout3d layers in DECODER (p={self.dropout_p})")
            self.print_to_log_file("=" * 60 + "\n")

            # 验证注入是否成功
            if dropout_count == 0:
                raise RuntimeError("[ERROR] Dropout injection failed! No StackedConvBlocks found.")
            
            # 打印网络详细信息以验证
            self._print_network_structure_sample()

    def _inject_dropout_to_network(self):
        """
        只在nnUNet的Decoder部分的卷积块中注入Dropout3d层
        
        Returns:
            int: 成功注入的dropout层数量
        """
        count = 0
        encoder_skipped = 0
        
        # 获取实际的网络模块（处理DDP和torch.compile包装）
        network = self._get_actual_network()

        # 遍历网络的所有模块，获取完整路径名
        for name, module in network.named_modules():
            # 查找StackedConvBlocks（nnUNet的主要构建块）
            if module.__class__.__name__ == 'StackedConvBlocks':
                # 检查是否在decoder中（通过模块路径名判断）
                is_in_decoder = 'decoder' in name.lower()
                
                if is_in_decoder:
                    # 只在decoder中注入dropout
                    for block in module.children():
                        # block是一个nn.Sequential
                        if isinstance(block, nn.Sequential):
                            # 检查是否已经有dropout（避免重复添加）
                            has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                                              for m in block.modules())

                            if not has_dropout:
                                # 在Sequential的最后添加Dropout3d
                                dropout_layer = nn.Dropout3d(p=self.dropout_p, inplace=False)
                                block.add_module("mc_dropout", dropout_layer)
                                count += 1
                else:
                    # 统计跳过的encoder块数量
                    encoder_skipped += sum(1 for block in module.children() 
                                          if isinstance(block, nn.Sequential))
        
        self.print_to_log_file(f"[INFO] Skipped {encoder_skipped} blocks in ENCODER (no dropout)")
        return count
    
    def _get_actual_network(self):
        """
        获取实际的网络模块（处理DDP和torch.compile的包装）
        
        Returns:
            nn.Module: 实际的网络模块
        """
        from torch.nn.parallel import DistributedDataParallel
        from torch._dynamo import OptimizedModule
        
        network = self.network
        
        # 处理DDP包装
        if isinstance(network, DistributedDataParallel):
            network = network.module
            
        # 处理torch.compile包装
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
            
        return network

    def _print_network_structure_sample(self):
        """
        打印网络结构样本，用于验证dropout注入（区分encoder和decoder）
        """
        network = self._get_actual_network()
        
        self.print_to_log_file("\n[Network Structure] Sample to verify Dropout injection:")
        self.print_to_log_file("-" * 60)
        
        encoder_blocks_printed = 0
        decoder_blocks_printed = 0
        
        for name, module in network.named_modules():
            if module.__class__.__name__ == 'StackedConvBlocks':
                is_in_decoder = 'decoder' in name.lower()
                
                # 打印encoder样本（前2个）
                if not is_in_decoder and encoder_blocks_printed < 2:
                    self.print_to_log_file(f"\n[ENCODER] {name}")
                    for block_name, block in module.named_children():
                        if isinstance(block, nn.Sequential):
                            self.print_to_log_file(f"  |-- {block_name}:")
                            for layer_name, layer in block.named_children():
                                self.print_to_log_file(f"      |-- {layer_name}: {layer.__class__.__name__}")
                            encoder_blocks_printed += 1
                            if encoder_blocks_printed >= 2:
                                break
                
                # 打印decoder样本（前2个）
                elif is_in_decoder and decoder_blocks_printed < 2:
                    self.print_to_log_file(f"\n[DECODER] {name}")
                    for block_name, block in module.named_children():
                        if isinstance(block, nn.Sequential):
                            self.print_to_log_file(f"  |-- {block_name}:")
                            for layer_name, layer in block.named_children():
                                marker = " <-- DROPOUT" if "dropout" in layer_name.lower() else ""
                                self.print_to_log_file(f"      |-- {layer_name}: {layer.__class__.__name__}{marker}")
                            decoder_blocks_printed += 1
                            if decoder_blocks_printed >= 2:
                                break
                
                if encoder_blocks_printed >= 2 and decoder_blocks_printed >= 2:
                    break
        
        self.print_to_log_file("-" * 60 + "\n")

    def on_train_start(self):
        """
        训练开始时的处理，确保网络处于训练模式
        """
        super().on_train_start()
        
        # 验证dropout层数量
        dropout_count = sum(1 for m in self.network.modules()
                          if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))
        self.print_to_log_file(f"\n[Training Start] Number of Dropout layers in model: {dropout_count}\n")

    def on_train_epoch_end(self, train_outputs: list):
        """
        每个epoch结束时的处理
        """
        super().on_train_epoch_end(train_outputs)

        # 每10个epoch打印一次dropout状态
        if self.current_epoch % 10 == 0:
            dropout_count = sum(1 for m in self.network.modules()
                              if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))
            self.print_to_log_file(f"[Epoch {self.current_epoch}] 模型中Dropout层数量: {dropout_count}")
    
    def on_validation_epoch_end(self, val_outputs: list):
        """
        验证epoch结束后的处理
        """
        super().on_validation_epoch_end(val_outputs)
        
        # 验证后切回训练模式
        self.network.train()