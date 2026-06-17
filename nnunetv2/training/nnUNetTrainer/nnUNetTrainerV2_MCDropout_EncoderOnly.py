import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import isfile, join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerV2_MCDropout_EncoderOnly(nnUNetTrainer):
    """
    nnU-Net Trainer with MC Dropout ONLY in Encoder for uncertainty estimation.

    This trainer injects Dropout3d layers only into the encoder part of the
    network. Dropout is active during training and can be used during inference
    for MC Dropout.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dropout_p = 0.5
        self.save_every = 1

    def initialize(self):
        if not self.was_initialized:
            super().initialize()

            self.print_to_log_file("\n" + "=" * 60)
            self.print_to_log_file("[MC Dropout] Injecting MC Dropout layers (ENCODER ONLY)...")
            dropout_count = self._inject_dropout_to_network()
            self.print_to_log_file(f"[SUCCESS] Injected {dropout_count} Dropout3d layers in ENCODER (p={self.dropout_p})")
            self.print_to_log_file("=" * 60 + "\n")

            if dropout_count == 0:
                raise RuntimeError("[ERROR] Dropout injection failed! No StackedConvBlocks found.")

            self._print_network_structure_sample()
            self._maybe_resume_from_latest_checkpoint()

    def _inject_dropout_to_network(self):
        """
        Inject Dropout3d only into encoder-side StackedConvBlocks.

        Returns:
            int: Number of injected dropout layers.
        """
        count = 0
        decoder_skipped = 0

        network = self._get_actual_network()

        for name, module in network.named_modules():
            if module.__class__.__name__ == 'StackedConvBlocks':
                is_in_decoder = 'decoder' in name.lower()

                if not is_in_decoder:
                    for block in module.children():
                        if isinstance(block, nn.Sequential):
                            has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                                              for m in block.modules())
                            if not has_dropout:
                                dropout_layer = nn.Dropout3d(p=self.dropout_p, inplace=False)
                                block.add_module("mc_dropout", dropout_layer)
                                count += 1
                else:
                    decoder_skipped += sum(1 for block in module.children()
                                           if isinstance(block, nn.Sequential))

        self.print_to_log_file(f"[INFO] Skipped {decoder_skipped} blocks in DECODER (no dropout)")
        return count

    def _get_actual_network(self):
        """
        Get the underlying network module, handling DDP and torch.compile wrappers.
        """
        from torch.nn.parallel import DistributedDataParallel
        from torch._dynamo import OptimizedModule

        network = self.network

        if isinstance(network, DistributedDataParallel):
            network = network.module

        if isinstance(network, OptimizedModule):
            network = network._orig_mod

        return network

    def _print_network_structure_sample(self):
        """
        Print a small network structure sample to verify encoder dropout injection.
        """
        network = self._get_actual_network()

        self.print_to_log_file("\n[Network Structure] Sample to verify Dropout injection:")
        self.print_to_log_file("-" * 60)

        encoder_blocks_printed = 0
        decoder_blocks_printed = 0

        for name, module in network.named_modules():
            if module.__class__.__name__ == 'StackedConvBlocks':
                is_in_decoder = 'decoder' in name.lower()

                if not is_in_decoder and encoder_blocks_printed < 2:
                    self.print_to_log_file(f"\n[ENCODER] {name}")
                    for block_name, block in module.named_children():
                        if isinstance(block, nn.Sequential):
                            self.print_to_log_file(f"  |-- {block_name}:")
                            for layer_name, layer in block.named_children():
                                marker = " <-- DROPOUT" if "dropout" in layer_name.lower() else ""
                                self.print_to_log_file(f"      |-- {layer_name}: {layer.__class__.__name__}{marker}")
                            encoder_blocks_printed += 1
                            if encoder_blocks_printed >= 2:
                                break

                elif is_in_decoder and decoder_blocks_printed < 2:
                    self.print_to_log_file(f"\n[DECODER] {name}")
                    for block_name, block in module.named_children():
                        if isinstance(block, nn.Sequential):
                            self.print_to_log_file(f"  |-- {block_name}:")
                            for layer_name, layer in block.named_children():
                                self.print_to_log_file(f"      |-- {layer_name}: {layer.__class__.__name__}")
                            decoder_blocks_printed += 1
                            if decoder_blocks_printed >= 2:
                                break

                if encoder_blocks_printed >= 2 and decoder_blocks_printed >= 2:
                    break

        self.print_to_log_file("-" * 60 + "\n")

    def _maybe_resume_from_latest_checkpoint(self):
        """Resume training from checkpoint_latest.pth if it already exists."""
        latest_checkpoint = join(self.output_folder, 'checkpoint_latest.pth')
        if isfile(latest_checkpoint):
            self.print_to_log_file(f"[RESUME] Found existing checkpoint: {latest_checkpoint}")
            self.load_checkpoint(latest_checkpoint)
            self.print_to_log_file(f"[RESUME] Resumed from epoch {self.current_epoch}")

    def on_train_start(self):
        """Ensure the network is in training mode at the start of training."""
        super().on_train_start()

        dropout_count = sum(1 for m in self.network.modules()
                            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))
        self.print_to_log_file(f"\n[Training Start] Number of Dropout layers in model: {dropout_count}\n")

    def on_train_epoch_end(self, train_outputs: list):
        """Log dropout status periodically during training."""
        super().on_train_epoch_end(train_outputs)

        if self.current_epoch % 10 == 0:
            dropout_count = sum(1 for m in self.network.modules()
                                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))
            self.print_to_log_file(f"[Epoch {self.current_epoch}] 模型中Dropout层数量: {dropout_count}")

    def on_validation_epoch_end(self, val_outputs: list):
        """Switch back to training mode after validation."""
        super().on_validation_epoch_end(val_outputs)
        self.network.train()