"""
Model utilities for MC Dropout inference.
Handles Dropout injection and activation.
"""
import torch
import torch.nn as nn
from typing import Optional


def inject_dropout_layers(
    network: nn.Module,
    dropout_p: float = 0.3,
    decoder_only: bool = False
) -> int:
    """
    Inject Dropout3d layers into StackedConvBlocks of the network.

    Args:
        network: The neural network (unwrapped from DDP/compile if needed)
        dropout_p: Dropout probability
        decoder_only: If True, only inject into decoder blocks

    Returns:
        int: Number of Dropout layers injected
    """
    count = 0
    for name, module in network.named_modules():
        if module.__class__.__name__ == 'StackedConvBlocks':
            if decoder_only and 'decoder' not in name.lower():
                continue
            for block in module.children():
                if isinstance(block, nn.Sequential):
                    # Check if dropout already exists
                    has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                                      for m in block.modules())

                    if not has_dropout:
                        dropout_layer = nn.Dropout3d(p=dropout_p, inplace=False)
                        block.add_module("mc_dropout", dropout_layer)
                        count += 1
    return count


def enable_mc_dropout(network: nn.Module) -> None:
    """
    Enable MC Dropout mode: set all Dropout layers to training mode
    while keeping BatchNorm in eval mode.
    
    Args:
        network: The neural network
    """
    for module in network.modules():
        # Keep BatchNorm in eval mode
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
        # Enable all Dropout layers
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def get_unwrapped_network(network: nn.Module) -> nn.Module:
    """
    Unwrap network from DDP or torch.compile wrappers.
    
    Args:
        network: Potentially wrapped network
        
    Returns:
        Unwrapped network module
    """
    from torch.nn.parallel import DistributedDataParallel
    from torch._dynamo import OptimizedModule
    
    # Unwrap DDP
    if isinstance(network, DistributedDataParallel):
        network = network.module
        
    # Unwrap torch.compile
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
        
    return network


def verify_dropout_injection(network: nn.Module) -> dict:
    """
    Verify that Dropout layers are correctly injected.
    
    Args:
        network: The neural network
        
    Returns:
        dict: Statistics about Dropout layers
    """
    dropout_count = 0
    dropout_modules = []
    
    for name, module in network.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dropout_count += 1
            dropout_modules.append({
                'name': name,
                'type': module.__class__.__name__,
                'p': module.p,
                'training': module.training
            })
    
    return {
        'total_dropout_layers': dropout_count,
        'dropout_modules': dropout_modules
    }
