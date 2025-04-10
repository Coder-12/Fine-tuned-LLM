import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    """
    A simplified LoRA adapter module for a transformer layer.
    """
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        self.original_layer = original_layer
        self.lora_A = nn.Parameter(torch.randn(original_layer.out_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, original_layer.in_features))
        self.scaling = 1 / rank

    def forward(self, x):
        return self.original_layer(x) + self.scaling * (self.lora_A @ (self.lora_B @ x.T)).T

def apply_lora(model, rank: int):
    """
    Replace linear layers in the model with LoRA-adapted layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # This is a simple demonstration: in practice, you may target specific layers.
            setattr(model, name, LoRAAdapter(module, rank))
    return model
