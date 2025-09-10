from typing import Iterable, Optional
import torch.nn as nn

from .layers import (
    LoRALinear, LoRAConv2d,
    VariationalLoRALinear, VariationalLoRAConv2d
)


def _is_target(name: str, targets: Optional[Iterable[str]]) -> bool:
    if not targets:
        return True
    return any(t in name for t in targets)


def apply_lora_adapters(
    model: nn.Module,
    variant: str = "vlora",                  # "vlora" or "lora"
    rank: int = 8,
    lora_alpha: float = None,                # default to rank if None
    freeze_base: bool = True,
    tie_alpha_per_rank: bool = True,
    local_reparam: bool = False,
    init_log_sigma2: float = -12.0,
    target_module_names: Optional[Iterable[str]] = None,  # e.g. ["layer3", "layer4", "fc"]
    include_conv: bool = True,
    include_linear: bool = True,
) -> nn.Module:
    """
    In-place wraps Conv2d/Linear modules with (Bayes)LoRA adapters.
    """
    if lora_alpha is None:
        lora_alpha = rank
    variant = variant.lower()
    assert variant in {"vlora", "lora"}

    def replace(parent: nn.Module, name: str, old: nn.Module):
        if isinstance(old, nn.Conv2d) and include_conv and _is_target(name, target_module_names):
            in_ch, out_ch = old.in_channels, old.out_channels
            if variant == "vlora":
                new = VariationalLoRAConv2d(
                    in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                    dilation=old.dilation, groups=old.groups, bias=(old.bias is not None),
                    r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                    tie_alpha_per_rank=tie_alpha_per_rank,
                    init_log_sigma2=init_log_sigma2,
                    local_reparam=local_reparam,
                )
            else:
                new = LoRAConv2d(
                    in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                    dilation=old.dilation, groups=old.groups, bias=(old.bias is not None),
                    r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                )
            # copy base weights
            new.base.weight.data.copy_(old.weight.data)
            if old.bias is not None and new.base.bias is not None:
                new.base.bias.data.copy_(old.bias.data)
            setattr(parent, name, new)

        elif isinstance(old, nn.Linear) and include_linear and _is_target(name, target_module_names):
            d_in, d_out = old.in_features, old.out_features
            if variant == "vlora":
                new = VariationalLoRALinear(
                    d_in, d_out, r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                    tie_alpha_per_rank=tie_alpha_per_rank,
                    init_log_sigma2=init_log_sigma2,
                    local_reparam=local_reparam,
                )
            else:
                new = LoRALinear(
                    d_in, d_out, r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                )
            # copy base weights
            new.base.weight.data.copy_(old.weight.data)
            if old.bias is not None and new.base.bias is not None:
                new.base.bias.data.copy_(old.bias.data)
            setattr(parent, name, new)

    # walk modules, replace leafs
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            replace(module, child_name, child)
    return model
