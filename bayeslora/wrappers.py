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
    local_reparam: bool = True,
    init_log_sigma2: float = -12.0,
    target_module_names: Optional[Iterable[str]] = None,  # e.g. ["layer3", "layer4", "fc"]
    include_conv: bool = True,
    include_linear: bool = True,
) -> nn.Module:
    """
    Recursively wrap Conv2d/Linear submodules whose *qualified name* matches any
    of the user-provided targets (substring OR regex). Example matches:
      - target "layer3" matches "layer3.0.conv1", "layer3.1.conv2", ...
      - target "fc" matches "fc"
      - target r"layer4\\..*conv2" matches only conv2 in layer4 blocks

    This function replaces child modules in-place on their parent.
    """
    # ----- imports for wrappers (expected to exist in your package scope) -----
    # from bayeslora.layers import VariationalLoRAConv2d, LoRAConv2d, VariationalLoRALinear, LoRALinear
    # (Assuming these are already available in your runtime)

    if lora_alpha is None:
        lora_alpha = rank
    variant = variant.lower()
    assert variant in {"vlora", "lora"}

    # ----- build a matcher over fully-qualified child names -----
    import re as _re

    def _compile_target_patterns(targets: Optional[Iterable[str]]):
        if not targets:
            return None, None  # match-all
        pats = []
        tokens = []
        for t in targets:
            if isinstance(t, _re.Pattern):
                pats.append(t)
            else:
                # If it looks like a regex, compile; else keep as a raw token for fast substring match.
                has_regex_meta = any(ch in str(t) for ch in r".^$*+?{}[]\|()")
                if has_regex_meta:
                    pats.append(_re.compile(str(t)))
                else:
                    tokens.append(str(t))
        return pats or None, tokens or None

    _regex_list, _token_list = _compile_target_patterns(target_module_names)

    def _is_target_fqname(fqname: str) -> bool:
        # No targets specified means "match-all eligible modules"
        if _regex_list is None and _token_list is None:
            return True
        if _token_list is not None:
            for tok in _token_list:
                if tok in fqname:
                    return True
        if _regex_list is not None:
            for rgx in _regex_list:
                if rgx.search(fqname):
                    return True
        return False

    # ----- core replacement -----
    def _wrap_conv(old: nn.Conv2d) -> nn.Module:
        in_ch, out_ch = old.in_channels, old.out_channels
        if variant == "vlora":
            new = VariationalLoRAConv2d(
                in_ch, out_ch,
                kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                dilation=old.dilation, groups=old.groups, bias=(old.bias is not None),
                r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                tie_alpha_per_rank=tie_alpha_per_rank,
                init_log_sigma2=init_log_sigma2,
                local_reparam=local_reparam,
            )
        else:
            new = LoRAConv2d(
                in_ch, out_ch,
                kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                dilation=old.dilation, groups=old.groups, bias=(old.bias is not None),
                r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
            )
        # Copy base weights/bias; preserve dtype/device
        new.base.weight.data.copy_(old.weight.data)
        if old.bias is not None and new.base.bias is not None:
            new.base.bias.data.copy_(old.bias.data)
        return new.to(old.weight.device, dtype=old.weight.dtype)

    def _wrap_linear(old: nn.Linear) -> nn.Module:
        d_in, d_out = old.in_features, old.out_features
        if variant == "vlora":
            new = VariationalLoRALinear(
                d_in, d_out,
                r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
                tie_alpha_per_rank=tie_alpha_per_rank,
                init_log_sigma2=init_log_sigma2,
                local_reparam=local_reparam,
            )
        else:
            new = LoRALinear(
                d_in, d_out,
                r=rank, lora_alpha=lora_alpha, freeze_base=freeze_base,
            )
        new.base.weight.data.copy_(old.weight.data)
        if old.bias is not None and new.base.bias is not None:
            new.base.bias.data.copy_(old.bias.data)
        return new.to(old.weight.device, dtype=old.weight.dtype)

    # Replace a named child on its parent
    def _replace_child(parent: nn.Module, child_name: str, new_child: nn.Module):
        # parent._modules maps attribute name -> child module (covers Sequential, etc.)
        parent._modules[child_name] = new_child

    # Walk the tree once; replace eligible leaves.
    # We iterate over named_modules() to get each parent path; then over named_children() for direct children.
    for parent_path, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            fqname = f"{parent_path}.{child_name}" if parent_path else child_name
            if not _is_target_fqname(fqname):
                continue
            if isinstance(child, nn.Conv2d) and include_conv:
                new_child = _wrap_conv(child)
                _replace_child(parent, child_name, new_child)
            elif isinstance(child, nn.Linear) and include_linear:
                new_child = _wrap_linear(child)
                _replace_child(parent, child_name, new_child)

    return model

