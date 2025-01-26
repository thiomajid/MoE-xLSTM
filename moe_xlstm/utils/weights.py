from dataclasses import dataclass

from torch import nn

from moe_xlstm.modules.model import MoExLSTMForCausalLM


@dataclass
class ParameterCountSummary:
    total: int
    million: float
    billion: float


def count_parameters(model: nn.Module):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterCountSummary(
        total=count,
        million=count / 1e6,
        billion=count / 1e9,
    )


def count_trainable_parameters(model: nn.Module):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterCountSummary(
        total=count,
        million=count / 1e6,
        billion=count / 1e9,
    )


def count_inference_parameters(model: MoExLSTMForCausalLM):
    """
    Count the number of parameters used during inference for a MoExLSTMForCausalLM model.
    This includes only the active parameters in the MoE layers (based on top-k experts)
    and all parameters in non-MoE layers.

    Args:
        model: An instance of MoExLSTMForCausalLM.

    Returns:
        int: Total number of parameters used during inference.
    """
    total_params = 0

    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # Check if the parameter belongs to an MoE layer
        if "experts" in name:
            # For MoE layers, only count parameters for the top-k experts
            # Assuming top_k is accessible from the model's config
            top_k = model.config.top_k_experts
            num_experts = model.config.num_experts

            # Calculate the fraction of parameters used (top_k / num_experts)
            fraction_active = top_k / num_experts
            total_params += param.numel() * fraction_active
        else:
            # For non-MoE layers, count all parameters
            total_params += param.numel()

    return ParameterCountSummary(
        total=total_params,
        million=total_params / 1e6,
        billion=total_params / 1e9,
    )
