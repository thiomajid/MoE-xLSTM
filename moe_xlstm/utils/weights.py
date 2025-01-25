from dataclasses import dataclass

from torch import nn


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
