from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class MoELayerOutput:
    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class MoECausalLMOutput(CausalLMOutputWithPast):
    layers_outputs: Optional[list[MoELayerOutput]] = None
    output_per_layer: dict[str, dict] = None
