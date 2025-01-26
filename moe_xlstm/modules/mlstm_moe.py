import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from xlstm import mLSTMBlock

from moe_xlstm.config import MoExLSTMConfig


class mLSTMMoELayer(nn.Module):
    def __init__(self, config: MoExLSTMConfig) -> None:
        super().__init__()

        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts

        self.gate = nn.Linear(
            config.xlstm_config.embedding_dim, config.num_experts, bias=config.gate_bias
        )
        self.experts = nn.ModuleList(
            [
                mLSTMBlock(config.xlstm_config.mlstm_block)
                for _ in range(config.num_experts)
            ]
        )

    # code adapted from transformers.models.mixtral.modeling_mixtral.py
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        flat_hidden_states = hidden_states.view(-1, hidden_dim)  # (B*S, D)

        router_logits = self.gate(flat_hidden_states)  # (B*S, num_experts)
        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # normalize routing weights to sum up to 1
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(flat_hidden_states)

        # for each expert find tokens that are routed to it
        expert_mask = rearrange(
            F.one_hot(selected_experts, num_classes=self.num_experts),
            "batch top_k experts -> experts top_k batch",
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # find tokens that are routed to the expert
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:  # Skip unused experts
                continue

            # Reshape to 3D for mLSTMBlock
            current_state = flat_hidden_states[top_x]  # (num_selected, D)
            current_state = current_state.unsqueeze(1)  # (num_selected, 1, D)

            expert_output = expert_layer(current_state)  # (num_selected, 1, D)
            expert_output = expert_output.squeeze(1)  # (num_selected, D)

            weighted_output = expert_output * routing_weights[top_x, idx, None]

            # Accumulate results
            final_hidden_states.index_add_(0, top_x, weighted_output)

        # reshape back to 3D
        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states, router_logits
