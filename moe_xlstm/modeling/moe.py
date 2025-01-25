import torch
import torch.nn.functional as F
from torch import nn
from xlstm import mLSTMBlock

from moe_xlstm.config import MoExLSTMConfig


class mLSTMMoEBlock(nn.Module):
    def __init__(self, config: MoExLSTMConfig) -> None:
        super().__init__()

        self.config = config
        self.experts = nn.ModuleList(
            [
                mLSTMBlock(config.xlstm_config.mlstm_block)
                for _ in range(config.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape
        top_k_scores, top_k_indices = torch.topk(
            scores, self.config.top_k_experts, dim=-1
        )  # [B, L, top_k]

        # Get weights via softmax over top-k scores
        weights = F.softmax(top_k_scores, dim=-1)  # [B, L, top_k]

        # Initialize output and process selected experts
        output = torch.zeros(B, L, D, device=hidden_states.device)
        for i in range(self.config.top_k_experts):
            expert_idx = top_k_indices[:, :, i]  # [B, L]
            expert_mask = F.one_hot(
                expert_idx, num_classes=self.config.num_experts
            )  # [B, L, num_experts]

            # Sum over all batches/sequences where expert is activated
            for e in range(self.config.num_experts):
                mask = expert_mask[:, :, e].bool()  # [B, L]
                if mask.any():
                    expert_input = hidden_states[mask].view(
                        -1, D
                    )  # [num_selected_tokens, D]
                    expert_out = self.experts[e](expert_input)
                    output[mask] += expert_out * weights[:, :, i][mask].unsqueeze(-1)

        return output


class MoExLSTMLayer(nn.Module):
    def __init__(self, config: MoExLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.xlstm_config.embedding_dim, config.num_experts)
        self.experts = mLSTMMoEBlock(config)

    def forward(self, hidden_states: torch.Tensor):
        scores = self.gate(hidden_states)  # [B, L, num_experts]
        return self.experts(hidden_states, scores)


class MoExLSTM(nn.Module):
    def __init__(self, config: MoExLSTMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.xlstm_config.vocab_size,
            embedding_dim=config.xlstm_config.embedding_dim,
        )

        self.embedding_dropout = (
            nn.Dropout(config.dropout)
            if config.xlstm_config.add_embedding_dropout
            else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [MoExLSTMLayer(config) for _ in range(config.xlstm_config.num_blocks)]
        )

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class MoExLSTMForCausalLM(nn.Module):
    def __init__(self, config: MoExLSTMConfig):
        super().__init__()

        self.config = config

        self.model = MoExLSTM(config)
        self.lm_head = nn.Linear(
            in_features=config.xlstm_config.embedding_dim,
            out_features=config.xlstm_config.vocab_size,
            bias=False,
        )

        if config.xlstm_config.tie_weights:
            self.lm_head.weight = self.model.token_embedding.weight

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
