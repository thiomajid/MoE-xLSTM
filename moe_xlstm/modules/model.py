from pathlib import Path
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from moe_xlstm.config import MoExLSTMConfig
from moe_xlstm.modules.mlstm_moe import mLSTMMoELayer


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
            [mLSTMMoELayer(config) for _ in range(config.xlstm_config.num_blocks)]
        )

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)

        return hidden_states


class MoExLSTMForCausalLM(PreTrainedModel):
    config_class = MoExLSTMConfig

    def __init__(self, config: MoExLSTMConfig):
        super().__init__(config)

        self.config = config

        self.model = MoExLSTM(config)
        self.lm_head = nn.Linear(
            in_features=config.xlstm_config.embedding_dim,
            out_features=config.xlstm_config.vocab_size,
            bias=False,
        )

        if config.xlstm_config.tie_weights:
            self.lm_head.weight = self.model.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids)
        logits: torch.Tensor = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
            shift_logits = rearrange(
                logits[..., :-1, :].contiguous(), "b s v -> (b s) v"
            )

            # shape: [batch, seq] -> [batch * (seq-1)]
            shift_labels = rearrange(labels[..., 1:].contiguous(), "b s -> (b s)")

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                input=shift_logits,
                target=shift_labels,
            )

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            attentions=None,
        )

    @staticmethod
    def from_safetensors(
        hf_repo: str,
        filename: Path | str,
        device: str = "cuda",
    ) -> "MoExLSTMForCausalLM":
        """
        Creates an instance of the model by loading its safetensors checkpoint downloaded from the Hugging Face Hub
        and using its configuration to initialize the model.


        Parameters
        ----------
        hf_repo : str
            Hugging Face repository where the model weights are stored as well as the configuration to be used.
        filename : Path | str
            Path to the safetensors checkpoint file.
        device : str, optional
            The device on which the model will be loaded, by default "cpu"

        Returns
        -------
        MoExLSTMForCausalLM

        Raises
        ------
        FileNotFoundError
            If the file does not exist on the disk.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist on the disk.")

        config = MoExLSTMConfig.from_pretrained(hf_repo)
        model = MoExLSTMForCausalLM(config=config)
        safetensors.torch.load_model(model=model, filename=filename, device=device)
        model = model.to(device)

        return model
