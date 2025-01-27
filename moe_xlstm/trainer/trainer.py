from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, Trainer

from moe_xlstm.modules.model import MoExLSTMForCausalLM
from moe_xlstm.modules.output import MoECausalLMOutput
from moe_xlstm.trainer.arguments import MoExLSTMTrainingArguments


class MoExLSTMTrainer(Trainer):
    def __init__(
        self,
        model: MoExLSTMForCausalLM,
        args: MoExLSTMTrainingArguments,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(model=model, args=args, tokenizer=tokenizer, **kwargs)

        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=args.logging_dir)

        # Store which layers to monitor - default to all layers if not specified
        self.monitored_layers = (
            list(range(len(model.moe.layers)))
            if args.monitored_layers == "all"
            else args.monitored_layers
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss and collect MoE metrics during training."""
        # Forward pass with layer outputs
        outputs: MoECausalLMOutput = model(**inputs, return_layers_outputs=True)

        loss = outputs.loss if outputs.loss is not None else 0.0
        total_router_z_loss = torch.tensor(0.0, device=loss.device)
        total_router_aux_loss = torch.tensor(0.0, device=loss.device)

        # Process each layer's outputs
        if outputs.layers_outputs is not None:
            for layer_idx, layer_output in enumerate(outputs.layers_outputs):
                if layer_idx in self.monitored_layers:
                    # Compute metrics for this layer
                    metrics = self._compute_router_metrics(layer_output.router_logits)

                    # Accumulate losses
                    total_router_z_loss += metrics["router_z_loss"]
                    total_router_aux_loss += metrics["router_aux_loss"]

                    # Log metrics if it's a logging step
                    if self.state.global_step % self.args.logging_steps == 0:
                        self._log_moe_metrics(
                            layer_idx=layer_idx,
                            metrics=metrics,
                            prefix="train",
                        )

        # Combine all losses
        router_loss_coef = getattr(self.args, "router_loss_coef", 0.001)
        total_loss = loss + router_loss_coef * (
            total_router_z_loss + total_router_aux_loss
        )

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_router_metrics(
        self, router_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute router-related metrics for logging."""
        # Calculate router z-loss for numerical stability
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        # Calculate expert usage distribution
        routing_weights = torch.softmax(router_logits, dim=-1)
        expert_usage = torch.mean(routing_weights, dim=0)

        # Calculate auxiliary loss for load balancing
        num_experts = router_logits.size(-1)
        ideal_usage = torch.ones_like(expert_usage) / num_experts
        aux_loss = torch.mean((expert_usage - ideal_usage) ** 2) * num_experts

        return {
            "router_z_loss": z_loss,
            "router_aux_loss": aux_loss,
            "expert_usage": expert_usage,
        }

    def _log_moe_metrics(
        self,
        layer_idx: int,
        metrics: Dict[str, torch.Tensor],
        prefix: str = "train",
    ) -> None:
        """Log MoE-specific metrics to TensorBoard for a single layer."""

        # Log expert usage statistics
        expert_usage = metrics["expert_usage"]

        # Add histogram of expert usage
        self.tb_writer.add_histogram(
            f"{prefix}/layer_{layer_idx}/expert_usage_dist",
            expert_usage.detach(),
            self.state.global_step,
        )

        # Log all metrics in a single dictionary
        self.log(
            {
                f"{prefix}/layer_{layer_idx}/expert_usage_std": expert_usage.std().item(),
                f"{prefix}/layer_{layer_idx}/expert_usage_max": expert_usage.max().item(),
                f"{prefix}/layer_{layer_idx}/expert_usage_min": expert_usage.min().item(),
                f"{prefix}/layer_{layer_idx}/router_z_loss": metrics[
                    "router_z_loss"
                ].item(),
                f"{prefix}/layer_{layer_idx}/router_aux_loss": metrics[
                    "router_aux_loss"
                ].item(),
            },
        )

    def close(self):
        """Cleanup when training is done."""
        self.tb_writer.close()
