from transformers import AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from moe_xlstm.modules.model import MoExLSTMForCausalLM
from moe_xlstm.trainer.arguments import MoExLSTMTrainingArguments


class MoExLSTMTrainer(Trainer):
    def __init__(
        self,
        model: MoExLSTMForCausalLM,
        args: MoExLSTMTrainingArguments,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(model=model, args=args, processing_class=tokenizer, **kwargs)

        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        output: CausalLMOutputWithPast = model(**inputs)

        return (output.loss, output) if return_outputs else output.loss
