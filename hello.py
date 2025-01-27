from typing import cast

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)

from moe_xlstm.config import MoExLSTMConfig
from moe_xlstm.modules.model import MoExLSTMForCausalLM
from moe_xlstm.trainer.arguments import MoExLSTMTrainingArguments
from moe_xlstm.utils.weights import count_parameters

if __name__ == "__main__":
    parser = HfArgumentParser(MoExLSTMTrainingArguments)
    args = parser.parse_yaml_file(yaml_file="./trainer_arguments.yaml")[0]
    args = cast(MoExLSTMTrainingArguments, args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = MoExLSTMConfig.from_yaml(args.xlstm_config_path)

    config.xlstm_config.vocab_size = tokenizer.vocab_size

    moe = MoExLSTMForCausalLM(config)
    print(count_parameters(moe))

    dummy_input = "Paris is the capital of"
    inputs = tokenizer(dummy_input, return_tensors="pt")
    max_new_tokens = 10

    temperature = 0.7

    for _ in tqdm(range(max_new_tokens)):
        logits = moe(**inputs).logits[:, -1, :] / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)

    print(tokenizer.decode(inputs["input_ids"].squeeze().tolist()))
