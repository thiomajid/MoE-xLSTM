import torch

from moe_xlstm.config import MoExLSTMConfig
from moe_xlstm.modules.model import MoExLSTMForCausalLM
from moe_xlstm.utils.weights import count_inference_parameters, count_parameters

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.randn(2).cuda())

    config = MoExLSTMConfig.from_yaml("./model_config.yaml")

    moe = MoExLSTMForCausalLM(config)
    print(count_parameters(moe))

    dummy_input = torch.randint(0, 100, (1, 10))
    output = moe(dummy_input)

    print(output.logits.shape)

    print("Done.")
    print(count_inference_parameters(moe))
