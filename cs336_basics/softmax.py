import torch


def softmax(input: torch.Tensor, dim: int):
    # batch, token_num, hidden_size
    # max
    (max_val, _) = input.max(dim=dim, keepdim=True)
    # 解决上溢问题
    input = input - max_val
    exp_input = torch.exp(input)
    exp_sum = torch.sum(exp_input, dim=dim, keepdim=True)
    result = exp_input / exp_sum
    return result
