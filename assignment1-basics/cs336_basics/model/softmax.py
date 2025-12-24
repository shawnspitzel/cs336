import torch.nn as nn
import torch
from jaxtyping import Float, Int
from torch import Tensor
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp



