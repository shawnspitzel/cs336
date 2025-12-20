import torch.nn as nn
import torch
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()