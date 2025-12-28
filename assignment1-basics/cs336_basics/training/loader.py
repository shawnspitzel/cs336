import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    N = len(x)
    max_start = N - context_length
    if max_start <= 0:
        raise ValueError("Dataset too small for given context_length")
    
    starts = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
        device=device,
    )
    for i, s in enumerate(starts):
        inputs[i] = torch.from_numpy(x[s : s + context_length])
        targets[i] = torch.from_numpy(x[s + 1 : s + context_length + 1])
    return inputs, targets
    