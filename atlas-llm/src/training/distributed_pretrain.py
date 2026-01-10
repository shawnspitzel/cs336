"""
CS336 Assignment: Distributed Data Parallel Training
=====================================================

In this assignment, you'll implement a naïve version of Distributed Data Parallel (DDP) training.

LEARNING OBJECTIVES:
- Understand how data parallelism splits batches across multiple devices
- Learn to use PyTorch's distributed communication primitives (broadcast, all_reduce)
- Implement gradient synchronization across multiple processes/GPUs

BACKGROUND:
Data parallelism enables training with larger effective batch sizes by splitting a batch
across multiple devices. For example, 4 GPUs × batch_size 32 = effective batch_size 128.

THE ALGORITHM (executed each training iteration):
1. Data Sharding: Split batch of n examples across d devices (each gets n/d examples)
2. Forward + Backward: Each device independently computes gradients on its subset
3. Gradient Synchronization: Use all_reduce to average gradients across all devices
4. Optimizer Step: Each device updates parameters (stays in sync due to same gradients)

YOUR TASK:
Implement the DDPIndividualParameters class and related functions below.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class DDPIndividualParameters(nn.Module):
    """
    A naïve implementation of Distributed Data Parallel training.

    This wrapper:
    1. Broadcasts parameters from rank 0 to all other ranks (ensures identical initialization)
    2. Registers hooks to asynchronously all-reduce gradients as they become available
    3. Provides a method to wait for all gradient synchronization to complete

    The approach is "naïve" because it all-reduces each parameter's gradient individually,
    rather than using optimizations like bucketing (which you'll implement later).
    """

    def __init__(self, module: nn.Module):
        """
        Initialize the DDP wrapper.

        Args:
            module: The PyTorch model to wrap with DDP functionality

        TODO: Implement the following steps:
        1. Call super().__init__()
        2. Store the module as self.module
        3. Initialize self.grad_handles as an empty list (will store async operation handles)
        4. Broadcast all parameters from rank 0 to other ranks
        5. Register gradient hooks on parameters that require gradients
        """
        super().__init__()

        # STEP 1: Store the wrapped module
        # TODO: self.module = ???
        self.module = module

        # STEP 2: Initialize list to track asynchronous all-reduce operations
        # This will store handles returned by async all_reduce operations
        # TODO: self.grad_handles = ???
        self.grad_handles = []
        # STEP 3: Broadcast parameters from rank 0 to all other ranks
        # This ensures all processes start with identical model weights
        #
        # HINT: Use dist.broadcast(tensor, src=0) for each parameter
        # HINT: Broadcast both parameters that require_grad and those that don't
        # HINT: You need to broadcast the parameter's .data attribute
        #
        # TODO: Implement parameter broadcasting
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # STEP 4: Register backward hooks for gradient synchronization
        # Hook will automatically all-reduce each gradient when it's computed
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._create_grad_hook(param))

    def _create_grad_hook(self, param: nn.Parameter):
        """
        Creates a hook that all-reduces the gradient when it's computed.

        The hook is called during backward pass and starts async all-reduce
        to overlap communication with remaining backward computation.
        """
        def hook(grad):
            # Start async all-reduce: sums gradients across all ranks
            # async_op=True returns immediately with a handle
            # all_reduce operates in-place on the tensor
            handle = dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.grad_handles.append(handle)
            return grad

        return hook

    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations, then average gradients.

        Must be called after backward() but before optimizer.step().
        """
        # Wait for all async all-reduce operations to complete
        for handle in self.grad_handles:
            handle.wait()

        # All-reduce sums gradients, so divide by world_size to get average
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data.div_(world_size)

        # Clear handles for next iteration
        self.grad_handles = []

    def forward(self, *args, **kwargs):
        """Forward pass delegates to wrapped module."""
        return self.module(*args, **kwargs)


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Factory function to create a DDP-wrapped model.

    Args:
        module: torch.nn.Module to wrap with DDP

    Returns:
        DDPIndividualParameters instance wrapping the module
    """
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    """
    Called after backward() but before optimizer.step().

    Waits for all gradient synchronization to complete.

    Args:
        ddp_model: The DDP-wrapped model
        optimizer: The optimizer (unused)
    """
    ddp_model.finish_gradient_synchronization()


# =============================================================================
# TESTING YOUR IMPLEMENTATION
# =============================================================================
"""
To test your implementation, run:

    pytest tests/test_ddp_individual_parameters.py -v

The test will:
1. Spawn 2 processes (simulating 2 GPUs/nodes)
2. Create a toy model on each process
3. Broadcast rank 0's weights to rank 1
4. Train for 5 iterations where:
   - Non-parallel model sees all 20 examples
   - Each DDP rank sees 10 disjoint examples
5. Verify DDP model matches non-parallel model

DEBUGGING TIPS:
- If parameters don't match after training, check your gradient averaging logic
- If you get "default group not initialized", ensure dist.init_process_group was called
- If gradients are None, check that you're only hooking parameters with requires_grad=True
- Print statements in hooks can help debug, but remember they run during backward pass

COMMON PITFALLS:
1. Forgetting to divide by world_size (all_reduce sums, we need average)
2. Not waiting for async operations before optimizer step
3. Hooking parameters that don't require gradients
4. Not clearing grad_handles between iterations
5. Forgetting to return the gradient in the hook function

UNDERSTANDING THE TEST:
- ToyModel has some parameters with requires_grad=False (fc2.bias, no_grad_fixed_param)
- ToyModelWithTiedWeights shares weights between fc2 and fc4
- Both should work with your implementation without special handling
"""
