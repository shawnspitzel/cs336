# Pretrain.py Implementation Steps

## What You're Building

A training script that loads configs, initializes your model, runs a training loop with logging/checkpointing, and handles validation. Think of it as the "main()" that ties together all your existing components.

---

## Core Concepts to Understand First

### 1. The Training Loop Pattern
- Get batch → Forward pass → Compute loss → Backward pass → Update weights
- This repeats thousands of times
- Periodically: log metrics, validate, save checkpoints

### 2. Configuration Flow
- YAML file contains all hyperparameters
- Script loads YAML into a dictionary
- Command-line args can override any config value
- Everything else uses these config values

### 3. State Management
- Model and optimizer have "state" (weights, momentum buffers, etc.)
- Checkpoints save this state + iteration number
- Resuming loads state and continues from that iteration

---

## Implementation in 5 Major Parts

### Part 1: Configuration System

**What you need:**
- Load YAML file
- Parse command-line overrides (--training.batch_size 128)
- Merge overrides into config
- Convert to easy-to-use format (dict or namespace)

**Key decisions:**
- How to handle nested keys (training.batch_size → config['training']['batch_size'])
- Type conversion (string "128" → integer 128)
- What happens if config file doesn't exist?

**Watch out for:**
- Scientific notation in YAML (6.0e-4)
- Boolean values ("true" string vs True boolean)
- Null/None values
- File paths relative to where script runs

---

### Part 2: Initialization

**What you need:**
- Set random seed (for reproducibility)
- Choose device (CUDA, MPS, or CPU)
- Load data files with np.memmap
- Create model instance
- Create optimizer instance
- Optionally apply torch.compile
- Initialize W&B if enabled

**Order matters:**
1. Seed first (affects model initialization)
2. Model before optimizer (optimizer needs model.parameters())
3. Both before loading checkpoint (checkpoint updates their state)
4. Compile after checkpoint (changes model object)

**Watch out for:**
- Device availability checks (don't assume CUDA exists)
- np.memmap dtype must match how data was saved
- Model arguments must exactly match your Transformer signature
- Optimizer choice (AdamW vs SGD have different parameters)

---

### Part 3: The Training Loop

**What you need:**
- Loop from start_iter to max_iters
- Each iteration:
  - Calculate current learning rate (warmup + cosine schedule)
  - Update optimizer's learning rate
  - Get a batch of data
  - Forward pass through model
  - Compute loss (with proper reshaping)
  - Backward pass (compute gradients)
  - Clip gradients
  - Optimizer step (update weights)
  - Zero gradients for next iteration

**Watch out for:**
- Model outputs shape [batch, seq_len, vocab_size]
- Loss expects [batch*seq_len, vocab_size]
- Need to reshape/flatten before loss computation
- Learning rate updates happen per-iteration, not per-batch
- optimizer.param_groups is where LR lives
- Gradient clipping happens after backward, before step

---

### Part 4: Monitoring and Logging

**What you need:**
- Console logging every N iterations (loss, LR, grad norm)
- W&B logging if enabled (same metrics)
- Validation every M iterations:
  - Switch to eval mode
  - Disable gradients
  - Compute loss on validation data
  - Average over multiple batches
  - Switch back to train mode
- Track iteration number for all logging

**Watch out for:**
- Call .item() on tensors before logging (prevents memory leaks)
- model.eval() and model.train() affect dropout/batchnorm
- torch.no_grad() is essential for validation (saves memory)
- Validation batches are random samples (your data_loading handles this)
- Return to train mode after validation

---

### Part 5: Checkpointing and Recovery

**What you need:**
- Save checkpoints every K iterations
- Save final checkpoint when done
- Resume from checkpoint if specified:
  - Load model state
  - Load optimizer state
  - Get starting iteration number
- Create checkpoint directory if needed

**Optional but useful:**
- Save config file alongside checkpoint
- Keep only last N checkpoints (delete old ones)
- Save separate "best" checkpoint based on val_loss
- Handle interrupted training (Ctrl+C)

**Watch out for:**
- Directory must exist before saving
- Your save_checkpoint/load_checkpoint already work - use them
- Checkpoint filenames should include iteration number
- Resume happens after model/optimizer creation but before training

---

## Critical Implementation Details

### Shape Handling
Your model outputs [batch_size, sequence_length, vocab_size] but cross-entropy expects [batch_size * sequence_length, vocab_size]. Reshape before computing loss.

### Learning Rate Scheduling
Must compute LR for current iteration using your learning_rate_schedule function, then update each param group in the optimizer.

### Data Loading
Your data_loading function returns (inputs, targets) where targets are already shifted. inputs[i] predicts targets[i]. Don't shift again.

### W&B Integration
Call wandb.init() once at start. Log metrics with wandb.log(). If in a sweep, wandb.config contains override values that need merging back.

### Error Recovery
Wrap training loop in try/except for KeyboardInterrupt. Save emergency checkpoint before exiting.

---

## Suggested Build Order

**Phase 1: Minimal Working Version**
1. Config loading
2. Model initialization
3. Single training step (one iteration)
4. Verify loss computes and gradients exist

**Phase 2: Basic Training**
1. Full training loop (10 iterations)
2. Console logging
3. Verify loss changes over iterations

**Phase 3: Persistence**
1. Checkpoint saving
2. Checkpoint loading/resuming
3. Test save→load→resume cycle

**Phase 4: Full Featured**
1. Validation evaluation
2. W&B logging
3. Error handling
4. Final polish

---

## Testing Your Implementation

### Test 1: Config System
Create minimal YAML file. Load it. Override one value from CLI. Print result. Verify override worked.

### Test 2: Initialization
Create tiny model (d_model=64). Print parameter count. Verify it's on expected device.

### Test 3: Data Pipeline
Load data file. Get one batch. Print shapes. Verify inputs and targets are correct size.

### Test 4: Forward Pass
Run one batch through model. Print logits shape. Should be [batch, seq_len, vocab_size].

### Test 5: Loss Computation
Reshape logits and targets. Compute loss. Should get single scalar value.

### Test 6: Backward Pass
Call loss.backward(). Check if gradients exist on model parameters. Print some gradient values.

### Test 7: Full Step
Zero grads → forward → loss → backward → clip → optimizer step. Verify weights actually change.

### Test 8: Multiple Iterations
Run 10 iterations. Print loss each time. Loss should change (usually decrease).

### Test 9: Checkpoint Cycle
Train 5 iterations → save → create new model → load → verify iteration number and state match.

### Test 10: Full Run
Use tiny config (small model, 100 iters). Run to completion. Check checkpoint files exist.

---

## Common Mistakes to Avoid

**Forgetting to reshape for loss**
Model gives [B, L, V], loss needs [B*L, V]. Use .view() or .reshape().

**Not updating learning rate**
LR schedule calculates new value, but you must assign it to optimizer.param_groups[]['lr'].

**Gradients during validation**
Always use torch.no_grad() context or decorator for validation.

**Staying in eval mode**
After validation, call model.train() to resume training mode.

**Logging tensors directly**
Call .item() to extract scalar before logging. Otherwise memory leak.

**Missing directory creation**
Create checkpoint_dir before first save, or you'll get file not found error.

**Wrong data dtype**
np.memmap dtype must match the actual file's dtype (probably uint16).

**Seed timing**
Set seed before creating model, otherwise "reproducible" isn't reproducible.

---

## What Working Looks Like

When successful, you should see:
- Command runs without errors
- Loss prints every N iterations
- Loss generally decreases over time
- Validation loss prints at intervals
- Checkpoint files appear in checkpoint directory
- Can Ctrl+C and resume from checkpoint
- Can override config values from command line
- W&B dashboard shows graphs (if enabled)

---

## Architecture Notes

### Main Function Structure
1. Parse arguments (config + overrides)
2. Set random seed
3. Initialize device
4. Load data
5. Create model
6. Create optimizer
7. Resume checkpoint (if specified)
8. Compile model (if enabled)
9. Initialize W&B (if enabled)
10. Create checkpoint directory
11. Training loop
12. Save final checkpoint

### Training Loop Structure
```
for iteration in range(start_iter, max_iters):
    - Compute LR and update optimizer
    - Get batch
    - Forward
    - Loss
    - Backward
    - Clip
    - Step
    - Periodic: log, validate, checkpoint
```

### Validation Structure
```
- Set eval mode
- Disable gradients
- Loop over eval_iters:
    - Get validation batch
    - Forward
    - Accumulate loss
- Average loss
- Log
- Set train mode
```

---

## Key Design Decisions

### Config Format
Use nested YAML for organization. Allows grouping related settings (model.*, training.*, optimizer.*).

### Override Syntax
Dot notation (--training.batch_size 128) is intuitive and matches YAML structure.

### Checkpoint Naming
Include iteration number in filename (checkpoint_5000.pt) for easy identification.

### Logging Frequency
log_interval < eval_interval < checkpoint_interval. Common: 100, 500, 1000.

### Device Handling
Try CUDA first, fall back to MPS, then CPU. Print which device is used.

### W&B Integration
Optional but valuable. If wandb import fails, training should still work.

---

## Integration with Existing Components

You already have:
- ✅ Transformer (model)
- ✅ AdamW and SGD (optimizers)
- ✅ cross_entropy_loss (loss function)
- ✅ learning_rate_schedule (LR scheduler)
- ✅ gradient_clipping (gradient processing)
- ✅ data_loading (batch loading)
- ✅ save_checkpoint / load_checkpoint (persistence)

Your job is to orchestrate these into a training pipeline. Each component is a puzzle piece - you're assembling the puzzle.

---

## Final Tips

**Start small**: Get minimal version working before adding features.

**Test constantly**: After every addition, run and verify it works.

**Print liberally**: While debugging, print shapes, values, and status messages.

**Use tiny configs**: Debug with small models and few iterations. Scale up when working.

**Trust your components**: Your model, optimizer, and loss functions are already tested. If something seems wrong, check the integration, not the components.

**Read error messages**: PyTorch errors usually point to the exact problem (shape mismatch, device mismatch, etc.).

**Monitor val_loss**: If validation loss doesn't eventually decrease, something's wrong with the training loop.

**Save early, save often**: Checkpoint frequently while debugging. You don't want to lose 30 minutes of training to a bug.

**One feature at a time**: Don't try to implement logging + validation + checkpointing simultaneously. Add one, test it, move to next.

**Refer to design doc when needed**: The detailed design document has code patterns for each piece when you need implementation details.

Good luck! You've built all the hard parts already. This is just the glue.