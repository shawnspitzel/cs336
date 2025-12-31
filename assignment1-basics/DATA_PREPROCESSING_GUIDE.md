# Data Preprocessing Guide

## Overview: The Two-Stage Process

Training a language model requires two stages:

### **Stage 1: Preprocessing (ONE TIME)** ← You need to do this first!
Convert raw text → tokenized binary files

### **Stage 2: Training (REPEATED)**
Load tokenized data → Train model

---

## Stage 1: Data Preprocessing

### What Happens:
1. **Train BPE Tokenizer** on your training text
2. **Tokenize** both train and validation text files
3. **Save** as binary `.bin` files for fast loading during training

### How to Run:

```bash
# Example with TinyStories dataset
python cs336_basics/training/preprocess_data.py \
    --train_file path/to/train.txt \
    --val_file path/to/val.txt \
    --vocab_size 50257 \
    --output_dir cs336_basics/data
```

### What Gets Created:
```
cs336_basics/data/
├── tinystories_train_tokens.bin   # Tokenized training data
├── tinystories_val_tokens.bin     # Tokenized validation data
└── tokenizer.json                  # Trained BPE tokenizer
```

---

## Stage 2: Training

### What Happens:
1. **Load** pre-tokenized `.bin` files
2. **Sample** random batches during training
3. **Train** your model

### How to Run:

```bash
# With default parameters
python cs336_basics/training/pretrain.py

# With custom config
python cs336_basics/training/pretrain.py --use_params

# With command-line args
python cs336_basics/training/pretrain.py \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --max_iters 10000
```

---

## Production Workflow

```
┌─────────────────────────────────────────────────┐
│  ONE-TIME PREPROCESSING                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  Raw Text Files                                 │
│    ├── train.txt                                │
│    └── val.txt                                  │
│         ↓                                       │
│  [preprocess_data.py]                          │
│    - Train BPE tokenizer                       │
│    - Tokenize text → token IDs                 │
│    - Save as binary arrays                     │
│         ↓                                       │
│  Tokenized Binary Files                        │
│    ├── tinystories_train_tokens.bin           │
│    ├── tinystories_val_tokens.bin             │
│    └── tokenizer.json                          │
│                                                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  TRAINING (CAN RUN MULTIPLE TIMES)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  [pretrain.py]                                 │
│    - Load tokenized .bin files                 │
│    - Create model                              │
│    - Training loop:                            │
│        • Sample batch from .bin file           │
│        • Forward pass                          │
│        • Backward pass                         │
│        • Update weights                        │
│    - Save checkpoints                          │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Why Separate Preprocessing?

### ✅ **Advantages:**
1. **Speed**: Tokenization is slow. Do it once, not every training run
2. **Consistency**: Same tokenization for all experiments
3. **Efficiency**: Binary files load 100x faster than parsing text
4. **Memory**: Can memory-map huge files without loading all into RAM
5. **Reproducibility**: Tokenized data is deterministic

### ❌ **What NOT to do:**
- ❌ Don't tokenize inside training loop (too slow!)
- ❌ Don't tokenize on-the-fly for each batch (wasteful!)
- ❌ Don't re-tokenize for every experiment (inconsistent!)

---

## Quick Start Checklist

- [ ] Get raw text data (train.txt and val.txt)
- [ ] Run `preprocess_data.py` to create .bin files
- [ ] Verify .bin files exist in `cs336_basics/data/`
- [ ] Run `pretrain.py` to start training
- [ ] Monitor training with W&B or console logs

---

## File Formats

### Binary Token Files (.bin)
- **Format**: Raw numpy array saved with `.tofile()`
- **Dtype**: `uint16` (supports vocab up to 65,536 tokens)
- **Loading**: `np.memmap(path, dtype='uint16', mode='r')`
- **Size**: ~2 bytes per token

### Why uint16?
- GPT-2 vocab: 50,257 tokens (fits in uint16)
- Llama vocab: 32,000 tokens (fits in uint16)
- If vocab > 65,536, use `uint32` instead

---

## Example: TinyStories Dataset

If you're using TinyStories:

```bash
# 1. Download TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 2. Preprocess
python cs336_basics/training/preprocess_data.py \
    --train_file TinyStoriesV2-GPT4-train.txt \
    --val_file TinyStoriesV2-GPT4-valid.txt \
    --output_dir cs336_basics/data

# 3. Train
python cs336_basics/training/pretrain.py --use_params
```

---

## Advanced: Re-using Tokenizer

If you already have a trained tokenizer:

```bash
python cs336_basics/training/preprocess_data.py \
    --train_file new_train.txt \
    --val_file new_val.txt \
    --tokenizer_path cs336_basics/data/tokenizer.json \
    --output_dir cs336_basics/data
```

This skips tokenizer training and uses the existing one.
