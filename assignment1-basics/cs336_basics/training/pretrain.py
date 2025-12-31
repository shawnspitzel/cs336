import yaml
import osp
import wandb
import numpy as np
import torch
from .loader import data_loading, load_checkpoint, save_checkpoint
from cs336_basics.model.optimizer import AdamW, SGDOptimizer
from cs336_basics.model.transformer import Transformer
from cs336_basics.model.loss import cross_entropy_loss, gradient_clipping, learning_rate_schedule
from cs336_basics.utils.args import get_args_pretrain


def pretrain(model, train_data, val_data, optimizer, params, iteration):
    """
    Single training iteration with evaluation
    
    Inputs:
        model: Transformer model
        train_data: numpy array of training tokens
        val_data: numpy array of validation tokens
        optimizer: AdamW or SGDOptimizer instance
        params: dict of hyperparameters

    Returns:
        train_loss: training loss for this iteration
        val_loss: validation loss (if eval time)
    """

    model.train()
    inputs, targets = data_loading(train_data, params["batch_size"], params["context_length"], params["device"])
    optimizer.zero_grad()

    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    loss.backward()

    gradient_clipping(model.parameters(), max_norm=params['gradient_clip_norm'])

    lr = learning_rate_schedule(
        curr_iter=iteration,
        max_lr=params["learning_rate"],
        min_lr=params["min_lr"], 
        warm_iters=params["warmup_iters"],
        cos_iters=params["cosine_iters"]
        )
    optimizer.lr = lr
    optimizer.step()

    train_loss = loss.item()
    val_loss = None
    
    if iteration % params["eval_interval"] == 0:
        model.eval()
        val_losses = []

        with torch.no_grad():
            for _ in range(params["eval_iters"]):
                val_inputs, val_targets = data_loading(
                    val_data,
                    params["batch_size"],
                    params["context_length"],
                    params["device"]
                )
                val_logits = model(val_inputs)
                val_loss_batch = cross_entropy_loss(val_logits, val_targets)
                val_losses.append(val_loss_batch.item())
        val_loss = sum(val_losses) / len(val_losses)

    return train_loss, val_loss


def run(params):
    """
    Main training loop

    Inputs:
        params: dict of all hyperparameters from args/config
    """
    
    torch.manual_seed(params["seed"])

    requested_device = params["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    train_data = np.memmap(params["train_data"], dtype=params["data_dtype"], mode='r')
    val_data = np.memmap(params["val_data"], dtype=params["data_dtype"], mode='r')
    
    model = Transformer(
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            d_ff=params['d_ff'],
            num_layers=params['num_layers'],
            vocab_size=params['vocab_size'],
            context_length=params['context_length'],
            theta=params['theta']
        )

    model = model.to(device)
    if params["compile"]:
        model = torch.compile(model)
        
    if params["optimizer"] == "sgd":
        optimizer = SGDOptimizer(params=model.parameters(), lr=params["learning_rate"])
    else:
        optimizer = AdamW(
            params=model.parameters(), 
            betas=(params["beta1"], params["beta2"]),
            weight_decay=params["weight_decay"],
            lr=params["learning_rate"]
            )

    start_iter = 0
    if params["resume_from"]:
        start_iter = load_checkpoint(params["resume_from"], model, optimizer=optimizer)

    for iter in range(start_iter, params["max_iters"]):
        train_loss, val_loss = pretrain(
            model=model, 
            train_data=train_data, 
            val_data=val_data,
            optimizer=optimizer,
            params=params,
            iteration=iter
            )
        if iter % params["log_interval"] == 0:
            print(f"Iter {iter}: train_loss={train_loss:.4f}, lr={optimizer.lr:.6f}")
            wandb.log({"train/loss": train_loss, "lr": optimizer.lr, "iteration": iter})

        if iter % params["eval_interval"] == 0 and val_loss is not None:
            print(f"Iter {iter}: val_loss={val_loss:.4f}")
            wandb.log({"val/loss": val_loss, "iteration": iter})

        if iter % params["checkpoint_interval"] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iter,
                out=params["checkpoint_dir"]
            )
            
    save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=params["max_iters"],
            out=params["checkpoint_dir"]
        )
    wandb.finish()

if __name__ == "__main__":
    params = get_args_pretrain()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    if params['use_params']:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'pretrain.yaml'),) as f:
            default_params = yaml.safe_load(f)
            params.update(default_params)

    wandb.init(
        project="AtlasLM-Pretrain",
        name=f"Pretrain_layers_{params['num_layers']}_bs{params['batch_size']}_lr{params['learning_rate']}",
        mode="online",
        config=params,
    )

    run(params)