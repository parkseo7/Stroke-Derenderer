"""Script to train the Unet attention model for binarization and thinning
the text image.
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from tqdm.auto import tqdm

from derenderer.loader import BinarizationDataset
from derenderer.model.unet_attention import UnetAttention, Loss
from derenderer.common import load_yaml, load_metrics, save_metrics
import logging


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of
    gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def setup_directories(vargs):
    """Given config arguments, set-up all directories for preparing
    model datasets and training.
    """

    vargs_dir = vargs["directories"]
    vargs_model = vargs["model"]

    model_name = vargs_model["name"]
    dir_base = Path(vargs_dir["model"]) / model_name
    dir_base.mkdir(parents=True, exist_ok=True)
    
    dir_model = dir_base / "models"
    dir_log = dir_base / "log" # For model logs.

    dir_model.mkdir(parents=True, exist_ok=True)
    dir_log.mkdir(parents=True, exist_ok=True)

    # Set-up model version folder (checkpoints)
    version = vargs_model["version"]
    version_name = f"{model_name}_{version:02d}"
    dir_version = dir_model / version_name
    dir_version.mkdir(parents=True, exist_ok=True)
    log_filepath = dir_log / f"{version_name}.pkl"
    dirs = {
        "model": dir_version,
        "log": log_filepath
    }
    return dirs


def lr_scheduler(**lr_params):
    """Creates a learning rate scheduler, which warms up to a certain
    epoch, followed by a decay to the last epoch.
    """
    
    epochs = lr_params["epochs"]
    peak_lr = lr_params["peak_lr"]
    peak_epoch = lr_params["peak_epoch"]
    base_lr = lr_params["base_lr"]
    decay_lr = lr_params["decay_lr"]

    warmup_slope = (peak_lr - base_lr) / (peak_epoch - 1)
    gamma = np.log(decay_lr / peak_lr) / (epochs - peak_epoch)

    def lr_fn(epoch):
        if epoch <= peak_epoch:
            lr = max(warmup_slope * (epoch - 1) + base_lr, base_lr)
        else:
            lr = peak_lr * np.exp(gamma * (epoch - peak_epoch))
        return lr
    
    return lr_fn


def prepare_datasets(vargs, dirs):
    """Prepares the data by loading in the training and validation batches.
    Returns the training and validation datasets.
    """

    vargs_dir = vargs["directories"]
    dir_data = vargs_dir["data"]
    vargs_data = vargs["dataset"]
    train_batches = vargs_data.get("train_batches")
    val_batches = vargs_data.get("val_batches")

    vargs_model = vargs["model"]
    height = vargs_model["height"]
    width = vargs_model["width"]
    init_features = vargs_model["init_features"]
    label_name = vargs_data.get("label_folder", "skeleton")

    train_batches = vargs_data["train_batches"]
    val_batches = vargs_data["val_batches"]
    bl_train = BinarizationDataset(height=height, width=width,
                                   label_name=label_name)
    bl_train.get_data_paths(dir_data, batches=train_batches)
    bl_val = BinarizationDataset(height=height, width=width,
                                 label_name=label_name)
    bl_val.get_data_paths(dir_data, batches=val_batches)

    model_name = vargs_model["name"]
    version = vargs_model["version"]
    # Set-up log metrics file and save:
    log_dict = {
        "name": model_name,
        "version": version,
        "label_name": label_name,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "parameters": {
            "height": height,
            "width": width,
            "init_features": init_features
        },
        "history": [] # For recording training metrics
    }
    save_metrics(log_dict, dirs["log"])

    vargs_train = vargs_model["train"]
    batch_size = vargs_train.get("batch_size", 16) # Low batch size
    num_workers = vargs_train.get("num_workers", 8)

    train_dl = DataLoader(bl_train,
                          batch_size=batch_size, 
                          shuffle=True, # Shuffle is on. Works with iter
                          collate_fn=bl_train.pad_collate_fn,
                          num_workers=num_workers,
                          pin_memory=False) # Keep this False, strains the GPU
    val_dl = DataLoader(bl_val,
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=bl_val.pad_collate_fn,
                        num_workers=num_workers,
                        pin_memory=False)
    
    return train_dl, val_dl, log_dict


def train_epoch(model, vargs_train, train_iter, criterion, opt, scheduler, 
                device, epoch=1):
    """Given a model, training arguments, and a loss function, optimizer, 
    and scheduler, does a single training epoch-run of the model. Returns
    metric values to print.
    """

    # Configure loss and optimizer:
    batch_size = vargs_train.get("batch_size", 16)
    grad_clip = vargs_train.get("grad_clip", 5.0)
    loss_weights = vargs_train.get("loss_weights", [0.5, 0.3, 0.2, 0.1])

    # Training mode:
    model.train()
    tloss = 0
    desc = f"Training epoch {epoch}"
    for (inputs, labels, lengths) in tqdm(train_iter, desc=desc):

        # Move to devices:
        inputs = inputs.to(device)
        # Zero-grad:
        opt.zero_grad()
        # Predictions: Predictions and labels have 4 resolutions.
        preds = model(inputs)
        loss = 0
        # We don't use lengths here. Rather go for batch backpropagation
        for i in range(4):
            img_pred = preds[i].to(device)
            img_true = labels[i].type(torch.float32).to(device)
            wgt = loss_weights[i]
            dice_loss, focal_loss = criterion(img_pred, img_true)
            add_loss = dice_loss + focal_loss
            loss += wgt * add_loss

        # Backward propagation:
        loss.backward()
        # Gradient clipping:
        clip_gradient(opt, grad_clip)
        # Optimizer step
        opt.step()
        tloss += loss.item()

    # Scheduler step (adjust learning rate)
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]
    tloss = tloss / (len(train_iter) * batch_size)

    return {
        "lr": lr_now,
        "tloss": tloss
    }


def val_epoch(model, vargs_train, val_iter, criterion, device, epoch=1):
    """Given a model, training arguments, does a single validation epoch-run of 
    the model. Returns metric values to print.
    """
    
    # Training parameters:
    batch_size = vargs_train.get("batch_size", 16)
    loss_weights = vargs_train.get("loss_weights", [0.5, 0.3, 0.2, 0.1])

    # Evaluation mode:
    model = model.eval()

    vloss = 0
    desc = f"Validation epoch {epoch}"
    with torch.no_grad():
        for (inputs, labels, lengths) in tqdm(val_iter, desc=desc):
            # Move to devices:
            inputs = inputs.to(device)
            # Prediction:
            preds = model(inputs)

            # Preds and labels have 4 resolutions:
            loss = 0
            for i in range(4):
                img_pred = preds[i].to(device)
                img_true = labels[i].to(device)
                wgt = loss_weights[i]
                dice_loss, focal_loss = criterion(img_pred, img_true)
                add_loss = dice_loss + focal_loss
                loss += wgt * add_loss

            vloss += loss.item()
    
    vloss = vloss / (len(val_iter) * batch_size)
    return {
        "vloss": vloss
    }


def main(vargs):
    """Main script to train a model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    logging.debug("Setting up directories")
    dirs = setup_directories(vargs)
    
    # Get datsets and metrics log
    logging.debug("Preparing datasets")
    train_dl, val_dl, metrics_log = prepare_datasets(vargs, dirs)

    # Create model:
    vargs_model = vargs["model"]
    model_name = vargs_model["name"]
    version = vargs_model["version"]
    init_features = vargs_model.get("init_features")
    in_channels = vargs_model.get("in_channels", 3)
    model = UnetAttention(in_channels=in_channels, init_features=init_features)
    model.to(device)

    # Optional checkpoint:
    checkpoint = vargs_model.get("checkpoint")
    if checkpoint is not None:
        model_weights = torch.load(checkpoint, weights_only=True,
                                   map_location=device)
        model.load_state_dict(model_weights)
    
    # Configure loss function, optimizer, scheduler:
    vargs_train = vargs_model["train"]
    epochs = vargs_train.get("epochs")
    save_per_epoch = vargs_train.get("save_per_epoch", 2)
    lr_lambda = lr_scheduler(**vargs_train)
    criterion = Loss()
    opt = torch.optim.Adam(params=model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    for epoch in range(1, epochs+1):
        # Reset iterations:
        train_iter = iter(train_dl)
        val_iter = iter(val_dl)
        
        train_log = train_epoch(model, vargs_train, train_iter, 
                                criterion, opt, scheduler, device,
                                epoch=epoch)
        val_log = val_epoch(model, vargs_train, val_iter, 
                            criterion, device, epoch=epoch)
        
        # Print statement:
        lr_now = train_log["lr"]
        tloss = train_log["tloss"]
        vloss = val_log["vloss"]

        print(
            f"""Epoch {epoch}:
              Learning rate = {lr_now},
              Training loss = {tloss},
              Validation loss = {vloss}""")

        # Add to log and save:
        log_entry = {
            "tloss": tloss,
            "vloss": vloss,
            "learning_rate": lr_now
        }
        metrics_log["history"].append(log_entry)
        save_metrics(metrics_log, dirs["log"])

        # SAVE MODEL STATE:
        if epoch % save_per_epoch == 0:
            save_name = f"{model_name}_{version:02d}_{epoch:03d}.pt"
            save_path = str(Path(dirs["model"]) / save_name)
            torch.save(model.state_dict(), save_path)
    
    return model, metrics_log


if __name__ == "__main__":
    pass