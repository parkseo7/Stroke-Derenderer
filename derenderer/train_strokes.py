"""Methods for training an encoder-decoder image captioning model.
"""

from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim

from derenderer.common import (
    load_yaml, 
    load_metrics, 
    save_metrics
)

from derenderer.model.decoder import Decoder
from derenderer.model.encoder import initialize_encoder
from derenderer.loader import StrokeEstimationDataset


def display_log(log_dict):
    """Given a dictionary log, displays all elements in the log.
    """

    display_str = ""
    for key, value in log_dict.items():
        display_str += f"{key} = {value}\n"
    print(display_str[:-1]) # Omit last \n
    

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def lr_scheduler(is_encoder=True, **lr_params):
    """Creates a learning rate scheduler, which warms up to a certain
    epoch, followed by a decay to the last epoch.
    """
    
    epochs = lr_params["epochs"]
    peak_epoch = lr_params["peak_epoch"]
    if is_encoder:
        name = "encoder"
    else:
        name = "decoder"    
    base_lr = lr_params[f"{name}_lr"]
    peak_lr = lr_params[f"{name}_peak_lr"]
    decay_lr = lr_params[f"{name}_decay_lr"]

    warmup_slope = (peak_lr - base_lr) / (peak_epoch - 1)
    gamma = np.log(decay_lr / peak_lr) / (epochs - peak_epoch)

    def lr_fn(epoch):
        if epoch <= peak_epoch:
            lr = max(warmup_slope * (epoch - 1) + base_lr, base_lr)
        else:
            lr = peak_lr * np.exp(gamma * (epoch - peak_epoch))
        return lr
    
    return lr_fn


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
    version = f"{model_name}_{version:02d}"
    dir_version = dir_model / version
    dir_version.mkdir(parents=True, exist_ok=True)
    log_filepath = dir_log / f"{version}.pkl"
    dirs = {
        "model": dir_version,
        "log": log_filepath
    }
    return dirs


def prepare_datasets(vargs, dirs):
    """Prepares the data by reading each .pkl file. Returns the training
    and test datasets.
    """
    vargs_dir = vargs["directories"]
    dir_data = vargs_dir["data"]
    vargs_data = vargs["dataset"]
    train_batches = vargs_data.get("train_batches")
    val_batches = vargs_data.get("val_batches")
    vargs_model = vargs["model"]

    train_dataset = StrokeEstimationDataset()
    train_dataset.get_data_paths(dir_data, train_batches)
    val_dataset = StrokeEstimationDataset()
    val_dataset.get_data_paths(dir_data, val_batches)

    version = Path(dirs["model"]).name

    # Set-up log metrics file.
    log_dict = {
        "name": version,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "parameters": vargs_model["parameters"],
        "history": []
    }
    save_metrics(log_dict, dirs["log"])

    # Configure dataloaders:
    vargs_model = vargs["model"]
    vargs_train = vargs_model["train"]
    batch_size = vargs_train.get("batch_size", 16) # Low batch size
    num_workers = vargs_train.get("num_workers", 16)

    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=train_dataset.pad_collate_fn,
                          num_workers=num_workers,
                          pin_memory=False) # Keep this False, strains the GPU
    val_dl = DataLoader(val_dataset,
                        batch_size=batch_size, 
                        shuffle=False, 
                        collate_fn=val_dataset.pad_collate_fn,
                        num_workers=num_workers,
                        pin_memory=False)
    
    return train_dl, val_dl, log_dict


def prepare_models(vargs, device="cpu"):
    """Prepares the models and the optimizer using the parameters given 
    in vargs.
    """

    vargs_model = vargs["model"]
    params = vargs_model["parameters"]

    # Get encoder, encoder dimension:
    enc_name = vargs_model["encoder"]
    encoder = initialize_encoder(enc_name, **params)

    # Load from checkpoints:
    enc_checkpoint = vargs_model.get("encoder_checkpoint")
    if enc_checkpoint is not None:
        enc_weights = torch.load(enc_checkpoint, weights_only=True, 
                                 map_location=device)
        encoder.load_state_dict(enc_weights)
    
    # Get encoder dimension:
    _, encoder_dim = encoder.resolution
    decoder = Decoder(encoder_dim=encoder_dim, **params)
    dec_checkpoint = vargs_model.get("decoder_checkpoint")
    if dec_checkpoint is not None:
        dec_weights = torch.load(dec_checkpoint, weights_only=True, 
                                 map_location=device)
        decoder.load_state_dict(dec_weights)

    # Loss + optimizer:
    criterion = nn.CrossEntropyLoss()

    # Optimizers: With scheduler, set learning rate to 1.0.
    dec_params = filter(lambda p: p.requires_grad, decoder.parameters())
    dec_optimizer = torch.optim.Adam(params=dec_params, lr=1.0)
    enc_params = filter(lambda p: p.requires_grad, encoder.parameters())
    enc_optimizer = torch.optim.Adam(params=enc_params, lr=1.0)

    # Put encoder, decoder, loss function to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = criterion.to(device)

    return encoder, decoder, enc_optimizer, dec_optimizer, criterion


def train_epoch(encoder, enc_optimizer, enc_scheduler,
                decoder, dec_optimizer, dec_scheduler,
                configs, train_dl, criterion, device, epoch=1):
    """Single training epoch.
    """
    # Parameters:
    batch_size = configs["batch_size"]
    grad_clip = configs["grad_clip"]
    alpha_c = configs["alpha_c"]

    # Reset iterations:
    train_iter = iter(train_dl)

    # TRAINING STEP:

    # Set both models to training mode:
    encoder.train()
    decoder.train()
    total_tloss = 0
    total_tloss_img = 0 # For record
    total_tloss_stoc = 0 # For record
    desc = f"Training epoch {epoch}"
    for (inputs, labels, lengths) in tqdm(train_iter, desc=desc):
        # Move to devices:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # Forward propagation
        imgs_enc = encoder(inputs) # Outputs img, cls encodings
        dec_output = decoder(imgs_enc, labels, lengths, device=device)
        predictions, focals, inds_sort = dec_output
        labels = labels[inds_sort]
        lengths = lengths[inds_sort]

        # Per image, calculate loss. Record both losses:
        B = lengths.size(0)
        tloss_img = None
        for b in range(B):
            pred = predictions[b]
            length = lengths[b]
            label = labels[b]
            
            # Unpadded prediction, labels:
            pred_unpad = pred[:length-1]
            label_unpad = label[1:length]
            loss = criterion(pred_unpad, label_unpad)
            if tloss_img is None:
                tloss_img = loss
            else:
                tloss_img += loss
        
        # Add doubly stochastic attention regularization:
        tloss_stoc = alpha_c * ((1. - focals.sum(dim=1)) ** 2).mean()
        tloss = tloss_img + tloss_stoc

        # Zero-grad:
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # Back propagation:
        tloss.backward()

        # Gradient clipping:
        clip_gradient(enc_optimizer, grad_clip)
        clip_gradient(dec_optimizer, grad_clip)
        
        # Optimizer step
        enc_optimizer.step()
        dec_optimizer.step()

        # Add to total loss (for display):
        total_tloss += tloss.item()
        total_tloss_img += tloss_img.item()
        total_tloss_stoc += tloss_stoc.item()

    # Adjust values after training epoch:
    enc_scheduler.step()
    dec_scheduler.step()
    enc_lr = enc_scheduler.get_last_lr()[0]
    dec_lr = dec_scheduler.get_last_lr()[0]
    
    # Return log update:
    return {
        "tloss": total_tloss / (len(train_iter) * batch_size),
        "encoder_lr": enc_lr,
        "decoder_lr": dec_lr,
        "tloss_img": total_tloss_img / (len(train_iter) * batch_size),
        "tloss_stoc": total_tloss_stoc / (len(train_iter) * batch_size)
    }


def val_epoch(encoder, decoder, configs, val_dl, criterion, device, epoch=1):
    """Single validation epoch.
    """

    # Parameters:
    batch_size = configs["batch_size"]
    alpha_c = configs["alpha_c"]

    # Reset iterations:
    val_iter = iter(val_dl)

    # VALIDATION STEP:
    encoder.eval()
    decoder.eval()
    total_vloss = 0
    total_vloss_img = 0 # For record
    total_vloss_stoc = 0 # For record
    desc = f"Validation epoch {epoch}"
    with torch.no_grad():
        for (inputs, labels, lengths) in tqdm(val_iter, desc=desc):
            # Move to devices:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward propagation
            imgs_enc = encoder(inputs) # Outputs img, cls encodings
            dec_output = decoder(imgs_enc, labels, lengths, device=device)
            predictions, focals, inds_sort = dec_output
            labels = labels[inds_sort]
            lengths = lengths[inds_sort]

            # Per image, calculate loss:
            B = lengths.size(0)
            vloss_img = None
            for b in range(B):
                pred = predictions[b]
                length = lengths[b]
                label = labels[b]
                
                # Unpadded prediction, labels:
                pred_unpad = pred[:length-1]
                label_unpad = label[1:length]
                loss = criterion(pred_unpad, label_unpad)
                if vloss_img is None:
                    vloss_img = loss
                else:
                    vloss_img += loss

            # Add doubly stochastic attention regularization:
            vloss_stoc = alpha_c * ((1. - focals.sum(dim=1)) ** 2).mean()
            vloss = vloss_img + vloss_stoc

            # Add to total loss (for display):
            total_vloss += vloss.item()
            total_vloss_img += vloss_img.item()
            total_vloss_stoc += vloss_stoc.item()

    # Return log update:
    return {
        "vloss": total_vloss / (len(val_iter) * batch_size),
        "vloss_img": total_vloss_img / (len(val_iter) * batch_size),
        "vloss_stoc": total_vloss_stoc / (len(val_iter) * batch_size)
    }


def main(vargs):
    """Main script to set-up and train a model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    # cudnn.benchmark = True

    dirs = setup_directories(vargs)
    
    # Get datasets:
    train_dl, val_dl, metrics_log = prepare_datasets(vargs, dirs)

    # Create model:
    vargs_model = vargs["model"]
    model_name = vargs_model["name"]
    version = vargs_model["version"]

    all_models = prepare_models(vargs, device=device)
    encoder, decoder, enc_optimizer, dec_optimizer, criterion = all_models

    # Optional checkpoint:
    checkpoint = vargs_model.get("checkpoint")
    if checkpoint is not None:
        enc_path = checkpoint["encoder"]
        dec_path = checkpoint["decoder"]
        enc_weights = torch.load(enc_path, weights_only=True,
                                 map_location=device)
        encoder.load_state_dict(enc_weights)
        dec_weights = torch.load(dec_path, weights_only=True,
                                 map_location=device)
        decoder.load_state_dict(dec_weights)
        
    # Configure loss and optimizer:
    vargs_train = vargs_model["train"]
    epochs = vargs_train.get("epochs", 10)
    save_per_epoch = vargs_train.get("save_per_epoch", 2)

    enc_lambda = lr_scheduler(is_encoder=True, **vargs_train)
    dec_lambda = lr_scheduler(is_encoder=False, **vargs_train)
    enc_scheduler = torch.optim.lr_scheduler.LambdaLR(enc_optimizer, enc_lambda)
    dec_scheduler = torch.optim.lr_scheduler.LambdaLR(dec_optimizer, dec_lambda)

    for epoch in range(1, epochs+1):

        # Implement training epoch:
        log_train = train_epoch(encoder, enc_optimizer, enc_scheduler,
                                decoder, dec_optimizer, dec_scheduler,
                                vargs_train, train_dl, criterion, device, 
                                epoch=epoch)
        display_log(log_train)

        # Implement validation epoch:
        log_val = val_epoch(encoder, decoder, vargs_train, val_dl, 
                            criterion, device, epoch=epoch)
        display_log(log_val)

        # Update log history and save:
        log_entry = {}
        log_entry.update(log_train)
        log_entry.update(log_val)
        metrics_log["history"].append(log_entry)
        save_metrics(metrics_log, dirs["log"])

        # Save the models:
        if epoch % save_per_epoch == 0:
            # SAVE ENCODER, DECODER MODEL STATE:
            enc_name = f"{model_name}_{version:02d}_encoder_{epoch:03d}.pt"
            dec_name = f"{model_name}_{version:02d}_decoder_{epoch:03d}.pt"
            enc_path = str(Path(dirs["model"]) / enc_name)
            dec_path = str(Path(dirs["model"]) / dec_name)
            torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)

    return encoder, decoder, train_dl, val_dl, metrics_log