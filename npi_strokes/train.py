import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms

from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from npi_class.model.efficientnet import EfficientNet
from npi_class.model.shufflenet import ShuffleNet
from npi_class.model.dataset import TextBWDataset

import pickle

EPS = 1e-6


class ModelTrainer:

    def __init__(self, folderpath):
        
        self.dir_base = Path(folderpath)
        self.dirs = self._setup_directories(folderpath)


    def lr_scheduler(self, **lr_params):
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
                lr = warmup_slope * (epoch - 1) + base_lr
            else:
                lr = peak_lr * np.exp(gamma * (epoch - peak_epoch))
            return lr
        
        return lr_fn


    def _setup_directories(self, folderpath):
        """Set-up directories with the given folderpath.
        """

        dir_model = folderpath / "models" # For checkpoint models
        dir_log = folderpath / "log" # For model logs

        dir_model.mkdir(parents=True, exist_ok=True)
        dir_log.mkdir(parents=True, exist_ok=True)

        dirs = {
            "model": dir_model,
            "log": dir_log
        }
        return dirs
    

    def train_model(self, args, train_dataset, val_dataset):
        """Given arguments for the model and training, train the model.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        
        # Model attributes:
        args_model = args["model"]
        model_name = args_model["name"]
        version = args_model["version"]
        log_filename = f"{model_name}_{version}.pkl"
        log_path = Path(self.dirs["log"]) / log_filename

        # Initialize log:
        metrics_log = {
            "history": [],
            "model": args_model,
            "stats": [train_dataset.mean, train_dataset.std]
        }

        args_train = args["train"]
        batch_size = args_train.get("batch_size", 64)
        num_workers = args_train.get("num_workers", 16)
        save_per_epoch = args_train.get("save_per_epoch", 10)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
        
        # Create model:
        n_ch = args_model["channels"]
        width = args_model["width"]
        depth = args_model["depth"]
        num_classes = args_model["classes"]

        # model = EfficientNet(n_ch, width_mult=width, depth_mult=depth, num_classes=num_classes)
        model = ShuffleNet(groups=3, in_channels=n_ch, num_classes=num_classes)

        # Move to GPU:
        model = model.to(device)

        # Optimizers and loss function:
        args_train = args["train"]
        num_epochs = args_train["epochs"]
        lr = args_train.get("lr", 1e-4)
        betas = args_train.get("betas", (0.9, 0.95))
        grad_clip = args_train.get("grad_clip", 5)
        weight_decay = args_train.get("weight_decay", 1e-4)
        lr_schedule = args_train.get("lr_schedule", [1.0, 0.1])
        fac_s, fac_f = lr_schedule[0], lr_schedule[1]
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_lambda = self.lr_scheduler(**args_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss().to(device)

        # TRAINING LOOP:
        for epoch in range(1, num_epochs+1):

            train_iter = iter(train_loader)
            val_iter = iter(val_loader)

            tloss = train_step(model, optimizer, train_iter, criterion, device, 
                               grad_clip=grad_clip, epoch=epoch)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Training loss = {tloss}, lr = {lr}")
            vloss = val_step(model, val_iter, criterion, device, epoch=epoch)
            print(f"Epoch {epoch}: Validation loss = {vloss}")

            # Add to log and save:
            log_entry = {
                "tloss": tloss,
                "vloss": vloss
            }
            metrics_log["history"].append(log_entry)
            save_metrics(metrics_log, log_path)

            if epoch % save_per_epoch == 0:
                save_name = f"{model_name}_{version}_{epoch}.pt"
                save_path = str(Path(self.dirs["model"]) / save_name)
                torch.save(model.state_dict(), save_path)
        
        return metrics_log, model


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


def train_step(model, opt, train_iter, criterion, device, grad_clip=5, epoch=1):
    """Training step for the model.
    """

    model.train()
    tloss = 0
    for batch in tqdm(train_iter, desc=f"Training epoch {epoch}"):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward propagation:
        probs = model(imgs)
        loss = criterion(probs, labels)

        # Training step:
        opt.zero_grad()
        loss.backward()
        tloss += loss.item()

        # Gradient clipping:
        clip_gradient(opt, grad_clip)

        # STEP:
        opt.step()

        # Wait for GPU to finish work
        # torch.cuda.synchronize()
    
    tloss = tloss / len(train_iter)
    return tloss


def val_step(model, val_iter, criterion, device, epoch=1):
    model.eval()
    vloss = 0
    with torch.no_grad():
        for batch in tqdm(val_iter, desc=f"Validation epoch {epoch}"):        
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward propagation:
            probs = model(imgs)
            loss = criterion(probs, labels)

            # Training step:
            vloss += loss.item()
        
        vloss = vloss / len(val_iter)
    return vloss



def save_metrics(metrics, filename):
    """Save metrics to a pickle file

    Args:
        variable (any): variable to be saved
        filename (string): location to save the data
    """
    with open(filename, "wb") as fid:
        pickle.dump(metrics, fid)
