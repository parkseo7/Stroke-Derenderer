import torch
import numpy as np

# Weight decay:
epochs = 30
peak_lr = 0.01
peak_epoch = 10
base_lr = 0.1 * peak_lr
decay_lr = 0.1 * peak_lr


# Combine this into a function:
def lr_lambda(epochs, peak_epoch, base_lr, peak_lr, decay_lr):

    warmup_slope = (peak_lr - base_lr) / (peak_epoch - 1)
    gamma = np.log(decay_lr / peak_lr) / (epochs - peak_epoch)

    def lr_fn(epoch):
        if epoch <= peak_epoch:
            lr = warmup_slope * (epoch - 1) + base_lr
        else:
            lr = peak_lr * np.exp(gamma*(epoch - peak_epoch))
            print(lr)
        
        return lr

    return lr_fn


def get_norm_stats(loader):
    """Given a Pytorch dataloader, get the mean and standard deviation.
    """

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0, 2])
        