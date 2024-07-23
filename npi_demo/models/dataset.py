"""Module to format the dataset and dataloader.
"""

from pathlib import Path
import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torch.utils.data import Dataset


class TextBWDataset(Dataset):

    def __init__(self, filepaths=None, labels=None, rs=(48, 48*4)):

        self.filepaths = filepaths
        self.labels = labels
        
        self.mean = [0]
        self.std = [1]
        self.rs = rs

        # Normalizer:
        self.normalize = transforms.Normalize([0], [1])
        self.resize = transforms.Resize(rs)

        self.handwritten = "handwritten" 
        self.printed = "printed" 

    
    def __len__(self):
        return len(self.filepaths)
    

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        img = read_image(filepath, ImageReadMode.GRAY)
        if (img.shape[-2], img.shape[-1]) != self.rs:
            img = self.resize(img)
        
        # Transform image:
        img_norm = self.normalize(img.type(torch.float32) / 255)
        return img_norm, label
    

    def get_stats(self):
        """Updates the mean and std transform using the current set
        of filepaths.
        """

        mean = 0.0
        meansq = 0.0
        N = len(self.filepaths)

        for idx in range(N):
            img = read_image(self.filepaths[idx], ImageReadMode.Gray)
            if (img.shape[-2], img.shape[-1]) != self.rs:
                img = self.resize(img)
            
            img_norm = img.type(torch.float32) / 255
            mean += img_norm.mean()
            meansq += (img_norm**2).mean()
        
        mean = mean / N
        meansq = meansq / N
        std = torch.sqrt(meansq - mean**2)

        self.update_transform(mean, std)

        return mean, std
    

    def update_transform(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean, std)


def get_filepaths(basepath, hw_folders, pr_folders, split=0.95):
    """Obtain the image filepaths, given a basepath with handwritten,
    printed folders. Specify a list of handwritten, printed folders
    we want to use.
    """
 
    basepath = Path(basepath)
   
    all_train_filepaths = []
    all_train_labels = []
    all_val_filepaths = []
    all_val_labels = []
 
    # Get all directories:
    hw_path = Path(basepath) / "handwritten"
    for folder in hw_folders:
        folderpath = hw_path / folder
 
        # Folder exists. Extract all .png paths
        if folderpath.exists() and folderpath.is_dir():
            img_filepaths = [str(x) for x in folderpath.glob("*.png")]
            img_labels = [1 for x in img_filepaths]
 
            # Train-validation split:
            inds = np.arange(len(img_filepaths))
            inds_perm = np.random.permutation(inds)
 
            num_split = max(int(inds_perm.size * (1 - split)), 1)
            inds_train = inds_perm[:-num_split]
            inds_val = inds_perm[-num_split:]
 
            train_filepaths = [img_filepaths[i] for i in inds_train]
            train_labels = [img_labels[i] for i in inds_train]
 
            val_filepaths = [img_filepaths[i] for i in inds_val]
            val_labels = [img_labels[i] for i in inds_val]
 
            all_train_filepaths += train_filepaths
            all_train_labels += train_labels
            all_val_filepaths += val_filepaths
            all_val_labels += val_labels
   
 
    pr_path = Path(basepath) / "printed"
    for folder in pr_folders:
        folderpath = pr_path / folder
        # Folder exists. Extract all .png paths
        if folderpath.exists() and folderpath.is_dir():
            img_filepaths = [str(x) for x in folderpath.glob("*.png")]
            img_labels = [0 for x in img_filepaths]
 
            # Train-validation split:
            inds = np.arange(len(img_filepaths))
            inds_perm = np.random.permutation(inds)
 
            num_split = max(int(inds_perm.size * (1 - split)), 1)
            inds_train = inds_perm[:-num_split]
            inds_val = inds_perm[-num_split:]
 
            train_filepaths = [img_filepaths[i] for i in inds_train]
            train_labels = [img_labels[i] for i in inds_train]
 
            val_filepaths = [img_filepaths[i] for i in inds_val]
            val_labels = [img_labels[i] for i in inds_val]
 
            all_train_filepaths += train_filepaths
            all_train_labels += train_labels
            all_val_filepaths += val_filepaths
            all_val_labels += val_labels
 
    return (all_train_filepaths, all_train_labels), \
        (all_val_filepaths, all_val_labels)