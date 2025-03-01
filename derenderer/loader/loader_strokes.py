"""Loader for the ResNet Stroke Estimation model. Uses a pre-trained resnet
model, while feeding in coloured black-and-white binarized images.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, Normalize, InterpolationMode
from torch.nn.utils.rnn import pad_sequence

# Parameters
IMG_SIZE = 224 # Keep this fixed for ResNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD, BOS, EOS = 0, 1, 2 # Tokens

class StrokeEstimationDataset(Dataset):
    """Loader for the Resnet stroke estimation model. Can load in multiple batch
    directories. Loads in coloured images and normalizes them.
    """

    def __init__(self, **params):
        # Parameters:
        img_size = params.get("img_size", IMG_SIZE)
        mean = params.get("mean", MEAN)
        std = params.get("std", STD)
        self.img_size = img_size

        # Transforms:
        self.resize = Resize((img_size, img_size))
        self.normalize = Normalize(mean, std)

        # Data pairs
        self.data_paths = []
        self.foldernames = params.get("foldernames", ["binarized", "skeleton"])
    

    def __len__(self):
        return len(self.data_paths)
    
    
    def __getitem__(self, idx):
        """Get the indexed input and label. Returns the (normalized) input image, 
        boolean label, and the image width (for padding). 
        """

        img_filepath, label_filepath = self.data_paths[idx]
        # Load and resize:
        img = self.load_and_resize(img_filepath)
        labels = torch.load(str(label_filepath), weights_only=True)
        length = labels.size(0)

        return img, labels, length
    

    def load_and_resize(self, img_filepath):
        """Loads in the image with 3 channels (RGB). Applies resizing.
        """
        img = read_image(img_filepath, ImageReadMode.RGB)
        h, w = img.size(1), img.size(2)
        if h != IMG_SIZE or w != IMG_SIZE:
            img = self.resize(img)
        
        return img
    

    def get_data_paths(self, folderpath, batches=None):
        """Get all image files from a folder containing batches of data.
        Here, batches is a dictionary of foldername: list of folder pair
        basenames (e.g. binarized, skeleton).
        """
        
        if batches is None:
            batch_paths = [x for x in Path(folderpath).glob('*') if x.is_dir()]
            batch_names = [x.stem for x in batch_paths]
            batches = {k: self.foldernames for k in batch_names}
        
        for batch_folder, foldernames in batches.items():
            for name in foldernames:
                base_path = Path(folderpath) / batch_folder
                data_pairs = self.get_filepaths(base_path, name, f"tokens_{name}")
                self.data_paths += data_pairs
    

    def get_filepaths(self, folderpath, input_folder, label_folder):
        """For a given folderpath with images, labels as subfolders, get
        the list of all input, label pairs.
        """
        img_folderpath = Path(folderpath) / input_folder
        label_folderpath = Path(folderpath) / label_folder

        # Get list of labels:
        labels_filepaths = [x for x in label_folderpath.glob("*.pt")]
        # Check if there's a corresponding image file:
        data_pairs = []
        for label_filepath in labels_filepaths:
            basename = label_filepath.stem
            img_filepath = img_folderpath / f"{basename}.png"
            if img_filepath.exists():
                data_pair = (str(img_filepath), str(label_filepath))
                data_pairs.append(data_pair)
            
        return data_pairs


    def pad_collate_fn(self, batch):
        """Function to collate data samples into batch tensors. Applies padding
        to the batch in order to stack the images into a single batch.
        Outputs the labels in 4 different resolutions using resize.
        Use with DataLoader.
        """
        imgs = []
        labels = []
        lengths = []
        for img, label, length in batch:
            imgs.append(img.unsqueeze(0))
            labels.append(label)
            lengths.append(length)

        # Batch, pad, and convert data types:
        img_batch = torch.cat(imgs, dim=0).type(torch.float32) / 255.
        # Apply normalization to (B, C, H, W):
        img_batch = self.normalize(img_batch)
        
        # Pad the labels, keep lengths:
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=PAD)
        labels_pad = labels_pad.type(torch.LongTensor)
        lengths = torch.tensor(lengths).type(torch.int32)

        return img_batch, labels_pad, lengths