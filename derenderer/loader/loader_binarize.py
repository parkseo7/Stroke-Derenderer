"""Data loader for the binarization model. Set-up the dataloader and processing 
functions to resize and pad the data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, InterpolationMode
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

# PARAMETERS:
HEIGHT = 128
WIDTH = 128 * 3
DOWNSCALE_FACS = [1, 2, 4, 8] # Fixed
UPSCALE_FACS = [1, 2, 4, 8] # Last one is most important, 8 for thicker lines

class BinarizationDataset(Dataset):
    """Loader for the binarization model. Can load in multiple batch
    directories.
    """

    def __init__(self, **params):

        # Parameters:
        self.height = params.get("height", HEIGHT)
        self.width = params.get("width", WIDTH) # Padded width
        self.data_filepaths = [] # Pairs of (input, label) image file paths
        
        # Input names, label name.
        self.input_names = params.get("input_names", ["original", "augmented"])
        self.label_name = params.get("label_name", "skeleton")

        # Define all resize transoforms:
        self.upscale_facs = params.get("upscale_facs", UPSCALE_FACS)
        downscale_facs = params.get("downscale_facs", DOWNSCALE_FACS)
        self.downscale_facs = downscale_facs

        # Define all resize transforms:
        fac1, fac2, fac3, fac4 = downscale_facs[:4]
        h1 = self.height // fac1
        h2 = h1 // fac2
        h3 = h1 // fac3
        h4 = h1 // fac4
        w1 = self.width // fac1
        w2 = w1 // fac2
        w3 = w1 // fac3
        w4 = w1 // fac4

        self.rs = Resize(h1)
        # Only do nearest interpolation for the first resizing.
        self.rs1 = Resize((h1, w1), interpolation=InterpolationMode.NEAREST)
        self.rs2 = Resize((h2, w2))
        self.rs3 = Resize((h3, w3))
        self.rs4 = Resize((h4, w4))


    def _load_and_resize(self, img_filepath, binary=True):
        """Loads the image and resizes into the stored height. Also permutes 
        the image to make it stackable (width, height, channels). Use
        binary=True for the labels.
        """

        if binary:
            img = read_image(img_filepath, ImageReadMode.GRAY)
            img = self._pad_image(img) # Ensure square image at least
            # Replace this with transforms:
            img = self.rs(img)
            img = img.permute(2, 1, 0)

        else:
            img = read_image(img_filepath, ImageReadMode.RGB)
            img = self._pad_image(img) # Ensure square image at least
            # Replace this with transforms:
            img = self.rs(img)
            img = img.permute(2, 1, 0)

        return img


    def _pad_image(self, img):
        """Pads the image to make it a square image. To be used when the width
        is smaller than the height, since the resizing function rescales
        the shorter dimension. Assumes image shape is (C, H, W)
        """

        c, h, w = img.shape
        # Don't pad if the width is long:
        
        if h <= w:
            return img
        
        else:
            pad_diff = self.height - w
            img_pad = pad(img, (0, pad_diff), value=0)
            return img_pad


    def _pad_stack(self, X):
        """Pads the stack of images to fit the image size.
        """

        b, c, h, w = X.shape
        pad_diff = self.width - w
        imgs_pad = pad(X, (0, pad_diff), value=0)
        return imgs_pad
    

    def pad_collate_fn(self, batch):
        """Function to collate data samples into batch tensors. Applies padding
        to the batch in order to stack the images into a single batch.
        Outputs the labels in 4 different resolutions using resize.
        Use with DataLoader.
        """
        imgs_input = []
        imgs_label = []
        widths = []
        for img_input, img_label, width in batch:
            imgs_input.append(img_input)
            imgs_label.append(img_label)
            widths.append(width)
        
        X = pad_sequence(imgs_input, batch_first=True, padding_value=0)
        Y = pad_sequence(imgs_label, batch_first=True, padding_value=0)

        X = X.permute((0, 3, 2, 1))
        Y = Y.permute((0, 3, 2, 1))

        imgs_input = self._pad_stack(X)
        imgs_label = self._pad_stack(Y)

        widths = torch.tensor(widths)

        # Lower the resolutions of the labels, to compare to different stages.
        imgs_input = self.rs(imgs_input) # In uint8, divide by 255
        imgs_label1 = self.rs1(imgs_label) # In uint8, convert to binary
        imgs_label2 = self.rs2(imgs_label)
        imgs_label3 = self.rs3(imgs_label)
        imgs_label4 = self.rs4(imgs_label)

        # For each label, apply different thresholds to get binarized version:
        imgs_input = imgs_input / 255.

        up1, up2, up3, up4 = self.upscale_facs[:4]
        imgs_label1 = imgs_label1 > (128 // up1)
        imgs_label2 = imgs_label2 > (128 // up2)
        imgs_label3 = imgs_label3 > (128 // up3)
        imgs_label4 = imgs_label4 > (128 // up4) # 8 for thicker strokes

        imgs_label = [imgs_label1, imgs_label2, imgs_label3, imgs_label4]
        return imgs_input, imgs_label, widths


    def get_data_paths(self, folderpath, batches=None):
        """Get all image files from a folder containing batches of data.
        Here, batches is a dictionary of foldername: list of image folders.
        Multiple inputs will have the same label image.
        """

        # Get all batch paths:
        if batches is None:
            batch_paths = [x for x in Path(folderpath).glob('*') if x.is_dir()]
            batch_names = [x.stem for x in batch_paths]
            batches = {k: self.input_names for k in batch_names}

        all_pairs = []
        for batch, input_names in batches.items():
            batch_folderpath = Path(folderpath) / batch
            label_folderpath = batch_folderpath / self.label_name
            label_filepaths = [x for x in label_folderpath.glob("*.png")]

            for label_filepath in label_filepaths:
                filename = label_filepath.stem
                # Check each sub-folder in input names:
                for name in input_names:
                    input_filepath = batch_folderpath / name / f"{filename}.png"
                    if input_filepath.exists():
                        pair = (str(input_filepath), str(label_filepath))
                        all_pairs.append(pair)
            
        # Update data paths:
        self.data_filepaths = all_pairs
        return all_pairs
    

    def __len__(self):
        return len(self.data_filepaths)
    
    
    def __getitem__(self, idx):
        """Get the indexed input and label. Returns the (normalized) input image, 
        boolean label, and the image width (for padding). 
        """

        item = self.data_filepaths[idx]
        input_filepath, label_filepath = item[0], item[1]

        # Load and resize:
        img_input = self._load_and_resize(input_filepath, binary=False)
        img_label = self._load_and_resize(label_filepath, binary=True)

        # Get width:
        w = img_label.shape[0]
        # Process both inputs and outputs:
        x = img_input
        y = img_label

        return x, y, w


def evaluate_loss(preds, img_labels, widths, loss_fn):
    """For reference on training. Evaluates the loss by unpadding the
    images.
    """

    y_pads = preds.squeeze(1).permute(0, 2, 1).permute(1, 0, 2)
    Y_unpad = unpad_sequence(y_pads, widths)

    # For computing loss (binary crossentropy)
    y_pred = torch.concat([img.flatten() for img in Y_unpad])
    y_true = torch.concat([img.flatten() for img in img_labels])

    loss = loss_fn(y_pred, y_true)
    return loss