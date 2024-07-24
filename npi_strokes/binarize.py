"""Module to binarize a coloured line of text.
"""

import numpy as np
import cv2
from pathlib import Path
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

from npi_strokes.helper.helper_bin import (
    image_correction,
    process_image_lab,
    normalize_image,
    binarize_bradley,
    resize_image,
    convert_to_binary,
    split_image,
    pad_image,
    save_binary_image
)

class Binarization:

    def __init__(self, **params):
        self.bg_blur_ratio = params.get("bg_blur_ratio", 0.6) # Percentage of image length
        self.gamma = params.get("gamma", 0.6)
        self.radius = params.get("radius", 8) # Radius of binarization
        self.wellner = params.get("wellner", 15) # Lower values is more strict
        self.bg_blur_max = params.get("bg_blur_max", 81)

        # Grayscale conversion:
        self.lab = params.get("lab", True)
        lab_coeffs = params.get("lab_coeffs", [2, 1, 1.5])
        lab_coeffs = lab_coeffs / np.sum(lab_coeffs)
        self.lab_coeffs = lab_coeffs

        rs_length = params.get("rs_length", 48)
        split_length = params.get("split_length", 48*4)

        self.rs_length = rs_length
        self.split_length = split_length
        self.axis = 0
        
        # Transformations:
        self.torch_resize = transforms.Resize((rs_length, split_length))


    def load_as_torch(self, img_filepath):
        """Loads an image file from the file path and sets it up as a tensor input.
        For reference only.
        """

        img = read_image(img_filepath, ImageReadMode.GRAY)
        if (img.shape[-2], img.shape[-1]) != (self.rs_length, self.split_length):
            img = self.torch_resize(img)
         
        return img # 1, H, W


    def process_text_line(self, img):
        """Given an image of a single text line, pre-processes into an input for
        the model.
        """

        img_pp = self.preprocess(img)

        # Actual preprocessed is 255 - img_pp
        img_rs = self.resize(255 - img_pp)

        # Use binarized image only to segment the text further:
        img_bin = self.binarize(img_pp)
        img_bin = np.clip(255 * img_bin.astype(np.int32), 0, 255).astype(np.uint8)

        img_bin_rs = self.resize(img_bin)
        img_bin_rs = self.binarize_simple(img_bin_rs)
        pad_imgs, lengths = self.split_text_line(img_rs, img_bin_rs)
        if len(pad_imgs) != len(lengths):
            lengths = lengths[:-1]
            
        return pad_imgs, lengths
    

    def load_and_save(self, img_filepath, save_folderpath, label=0):
        """Loads a line of image text. Segments the image and saves each cut image.
        Attaches the label and the segment number. If handwritten, label = 1.
        Otherwise, label = 0.
        """

        # Load and binarize:
        img_pp, img_bin = self.process_and_binarize(img_filepath)

        # Split and pad images:
        pad_imgs = self.split_text_line(img_pp, img_bin)
        basename = Path(img_filepath).stem
        for i, pad_img in enumerate(pad_imgs):
            frac = np.count_nonzero(pad_img)
            if frac == 0:
                continue
            filepath = Path(save_folderpath) / f"{basename}_{i:02d}_{label}.png"
            save_binary_image(filepath, pad_img)
        
        return pad_imgs


    def process_and_binarize(self, img_filepath):
        """Applies pre-processing and binarization, followed by resizing on both.
        Returns both the resized pre-processed image (input) and binary image to
        be used for further segmentation.
        """

        img = cv2.imread(img_filepath)

        img_pp = self.preprocess(img)

        # Actual preprocessed is 255 - img_pp
        img_rs = self.resize(255 - img_pp)

        # Use binarized image only to segment the text further:
        img_bin = self.binarize(img_pp)
        img_bin = np.clip(255 * img_bin.astype(np.int32), 0, 255).astype(np.uint8)

        img_bin_rs = self.resize(img_bin)
        img_bin_rs = self.binarize_simple(img_bin_rs)

        return img_rs, img_bin_rs
    
    
    def split_text_line(self, img_pp, img_bin):
        """Splits the pre-processed image using the binary image into
        fixed-length input images to train a model.
        """

        w = self.split_length
        images = split_image(img_pp, img_bin, w, axis=0)
        lengths = [x.shape[1-self.axis] for x in images]
        pad_images = [pad_image(x, w, axis=self.axis) for x in images]

        return pad_images, lengths
    

    def preprocess(self, img):
        """Preprocesses the image into a grayscale image.
        """

        # Normalize image:
        img_rs = normalize_image(img)

        # Get blur size:
        length = min(img_rs.shape[0], img_rs.shape[1])
        blur_size = min(int(length * self.bg_blur_ratio), self.bg_blur_max)
        # Blur size must always be odd
        if blur_size % 2 == 0:
            blur_size += 1

        # Use LAB space:
        if self.lab:
            img_out = process_image_lab(img_rs,
                                        coeffs=self.lab_coeffs,
                                        gamma=self.gamma,
                                        blur_size=blur_size)
        # Convert to grayscale:
        else:
            img_gray = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
            img_out = image_correction(img_gray, 
                                       gamma=self.gamma, 
                                       blur_size=blur_size)
        
        # Clip and convert to int8:
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)

        # Text should be minority so average must be less than 128:
        avg_value = np.mean(img_out)
        if avg_value < 128:
            img_out = 255 - img_out
        
        return img_out


    def binarize(self, img_bw):
        """Binarizes the image using Bradley's method, which is a fast
        local thresholding method.
        """
        img_bin = binarize_bradley(img_bw, 
                                   radius=self.radius, 
                                   wellner=self.wellner)
        return img_bin.astype(np.uint8)
    

    def resize(self, img):
        img_rs = resize_image(img, self.rs_length, axis=0)
        return img_rs
    

    def binarize_simple(self, img_bw):
        """Binarizes using a simple thresholding method. To be used on
        shrunken binary images.
        """

        img_bin = convert_to_binary(img_bw)
        return img_bin

