"""Module to binarize a coloured line of text.
"""

import numpy as np
import cv2

from npi_demo.helper.helper_bin import (
    image_correction,
    process_image_lab,
    normalize_image,
    binarize_bradley,
    resize_image,
    convert_to_binary
)

class Binarization:

    def __init__(self, **params):
        self.bg_blur_ratio = params.get("bg_blur_ratio", 0.6) # Percentage of image length
        self.gamma = params.get("gamma", 0.6)
        self.radius = params.get("radius", 8) # Radius of binarization
        self.wellner = params.get("wellner", 15) # Lower values is more strict

        # Grayscale conversion:
        self.lab = params.get("lab", True)
        lab_coeffs = params.get("lab_coeffs", [2, 1, 1.5])
        lab_coeffs = lab_coeffs / np.sum(lab_coeffs)
        self.lab_coeffs = lab_coeffs

        self.rs_length = params.get("rs_length", 48)


    def process_and_binarize(self, img_filepath):
        """Applies pre-processing and binarization, followed by resizing on both.
        Returns both the resized pre-processed image (input) and binary image to
        be used for further segmentation.
        """

        img = cv2.imread(img_filepath)

        br = Binarization()
        img_pp = br.preprocess(img)

        # Actual preprocessed is 255 - img_pp
        img_rs = br.resize(255 - img_pp)

        # Use binarized image only to segment the text further:
        img_bin = br.binarize(img_pp)
        img_bin = np.clip(255 * img_bin.astype(np.int32), 0, 255).astype(np.uint8)

        img_bin_rs = br.resize(img_bin)
        img_bin_rs = br.binarize_simple(img_bin_rs)

        return img_rs, img_bin_rs
    

    def preprocess(self, img):
        """Preprocesses the image into a grayscale image.
        """

        # Normalize image:
        img_rs = normalize_image(img)

        # Get blur size:
        length = min(img_rs.shape[0], img_rs.shape[1])
        blur_size = int(length * self.bg_blur_ratio)
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

