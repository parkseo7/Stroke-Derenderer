"""Module to binarize a coloured line of text.
"""

import numpy as np
import cv2

from npi_demo.helper.helper_bin import (
    image_correction,
    process_image_lab,
    binarize_bradley,
    normalize_image
)

class Binarization:

    def __init__(self, **params):
        self.bg_blur_ratio = params.get("bg_blur_ratio", 0.3) # Percentage of image length
        self.gamma = params.get("gamma", 0.6)
        self.radius = params.get("radius", 8) # Radius of binarization
        self.wellner = params.get("wellner", 15) # Lower values is more strict
        self.bg_blur_max = params.get("bg_blur_max", 81)

        # Grayscale conversion:
        self.lab = params.get("lab", True)
        lab_coeffs = params.get("lab_coeffs", [2, 1, 1.5])
        lab_coeffs = lab_coeffs / np.sum(lab_coeffs)
        self.lab_coeffs = lab_coeffs
        self.br_method = params.get("method", "bradley")


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

        # Default method:
        if self.br_method == "bradley":
            img_bin = binarize_bradley(img_bw, 
                                       radius=self.radius, 
                                       wellner=self.wellner)
        
        else:
            _, img_bin = cv2.threshold(255 - img_bw, 0, 255,
                                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img_bin.astype(np.uint8)

