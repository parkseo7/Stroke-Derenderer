"""Functions for pre-processing and binarization.
"""

import numpy as np
import cv2
COEFFS = [4/9, 2/9, 3/9]

def binarize_bradley(img, radius=8, wellner=15):
    """Implement Bradley's adaptive thresholding technique to binarize the 
    image. This is a quick, conventional binarization method that is
    robust against variable lighting.
    """

    h, w = img.shape[0], img.shape[1]
    length = min(h, w)
    s = int(length / radius) // 2 # Use 1/8 of shorter length
    t = wellner # Wellner values
    t_per = (100 - t) / 100
    
    # Get integral image:
    img_int = integral_img(img)
    img_out = np.zeros(img.shape, dtype=np.bool_)

    # Define vectorized matrices:
    x1 = np.maximum(np.arange(w) - s, 0)
    x2 = np.minimum(np.arange(w) + s, w-1)
    y1 = np.maximum(np.arange(h) - s, 0)
    y2 = np.minimum(np.arange(h) + s, h-1)

    # Count matrix:
    count_x, count_y = np.meshgrid(x2 - x1, y2 - y1)
    count = count_x * count_y

    # Image integral matrix:
    x2_, y2_ = np.meshgrid(x2, y2)
    x1_, y1_ = np.meshgrid(x1, y1)
    sum1 = img_int[y2_, x2_] + img_int[y1_, x1_]

    x2_, y1_ = np.meshgrid(x2, y1)
    x1_, y2_ = np.meshgrid(x1, y2)
    sum2 = img_int[y1_, x2_] + img_int[y2_, x1_]
    SUM = sum1 - sum2

    img_out = (img * count) > (SUM * t_per)
    return (1 - img_out).astype(np.bool_)


def image_correction(img_bw, gamma=0.7, blur_size=51):
    """Given a grayscale image, subtracts both the median background
    colour and a blurred background. Applies the subtractions separately,
    then takes a skewed average.
    """

    bg_blur = cv2.medianBlur(img_bw, blur_size)
    img_bg = np.abs(img_bw - np.median(bg_blur)).astype(np.uint8)

    # Here, background is white, foreground is black (output format):
    img_out = 255 - img_bg
    img_out = gamma_correction(img_out, gamma=gamma)
    return img_out
    

def process_image_lab(img, coeffs=COEFFS, gamma=0.7, blur_size=51):
    """Use each of the LAB channels to process the image into grayscale.
    Do this as the final step of preprocessing (after resizing).
    """

    # Image is BGR. Convert to LAB:
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Take all three LAB channels, and apply image correction:
    img_out = np.zeros((img.shape[0], img.shape[1]))

    for i in range(3):
        ch = image_correction(img_lab[:,:,i], gamma=gamma, blur_size=blur_size)
        img_out += ch * coeffs[i]
    
    # Normalize:
    img_out = normalize_image(img_out)
    return img_out
    

# SUPPLEMENTARY FUNCTIONS
def integral_img(img):
    """Computes the integral image from the input, which is used in 
    Bradley's method. Assumes the image is a 2D array. 
    This is the sum of left, diagonal, and above pixels.
    """

    # Ensure image data type is np.uint8:
    img0 = img.astype(np.uint8)
    img_int = cv2.integral(img0)

    return img_int[1:,1:]


def gamma_correction(img, gamma=0.7):
    """Build a lookup table to apply gamma correction onto an image.
    Here, image is a cv2 image, for which you use cv2.imread to open
    a string file path.
    """

    inv_gamma = 1.0 / gamma
    table = (255 * (np.arange(256) / 255.0) ** inv_gamma).astype(np.uint8)
    adj_img = cv2.LUT(img, table)

    return adj_img


def normalize_image(image):
    """Normalizes the image by converting to float and centering by mean, std.
    """

    normalized_image = cv2.normalize(image, None, 0, 255, 
                                     norm_type=cv2.NORM_MINMAX)
    return normalized_image


def resize_image(image, rs_length, axis=0):
    """Given an image, resize it so that the shorter length matches short_dim.
    """
    
    ratio = rs_length / image.shape[axis]
    new_max = int(ratio * image.shape[axis])
    new_min = int(ratio * image.shape[1-axis])

    if axis == 0:
       new_image = cv2.resize(image, (new_min, new_max))
    else:
       new_image = cv2.resize(image, (new_max, new_min))
    return new_image


def convert_to_binary(img_bw, thr=128):
    """Given a BW image, convert to binary. Ensure that the text is in the
    foreground by taking minority colour. Foreground is labelled 1 and
    background is labelled 0.
    """
    
    img_bin = img_bw < thr
    frac = np.count_nonzero(img_bin) / img_bin.size
    # Get inverse if True is majority (ensure text is foreground):
    if frac > 0.5:
        img_bin = 1 - img_bin

    return img_bin
