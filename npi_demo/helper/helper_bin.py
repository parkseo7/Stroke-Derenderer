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


def split_image(img, img_bin, split_length, axis=0):
    """Given a binarized image, applies a vertical bar scan to determine
    where to divide up the image to be below the split dimension, such that
    each split image can be padded.
    """
    min_axis = axis
    max_axis = 1 - min_axis
 
    length = max(img_bin.shape[0], img_bin.shape[1])
    bool_clear = ~np.any(img_bin, axis=min_axis)
    inds_clear = np.where(bool_clear)[0]
    # Find indices in bool_clear < split_length
    done = False
    inds = [0]
    ind_cut = 0
    while not done:
       
        # Get as close to split_length as possible.
        inds0 = inds_clear - ind_cut
        inds_cand = np.where(np.logical_and(inds0 > 0, inds0 < split_length))
        inds_cand = inds0[inds_cand]
        # No candidates: Cut at split length:
        if inds_cand.size == 0:
            ind_new = split_length
        else:
            ind_new = np.max(inds_cand)
       
        inds.append(ind_cut + ind_new)
        ind_cut += ind_new
 
        # Check if done:
        if ind_cut >= length:
            done = True
   
    # Slice the image:
    images = []
    for i in range(len(inds)-1):
        ind_s = inds[i]
        ind_f = inds[i+1]
        if min_axis == 0:
            new_image = img[:,ind_s:ind_f]
        else:
            new_image = img[ind_s:ind_f,:]
       
        images.append(new_image)
   
    # Determine whether to remove the last image:
    image_f = images[-1]
 
    # Check if it has any pixels:
    frac_pos = np.sum(image_f) / image_f.size
    is_neg = frac_pos < 0.10
    is_short = image_f.shape[max_axis] < image_f.shape[min_axis] // 2 
 
    if is_short or is_neg:
        return images[:-1]
    else:
        return images
 
 
def pad_image(image, pad_length, axis=0):
    """Pads a binary image so that it matches the padded dimension. Fills it
    with black. Here, make the pad length the same as the split length above.
    Otherwise, cut the image so that it matches pad length.
    """
 
    min_axis = axis
    max_axis = 1 - min_axis
   
    min_length = image.shape[min_axis]
    max_length = image.shape[max_axis]
 
    diff_length = pad_length - max_length
    # Pad image:
    if diff_length >= 0:
        pad_length1 = diff_length // 2
        if diff_length % 2 == 0:
            pad_length2 = pad_length1
        else:
            pad_length2 = pad_length1 + 1
 
        if min_axis == 0:
            pad_shape1 = (min_length, pad_length1)
            pad_shape2 = (min_length, pad_length2)
        else:
            pad_shape1 = (pad_length1, min_length)
            pad_shape2 = (pad_length2, min_length)
       
        # Concatenate:
        pad1 = np.zeros(pad_shape1, dtype=image.dtype)
        pad2 = np.zeros(pad_shape2, dtype=image.dtype)
        pad_image = np.concatenate((pad1, image, pad2), axis=max_axis)
 
    else:
        pad_length1 = -diff_length // 2
        if diff_length % 2 == 0:
            pad_length2 = pad_length1
        else:
            pad_length2 = pad_length1 + 1
       
        pad_image = image[pad_length1:-pad_length2]
   
    return pad_image
 
 
def save_binary_image(filepath, image):
    """Converts a binary image into a colored array to be saved into a .png.
    """
 
    # img_save = np.clip(255 * image.astype(np.int32), 0, 255).astype(np.uint8)
    img_save = np.stack([image, image, image], axis=2)
    cv2.imwrite(filepath, img_save)
    return img_save