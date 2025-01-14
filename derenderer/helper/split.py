"""Helper functions to split and pad an image into equally sized subimages.
Also provides functions to glue back the images to recover the original image.
Used in preprocessing and postprocessing methods for the text segmentation model.
"""

import numpy as np
import cv2


def split_image(img, target_width, overlap, pad_value=0):
    """Given an image and a target width, partitions the image to cut images
    with the given target width. Applies padding to ensure the width is the
    same.
    """

    h, w = img.shape[0], img.shape[1]

    # Case: Image width is shorter than target width. No partition required.
    if w < target_width:
        img_pad = pad_image(img, target_width, pad_value=pad_value)
        return [img_pad], [w]
    
    else:
        # Determine number of partitions:
        num_imgs = w // (target_width - overlap) + 1
        w_unpad = w // num_imgs
        imgs_cut = []
        widths_unpad = [] # Keep track of original widths to remove pad later
        for i in range(num_imgs):
            ind_s = i * w_unpad
            ind_f = (i+1) * w_unpad + overlap # Last image has no overlap
            img_cut = img[:, ind_s:ind_f]
            widths_unpad.append(img_cut.shape[1])

            # Pad image:
            img_pad = pad_image(img_cut, target_width, pad_value=pad_value)
            imgs_cut.append(img_pad)
        
        return imgs_cut, widths_unpad
    

def pad_image(img, width, pad_value=0):
    """Pads the image with the pad value to the desired width. Adds the padding
    to the right of the image.
    """

    pad_diff = width - img.shape[1]
    if pad_diff > 0:
        img_pad = cv2.copyMakeBorder(img, 0, 0, 0, pad_diff, 
                                     cv2.BORDER_CONSTANT, value=pad_value)
    # Do not apply padding
    else:
        img_pad = img[:, :width]
    return img_pad


def cut_and_stack(imgs_text, target_dim, overlap, pad_value=0):
    """Partitions each text line image after resizing, so that each
     piece is the same size. Returns a stack of cut images, alongside the
     original widths and the index to the whole image per cut piece.
     """

    # Cut image and stack
    B, C, H, W = target_dim
    img_stack = []
    stack_indices = []
    stack_widths = []
    img_widths = []
    counter = 0
    for i, img in enumerate(imgs_text):
        img_rs = resize_to_height(img, H)
        imgs_cut, widths = split_image(img_rs, W, overlap, 
                                           pad_value=pad_value)
        stack_widths.append(widths)
        stack_indices.append([counter + k for k in range(len(imgs_cut))])
        img_stack += imgs_cut
        img_widths.append(img_rs.shape[1])
        counter += len(imgs_cut)
    
    # Apply permutation on each image in the stack and then stack:
    if C == 1:
        img_stack = [x[:, :, None] for i, x in enumerate(img_stack)]
    img_stack = [np.transpose(x, (2, 0, 1)) for _, x in enumerate(img_stack)]
    img_stack = np.stack(img_stack, axis=0)

    return img_stack, stack_indices, stack_widths, img_widths


def reconstruct_images(img_output, imgs_widths, 
                       stack_indices, stack_widths, 
                       overlap):
    """Given the binarized output from the model, reconstructs the original
    images using the stack indices and stack widths. Here, stack indices
    and stack widths are a list of lists. The image output is a padded array
    of size B, C, H, W.
    """

    B, C, H, W = img_output.shape

    # Create list of (binarized images):
    num_imgs = len(stack_indices)
    img_bins = []
    for i in range(num_imgs):
        img_width = imgs_widths[i]
        indices = stack_indices[i]
        widths = stack_widths[i]
        
        # Create blank image:
        img_bin = np.zeros((H, img_width, C)).astype(np.uint8)
        ind_s = 0
        # Decode all images from stack:
        for k, ind in enumerate(indices):
            img = img_output[ind]
            width = widths[k]
            img_tr = np.transpose(img[:, :, :width], (1, 2, 0))
            # Paste the image onto binary image (max to handle overlap):
            img_bin[:, ind_s:ind_s+width, :] = \
                np.maximum(img_bin[:, ind_s:ind_s+width,:], img_tr)
            ind_s += width - overlap
        
        # Add to list of (binarized) images:
        img_bins.append(img_bin)
    
    return img_bins


def resize_to_height(img, height):
    """Resizes the given image to match the height.
    """

    h, w = img.shape[0], img.shape[1]
    ratio = height / h
    width = int(w * ratio)
    img_rs = cv2.resize(img, (width, height))
    return img_rs