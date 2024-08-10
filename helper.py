import cv2
import tkinter as tk
import numpy as np
from PIL import Image
 
 
def load_image(img_filepath):
    image = cv2.imread(img_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
 
 
def photo_image(image):
    height, width = image.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    # data = ppm_header + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()
    data = ppm_header + image.tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')
 
 
def rescale_image(image, canvas_dims):
 
    height, width = image.shape[0], image.shape[1]
    canvas_height, canvas_width = canvas_dims[0], canvas_dims[1]
    # Ensure longer axis matches:
    if height > width:
        ratio = height / canvas_height
        new_dims = (int(width / ratio), canvas_height)
    else:
        ratio = width / canvas_width
        new_dims = (canvas_width, int(height / ratio))
   
    new_image = cv2.resize(image, new_dims)
    return new_image
 
 
def render_mask(mask, color, alpha):
    """Given a mask Boolean image, color, and alpha value, returns the
    PIL image to be plotted.
    """
 
    R = mask * color[0]
    G = mask * color[1]
    B = mask * color[2]
    alpha_mask = 255 * mask * alpha
 
    img_arr = np.stack([R, G, B, alpha_mask], axis=2).astype(np.uint8)
    img = Image.fromarray(img_arr)
    return img
 
 
def render_back(mask, alpha):
    """Renders a dim background around the segmentation mask.
    """
 
    B = 255 * mask
    alpha_mask = 255 * (1 - mask) * alpha
    img_arr = np.stack([B, B, B, alpha_mask], axis=2).astype(np.uint8)
    img = Image.fromarray(img_arr)
    return img
 
 
def get_neighbour_points(x, y, width, img_shape):
 
    # Star point:
    x_arr = [x, x-width, x+width, x, x]
    y_arr = [y, y, y, y-width, y+width]
 
    # Clip onto image shape:
    x_arr = np.clip(x_arr, 0, img_shape[0]).astype(np.int32)
    y_arr = np.clip(y_arr, 0, img_shape[1]).astype(np.int32)
 
    coords = []
    for i in range(x_arr.size):
        coords.append((x_arr[i], y_arr[i]))
   
    return coords