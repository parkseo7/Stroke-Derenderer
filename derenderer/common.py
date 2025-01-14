"""A library containing frequently used functions for all methods.
"""

import pickle
import yaml
import json
import cv2
import onnxruntime

EPS = 1e-6


def load_image(img_filepath, grayscale=False):
    """Loads the image as a colored image.
    """

    image = cv2.imread(img_filepath)
    if grayscale:   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, None] # Add channel axis
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(img, save_filepath, grayscale=False):
    """Saves a colored image into a .png file.
    """

    if grayscale:
        cv2.imwrite(save_filepath, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imwrite(save_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_metrics(metrics, filename):
    """Save metrics to a pickle file

    Args:
        variable (any): variable to be saved
        filename (string): location to save the data
    """
    with open(filename, "wb") as fid:
        pickle.dump(metrics, fid)


def load_metrics(filename):
    """Loads metrics from a pickle file

    Args:
        filename (string): location of the pikle file

    Returns:
        any: variables stored in the pickle file
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
    

def load_yaml(filepath):
    """Loads a .yaml file.
    """
    
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    return data_loaded


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(json_dict, save_path):
    """Saves the json dictionary.
    """

    with open(save_path, 'w') as out:
        json.dump(json_dict, out)


def resize_to_height(img, height):
    """Resizes the given image to match the height.
    """

    h, w = img.shape[0], img.shape[1]
    ratio = height / h
    width = int(w * ratio)
    img_rs = cv2.resize(img, (width, height))
    return img_rs


def normalize_image(image):
    """Normalizes the image by converting to float and centering by mean, std.
    """

    normalized_image = cv2.normalize(image, None, 0, 255, 
                                     norm_type=cv2.NORM_MINMAX)
    return normalized_image


def init_onnx_session(onnx_path):
    """Start an ONNX inference, for testing.
    """

    providers = ['CPUExecutionProvider']
    ort = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return ort