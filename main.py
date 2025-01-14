"""Main script module to run both the ONNX binarization model and ONNX
stroke estimation model.
"""

import time
import numpy as np
import cv2
from pathlib import Path
import argparse

from derenderer.evaluate_binarize import BinarizationSession
from derenderer.evaluate_strokes import StrokeEstimationSession
from derenderer.common import (
    load_image, 
    save_image,
    save_json,
    normalize_image
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-modelspath", "-modelspath", required=True,
                        help="Path to the folder containing all model files.")
    parser.add_argument("-input", "-input", default="./images/input",
                        help="Path to the folder containing all input images.")
    parser.add_argument("-output", "--output", default="./images/output",
                        help="Output directory to save all image outputs.")
    
    args = parser.parse_args()
    return args


def initialize_sessions(folderpath):
    """Initializes the binarization and stroke estimation sessions.
    Returns the following:
    - BinarizationSession: Contains methods and hyperparameters for binarization.
    - ORT (binarization): The ORT session for binarization.
    - StrokeEstimationSession: Contains methods and hyperparameters for stroke
    estimation.
    - ORT (stroke estimation): The ORT sessions for stroke estimation.
    """

    bin_filepath = str(Path(folderpath) / "binarizer.onnx")
    configs_bin_path = str(Path(folderpath) / "configs_binarizer.json")

    # Initialize all sessions:
    bs = BinarizationSession(configs_path=configs_bin_path)
    ort_bs = bs.init_onnx_inference(bin_filepath)

    configs_se_path = str(Path(folderpath) / "configs_strokes.json")
    se = StrokeEstimationSession(configs_path=configs_se_path)
    ort_paths = {
        "encoder": str(Path(folderpath) / "encoder.onnx"),
        "decoder_init": str(Path(folderpath) / "decoder_init_hc.onnx"), 
        "decoder_iter": str(Path(folderpath) / "decoder_iter.onnx"), 
        "decoder_embedding": str(Path(folderpath) / "decoder_embedding.onnx")
    }

    # Optionally add projection:
    if (Path(folderpath) / "projection.onnx").exists():
        ort_paths["projection"] = str(Path(folderpath) / "projection.onnx")
    orts_se = se.load_orts(ort_paths)

    return bs, ort_bs, se, orts_se


def load_images(img_filepaths):
    """Given a list of image file paths, loads all images. For each image,
    also returns the corresponding image file name.
    """

    imgs = []
    for img_filepath in img_filepaths:
        filename = Path(img_filepath).stem
        img = load_image(img_filepath)
        imgs.append((img, filename))
    
    return imgs


def convert_to_float(X, Y):
    """Converts numpy array into float lists.
    """

    N = min(X.size, Y.size)
    X_list = [float(X[n]) for n in range(N)]
    Y_list = [float(Y[n]) for n in range(N)]
    return X_list, Y_list


def main(imgs, bs, ort_bs, se, orts_se, output_folder, strokes=True):
    """Given a list of images (with filenames) and all initialized sessions,
    apply binarization and stroke estimation for each image. If strokes is
    False, then only apply binarization to each image. Saves the binarized images
    and strokes to the output folder with the image file name.

    Note: The binarized image will be saved at a fixed height resolution.
    The strokes will be rescaled to fit the original image dimension.
    """
    
    for (img, filename) in imgs:

        height = img.shape[0]

        # Apply binarization:
        start_bin = time.time()
        img_bin = bs.binarize_image(img, ort_bs)
        img_bin = img_bin[:, :, 0] > (255 * bs.bin_thr)
        end_bin = time.time()
        comp_bin_time = round(end_bin - start_bin, 4)
        
        # Save binarized image:
        img_bin_save = normalize_image(img_bin.astype(np.uint8))
        bin_filepath = str(Path(output_folder) / f"{filename}_BINARIZED.png")
        save_image(img_bin_save, bin_filepath, grayscale=True)
        print(f"{filename} took {comp_bin_time} seconds to binarize. " +
              f"Result is saved to {bin_filepath}")
        
        # Get stroke estimation (optional):
        if strokes:
            # Get height ratio (for strokes rescaling):
            ratio = height / img_bin.shape[0]
            start_se = time.time()
            strokes = se.process_image(img_bin, orts_se, max_length=None)
            end_se = time.time()
            comp_se_time = round(end_se - start_se, 4)
            # Rescale strokes to original image resolution:
            strokes_rs = []
            for (X, Y) in strokes:
                X_list, Y_list = convert_to_float(X*ratio, Y*ratio)
                strokes_rs.append((X_list, Y_list))
            
            st_filepath = str(Path(output_folder) / f"{filename}_STROKES.json")
            save_json(strokes_rs, st_filepath)
            print(f"{filename} took {comp_se_time} seconds to estimate strokes. " +
                  f"Result is saved to {st_filepath}")


if __name__ == "__main__":
    # Default script: Uses the input, output folders in ./images
    vargs = parse_args()
    output_folder = vargs.output
    
    # Get all images in input folder:
    input_folder = vargs.input
    img_filepaths = [str(x) for x in Path(input_folder).glob("*.png")]
    imgs = load_images(img_filepaths)

    # Initialize models:
    modelpath = vargs.modelspath
    bs, ort_bs, se, orts_se = initialize_sessions(modelpath)

    # Get inferences for every image:
    main(imgs, bs, ort_bs, se, orts_se, output_folder, strokes=True)