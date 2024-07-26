"""The main script to apply NPI to batches of images. Converts a text image
into strokes. Saves .json files for the HWR engine to read and render.
"""

import argparse
import sys
import os

import numpy as np
import cv2

from npi_demo.detection.core.submodules.sub_module_text_det import TextDetection
from npi_demo.detection.core.utilis.onnx_helper import ONNXHelper
from npi_demo.stroke_estimation import PageStrokeEstimation


def parse_args(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('img_filepath', type=str, help="Path to the image file")
    parser.add_argument('json_filepath', type=str, help="Path to the JSON file")
    parser.add_argument('num_workers', type=int, default=0, help="Number of CPUs used")
    args = parser.parse_args()
    return args


def main(img_filepath, json_filepath, num_workers=0):
    """The main function that converts an image into strokes, formatted as a
    .json file.
    """

    # Load onnx sessions:
    onnx_handler = ONNXHelper()
    onnx_sessions = onnx_handler.load_sessions(["text OCR"])
    td = TextDetection(onnx_sessions) # Changed parameters (lower threshold)

    # LOAD AND PREDICT BOUNDING BOXES
    img = cv2.imread(str(img_filepath)) #, cv2.COLOR_BGR2RGB)
    bboxes = td.predict_onnx_det(img)

    pse = PageStrokeEstimation(img)
    all_strokes, all_lws = pse.estimate_strokes(bboxes, max_workers=num_workers)
    json_file = pse.export_json(all_strokes, all_lws, json_filepath)

    results = {
        "strokes": all_strokes,
        "line_widths": all_lws,
        "image": img,
        "bboxes": bboxes
    }
    return json_file, results


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.img_filepath, args.json_filepath, num_workers=args.num_workers)