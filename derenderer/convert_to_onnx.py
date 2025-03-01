"""Script to convert the binarization and stroke estimation models
into ONNX files.
"""

import numpy as np
import time
from pathlib import Path

from derenderer.convert_bin_to_onnx import BinarizationModel
from derenderer.convert_se_to_onnx import EncoderOnnx, DecoderOnnx
from derenderer.common import load_metrics, save_json


def main(filepaths, bin_json, se_json):
    """Given a dictionary of filepaths containing the model .pt files and
    metrics, saves all models to the designated output folder with fixed
    names.
    """

    # Create output folder:
    output_folder = filepaths["output"]
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    f = Path(output_folder)

    # Compile and export binarization model:
    bin_metrics = load_metrics(filepaths["metrics_bin_path"])
    bin_params = bin_metrics["parameters"]
    bm = BinarizationModel(**bin_json)
    init_features = bin_params["init_features"]
    bin_model = bm.load_pt_model(filepaths["bin_filepath"], init_features)

    bm.export_model_to_onnx(bin_model, str(f / "binarizer.onnx"))
    save_json(bin_json, str(f / "configs_binarizer.json"))

    # Compile and export all stroke estimation models:

    # Save the encoder model:
    encoder = EncoderOnnx(img_size=224, encoded_image_size=14)
    encoder.load(filepaths["enc_filepath"])
    encoder.save(str(f / "encoder.onnx"))

    # Save decoder models:
    se_metrics = load_metrics(filepaths["metrics_se_path"])
    se_params = se_metrics["parameters"]
    decoder = DecoderOnnx(**se_params)
    decoder.load(filepaths["dec_filepath"])

    # Save embedding:
    decoder.save_embedding(str(f /"decoder_embedding.onnx"))
    # Save initial h, initial c:
    decoder.save_init(str(f / "decoder_init_hc.onnx"))
    # Save iterator:
    decoder.save_iter(str(f / "decoder_iter.onnx"))
    # Save JSON:
    save_json(se_json, str(f / "configs_strokes.json"))


def save_jsons(filepaths, bin_json, se_json):
    # Create output folder:
    output_folder = filepaths["output"]
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    f = Path(output_folder)

    save_json(bin_json, str(f / "configs_binarizer.json"))
    save_json(se_json, str(f / "configs_strokes.json"))


if __name__ == "__main__":
    # Compile file paths and JSON files manually here:
    p = {
        "bin_filepath": "./output/binarize_128_04_020.pt",
        "metrics_bin_path": "./output/binarize_128_04.pkl",
        "dec_filepath": "./output/SE_RESNET_04_decoder_017.pt",
        "metrics_se_path": "./output/SE_RESNET_04.pkl",
        "enc_filepath": "./output/SE_RESNET_04_encoder_017.pt",
        "output": "./output/onnx_models"
    }

    bin_json = {
        "height": 128,
        "width": 128*3,
        "channels": 3,
        "overlap": 64,
        "bin_thr": 0.5,
        "minibatch": 8
    }

    se_json = {
        "image_size": 224,
        "margins": 2,
        "encode_image_size": 14,
        "max_length": 384
    }

    # main(p, bin_json, se_json)
    save_jsons(p, bin_json, se_json)