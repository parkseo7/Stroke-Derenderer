# Stroke Derenderer

This repository contains the models and processing code to derender images containing a single line of horizontally written text.

Stroke derendering is an advanced handwriting recognition task that converts offline pixel images of written characters to online strokes. That is, derendering is the estimation of strokes that trace an image’s words to recover their natural writing order.

## Installation

To install using Anaconda, run the following commands in the repository's base directory to install using `setup.py`.

```
conda create --name myenv
conda activate myenv
conda install python==3.10
pip install -e .
```

All model training is done with Pytorch. Inferences are done with OnnxRuntime.

Use ONNX version `1.16` and onnxruntime version `1.18`.

## Inferencing

Stroke derendering is composed of two main components: Text segmentation and stroke estimation.

To set-up inferencing,
1. Download the onnx model and configuration files [here](https://drive.google.com/drive/folders/1XbTwFgEDDENve8XuwkHnIYvpSqTnpGTS?usp=drive_link).
2. In the root directory, run `python main.py --models=<model dir> -input=<input dir> --output=<output dir>` where:

- `<model dir>`: Path to the downloaded models and configurations folder.
- `<input dir>`: Path to the folder containing .png images to run inferences on.
- `<output dir>`: (Optional) Path to the folder where all model outputs will be exported. By default it will save to ./images/output.

## Repository structure

Submodule | Description
:--------:|:-----------
models | Pytorch and onnx methods for loading data, training models, and running inferences.
helper | Helper functions for all methods.

##  Outline of methodology

Text segmentation works by applying the UNet model with attention onto images of fixed height and variable width.
The image is resized and partitioned into padded subimages of height 128px and width 384px. The model binarizes the subimages individually, then the subimages are glued to obtain the output of the original image.

Stroke estimation works by applying a vision language model onto a binary image containing only text. The image is partitioned into character-sized subimages by isolating each connected binary island, then clustering nearby islands together. The stroke estimation model is applied to each subimage, a 224px by 224px binary image. The strokes are then rescaled and translated to align with the original image.