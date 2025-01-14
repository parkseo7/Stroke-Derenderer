# Stroke Derenderer

This repository contains the models and processing code to derender images containing a single line of horizontally written text.

Stroke derendering is an advanced handwriting recognition task that converts offline pixel images of written characters to online strokes. That is, derendering is the estimation of strokes that trace an image’s words to recover their natural writing order.

## Installation

Run the following commands in the repository's base directory to install using `setup.py`.

```
conda create --name myenv
conda activate myenv
conda install python==3.10
pip install -e .
```

To enable and run pytesting, use the commands
```
pip install -e '.[test]'
pytest
```

All model training is done with Pytorch. Inferences are done with OnnxRuntime.

Use ONNX version `1.16` and onnxruntime version `1.18`.

## Models

Stroke derendering is composed of two main components: Text segmentation and stroke estimation.

To run inferences, download the onnx model files [here](LINK).

## Repository structure

Submodule | Description
:--------:|:-----------
config | Configuration `.sh` files and `.yaml` files.
data | Processes and exports usable offline and online strokes for training.
models | Pytorch and onnx methods for loading data, training models, and running inferences.
helper | Helper functions for all methods.