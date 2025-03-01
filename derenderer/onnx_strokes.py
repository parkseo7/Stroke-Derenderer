"""Script to convert model to ONNX, and run the ONNX model.
"""

import torch
import numpy as np
from derenderer.model.unet_attention import UnetAttentionEval
import onnxruntime

from derenderer.helper.split import (
    cut_and_stack,
    reconstruct_images
)

from derenderer.common import (
    load_image,
    save_image,
    normalize_image,
    resize_to_height
)

# Default parameters
HEIGHT = 256
WIDTH = 256 * 3
CHANNELS = 3
OVERLAP = 256 // 2
BIN_THR = 0.5
MINIBATCH = 8

class BinarizationModel:


    def __init__(self, **params):
        
        # ONNX INPUTS
        self.height = params.get("height", HEIGHT)
        self.width = params.get("width", WIDTH)
        self.channels = params.get("channels", CHANNELS)

        self.overlap = params.get("overlap", OVERLAP)
        # Binarization threshold
        self.bin_thr = params.get("bin_thr", BIN_THR)
        # Batch size for model inference
        self.minibatch = params.get("minibatch", MINIBATCH)
    

    def load_pt_model(self, pt_filepath, init_features):
        """Loads the pytorch model.
        """

        model = UnetAttentionEval(in_channels=3, init_features=init_features)
        model_weights = torch.load(pt_filepath, weights_only=True, 
                                   map_location=torch.device('cpu'))
        model.load_state_dict(model_weights)
        # Set to inference mode
        model.eval()
        return model
    

    def export_model_to_onnx(self, model, onnxpath):
        """Exports the given model to ONNX, fixing the channels, height,
        width.
        """

        # Input:
        C = self.channels
        H = self.height
        W = self.width
        X = torch.randn(1, C, H, W, requires_grad=True)
        y = model(X) # To check

        # Configure dynamic axis (batch dimension):
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        # Export the model:
        torch.onnx.export(model, X, onnxpath,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes=dynamic_axes
                          )
        
    
    def init_onnx_inference(self, onnxpath):
        """Given an ONNX file path, initialize an inference session.
        """
        ort = onnxruntime.InferenceSession(onnxpath, 
                                           providers=["CPUExecutionProvider"])
        return ort
    

    def ort_predict(self, input_numpy, ort):
        """Given a stack of Numpy inputs in the form (B, C, H, W), returns
        the output from the ort session.
        """

        ort_input = {"input": input_numpy}
        ort_outputs = ort.run(None, ort_input)
        output = ort_outputs[0]
        return output
    

    def preprocess_image(self, images):
        """Pre-processes the image by resizing to the target height, then
        partitioning the image into a batch stack. Returns the tensor
        stack and the information to reconstruct the image for post-processing.
        Warning: Keep the number of images to a minimum to avoid memory issues.
        """

        images_rs = []
        for i, image in enumerate(images):
            img_rs = resize_to_height(image, self.height)
            images_rs.append(img_rs)
        
        target_dim = (1, 3, self.height, self.width)
        stack = cut_and_stack(images_rs, target_dim, self.overlap)
        img_stack, stack_indices, stack_widths, img_widths = stack
        return img_stack, stack_indices, stack_widths, img_widths
    

    def model_predict(self, img_stack, ort):
        """Given a processed image stack, returns the model output.
        Processes the stack in mini-batches, then concatenates them
        together.
        """

        # Split the stack into mini-stacks:
        B = img_stack.shape[0]
        num_batches = B // self.minibatch + 1
        batch_outputs = []
        for num in range(num_batches):
            ind_s = num * self.minibatch
            ind_f = (num+1) * self.minibatch
            img_ministack = img_stack[ind_s:ind_f]
            ort_input = {"input": (img_ministack / 255.).astype(np.float32)}
            ort_outputs = ort.run(None, ort_input)
            output = ort_outputs[0]
            # In the shape (B, H, W)
            img_out = 255 * (output > self.bin_thr).astype(np.uint8)
            # Unsqueeze channel dimension to shape (B, C, H, W):
            if len(img_out.shape) == 3:
                img_out = img_out[None, :, :, :]
            batch_outputs.append(img_out)

        # Concatenate the outputs together:
        if len(batch_outputs) == 1:
            imgs_output = batch_outputs[0]
        
        else:
            imgs_output = np.concatenate(batch_outputs, axis=0)
        return imgs_output


    def postprocess_stack(self, imgs_output, stack_indices, 
                          stack_widths, img_widths):
        """Given the processed model output in the shape (B, C, H, W),
        returns them reconstructed image.
        """

        imgs_pp = reconstruct_images(imgs_output, img_widths, 
                                     stack_indices, stack_widths, 
                                     self.overlap)
        # Convert to binary and remove lone channel
        imgs_pp = [img[:, :, 0] > 128 for i, img in enumerate(imgs_pp)]
        return imgs_pp
    

    def binarize_images(self, images, ort):
        """The full pipeline of image thinning, using the UNet attention model.
        Warning: Keep the number of images to a minimum to avoid memory issues.
        """

        stack = self.preprocess_image(images)
        imgs_stack, stack_indices, stack_widths, img_widths = stack
        imgs_output = self.model_predict(imgs_stack, ort)
        imgs_pp = self.postprocess_stack(imgs_output, stack_indices, 
                                         stack_widths, img_widths)
        return imgs_pp
    

    def binarize_image(self, image, ort):
        """Binarizes a single image.
        """

        imgs_pp = self.binarize_images([image], ort)
        return imgs_pp[0]