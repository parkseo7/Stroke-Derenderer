"""Module to implement handwriting-printed classification on a page, using
a binarized classification model. Here, we take an image and its bounding boxes,
and feed each line to the model. We assign the model prediction to each
bounding box.
"""
 
import torch
from torchvision import transforms

import numpy as np
 
from npi_strokes.binarize import Binarization
from npi_strokes.helper.helper_image import (
    expand_bbox,
    crop_rotated_bbox,
)
 
 
class PageTextClassification:
 
    def __init__(self, img, model, **params):
        self.img = img
 
        # Parameters
        self.expand_per = params.get("expand_per", 0.25)
        self.mask_value = params.get("mask_value", (1, 255, 1))
        self.upscale = params.get("upscale", 256)
 
        self.lw_scale = params.get("lw_scale", 0.25) # Scaling between 0.15-0.2
        br_method = params.get("br_method", "bradley")
 
        # Binarizer, stroke estimator:
        self.br = Binarization(method=br_method)
 
        # Model parameters:
        input_dim = params.get("input_dim", (48, 48*4))
        self.input_dim = input_dim
        self.thr = params.get("thr", 128)
        self.axis = params.get("axis", 0) # By default, height is shorter axis
 
        # Transform:
        mean, std = params.get("stats", [0, 1])
        self.resize = transforms.Resize(input_dim)
        self.normalize = transforms.Normalize(mean, std)
 
        self.model = model.eval()
        self.softmax = torch.nn.Softmax(dim=1)
 
   
    def _get_model_inputs(self, bboxes):
        """Obtain the model inputs using the bounding boxes.
        """
        batch_imgs, batch_labels, weights = self.process_batch(bboxes)
 
        batch_labels = np.array(batch_labels)
        weights = np.array(weights)

        X = []
        for k, batch_img in enumerate(batch_imgs):
            img_torch = torch.from_numpy(batch_img)
            if (img_torch.shape[-2], img_torch.shape[-1]) != self.input_dim:
                img_torch = self.resize(img_torch.unsqueeze(0))
                img_torch = img_torch.squeeze(0)
            X.append(img_torch)

        X = torch.stack(X, axis=0).unsqueeze(axis=1).type(torch.float) / 255
        X = self.normalize(X)

        return X, batch_labels, weights
   
 
    def evaluate_texts(self, bboxes):
        """Given bounding boxes for an image, run the inference using the
        model and process the result by taking the weighted mean probability.
        """
 
        X, batch_labels, weights = self._get_model_inputs(bboxes)
 
        probs = self.model(X)
        probs = self.softmax(probs)
        probs = probs.detach().numpy()
 
        labels = np.unique(batch_labels)
        
        cats = np.zeros(len(bboxes), dtype=np.int32)
        preds = []
        for label in labels:
            inds = np.where(np.array(batch_labels) == label)[0]
            probs_cls = probs[inds]
            weights_cls = weights[inds]
 
            weights_cls = weights_cls[:, None]
            # weights_cls = np.stack([weights_cls, weights_cls], axis=1)
            # Get mean probabilities:
            probs_mean = np.sum(weights_cls * probs_cls, axis=0)
            cats[label] = np.argmax(probs_mean).astype(np.int32)
            preds.append(probs_mean)
 
        return cats, preds
   
 
    def process_batch(self, bboxes):
        """Creates a batch of text lines as input for the classification model.
        """
 
        batch_images = []
        batch_labels = []
        batch_weights = []
        for i, bbox in enumerate(bboxes):
            pad_images, lengths = self.process_segment(bbox)
            labels = [i for k in range(len(pad_images))]
 
            batch_images += pad_images
            batch_labels += labels
            batch_weights += lengths
 
        return batch_images, batch_labels, batch_weights
 
 
    def process_segment(self, bbox, img=None):
        """Segment an image using the bounding box. Can be used to generate
        training data from a full image page.
        """
 
        if img is None:
            img = self.img
 
        bbox_exp = expand_bbox(bbox, per=self.expand_per)
        # Perspective transform:
        img_seg, mat = crop_rotated_bbox(img, bbox_exp,
                                         mask_value=self.mask_value)
        pad_images, lengths = self.br.process_text_line(img_seg)
        return pad_images, lengths