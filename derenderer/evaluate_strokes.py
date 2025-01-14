"""Script to implement stroke estimation from a binarized image, using the
ONNX stroke estimation model.
"""

import numpy as np
import cv2

from derenderer.common import (
    load_json,
    init_onnx_session,
    normalize_image
)

from derenderer.helper.partition import (
    get_binarized_islands,
    group_islands,
    resize_and_pad_image,
    clip_strokes,
    inverse_transform,
    sort_strokes
)

# Default parameters:
IMG_SIZE = 224
MARGIN = 2
MAX_LENGTH = 384

# Keep these default as they pertain to Resnet normalization:
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD, BOS, EOS = 0, 1, 2

class StrokeEstimationSession:

    def __init__(self, configs_path=None, **params):

        if configs_path is not None:
            params_configs = load_json(configs_path)
            params.update(params_configs)
        
        # Optional parameter:
        self.max_length = params.get("max_length", MAX_LENGTH)

        # Parameters:
        self.img_size = params.get("image_size", IMG_SIZE)
        self.margin = params.get("margin", MARGIN)
        
        self.mean = params.get("mean", MEAN)
        self.std = params.get("std", STD)
        self.enc_image_size = params.get("encode_image_size", 14)


    @property
    def tgt_shape(self):
        return (self.img_size, self.img_size)
    

    def _normalize_image(self, img_bin):
        """Normalizes a binarized image, converting it into 3 channels.
        """

        img_norm = normalize_image(img_bin.astype(np.uint8))
        imgs_rgb = []
        for i in range(3):
            img_ch = (img_norm / 255. - self.mean[i]) / self.std[i]
            imgs_rgb.append(img_ch)
        
        img_rgb = np.stack(imgs_rgb, axis=0).astype(np.float32)
        return img_rgb
    

    def _encode_postprocess(self, enc):
        """Given a stack of encoded images, applies post-processing. This is
        to replace the AdaptiveAveragePool2d layer in the encoder model.
        """

        # Expand:
        B, C = enc.shape[0], enc.shape[1]
        E = self.enc_image_size
        # Adaptive average pooling (this is just repeating value on 2x2 grid):
        enc_adp = np.zeros((B, C, E, E), dtype=np.float32)
        enc_adp[:, :, ::2, ::2] = enc
        enc_adp[:, :, 1::2, 1::2] = enc
        enc_adp[:, :, ::2, 1::2] = enc
        enc_adp[:, :, 1::2, ::2] = enc
        
        # Permute:
        enc_out = np.transpose(enc_adp, (0, 2, 3, 1))
        # Flatten out the center
        enc_out = np.reshape(enc_out, (B, -1, C))
        return enc_out.astype(np.float32)
    

    def _decode_to_stroke_points(self, token_seq):
        """Given a token sequence of ordered stroke points, returns the
        sequences of stroke points.
        """

        length_seq = len(token_seq)
        seqs_decode = []
        seq = []
        for n in range(length_seq):
            token = token_seq[n]
            
            # Add current stroke and start a new stroke
            if token == BOS:
                seqs_decode.append(seq)
                seq = []
            
            # Add current stroke and terminate
            elif token == EOS:
                seqs_decode.append(seq)
                break

            # Ignore token if out of range (invalid vertex token)
            elif token == PAD:
                continue

            else:
                index_v = token - EOS - 1
                seq.append(index_v)

        # Remove all empty sequences:
        seqs_decode = [L for L in seqs_decode if len(L) > 0]

        # Convert each stroke sequence into coordinate values:
        strokes = []
        for i, seq in enumerate(seqs_decode):
            # Odd number of coordinates: Remove the last one.
            if len(seq) % 2 != 0:
                seq = seq[:-1]
            X = np.array(seq[::2])
            Y = np.array(seq[1::2]) - self.img_size
            strokes.append((X, Y))
        return strokes


    def _inverse_transform(self, strokes, partition):
        """Given a partition, apply the inverse transform to get the strokes
        back to the original position.
        """

        trans1 = partition["translate1"]
        ratio = partition["ratio"]
        trans2 = partition["translate2"]
        strokes_trans = inverse_transform(strokes, trans1, ratio, trans2)
        return strokes_trans
    

    def load_orts(self, filepaths):
        """Loads the following ORT sessions from ONNX files:
        - encoder: The image encoder
        - projection: The encoder projection
        - decoder_init: The initial h, c state linear layers
        - decoder_iter: The iterative stroke estimator
        - decoder_embedding: The token embedder
        """

        orts = {k: init_onnx_session(v) for k, v in filepaths.items()}
        return orts
    

    def process_image(self, img_bin, orts, max_length=None):
        """Given a binarized image, obtain the estimated strokes. Optionally,
        specify the maximum token sequence length, such that the inferencing
        will terminate if it reaches the maximum length. The default max length
        value will be used otherwise.
        """

        if max_length is None:
            max_length = self.max_length

        partitions = self.get_partitions(img_bin)
        imgs_input = []
        for i, part in enumerate(partitions):
            img_part = part["image_input"]
            imgs_input.append(img_part.astype(np.float32))

        imgs_input = np.stack(imgs_input, axis=0)
        predictions = self.estimate_strokes(imgs_input, orts, 
                                            max_length=max_length)
        strokes = self.process_tokens(predictions, partitions)
        return strokes
    

    def get_partitions(self, img_bin):
        """Partition the image using the binarized islands (connected areas),
        then apply resizing and padding to fit each partitioned image to
        the model input size.
        """

        # Gather segmented images:
        island_info = get_binarized_islands(img_bin.astype(np.uint8), 
                                            margin=self.margin)
        islands, img_islands, num_islands = island_info

        # Group the binarized islands:
        img_h = img_bin.shape[0]
        img_shape = (img_h, img_h)
        islands_groups = group_islands(islands, img_shape)

        # Resize and pad each partitioned images:
        partitions = []
        for i, part in enumerate(islands_groups):
            # Crop and resize the binarized image:
            img, (y, x) = part[0], part[1]
            img_norm = normalize_image(img)
            output_rs = resize_and_pad_image(img_norm, self.tgt_shape,
                                             margin=1, 
                                             pad_value=0)
            img_rs, ratio, (x2, y2) = output_rs
            # Apply image normalization using mean and std for model input:
            img_model = self._normalize_image(img_rs)
            
            # Compile partition:
            partitions.append({
                "image": img_rs,
                "image_input": img_model, # Save as a main input
                "translate1": (x, y),
                "ratio": ratio,
                "translate2": (x2, y2),
            })
        
        return partitions


    def process_tokens(self, tokens, partitions):
        """Given the token outputs of the model and the transformation
        information in the partitions, decodes the tokens into strokes
        and transforms each set of strokes to align with the original image.
        """

        strokes = []
        strokes_parts = []
        N = min(tokens.shape[0], len(partitions))
        for n in range(N):
            tokens_part = tokens[n]
            part = partitions[n]
            strokes_part = self._decode_to_stroke_points(tokens_part)
            strokes_part = clip_strokes(strokes_part, self.tgt_shape)
            strokes_transf = self._inverse_transform(strokes_part, part)
            strokes += strokes_transf
            strokes_parts.append(strokes_part)
        
        # Sort the strokes and re-orient into standard orientation:
        strokes_sort = sort_strokes(strokes)
        return strokes_sort
    

    def estimate_strokes(self, images, orts, max_length=384):
        """Given a batch of images with the proper shape, applies iterative
        stroke estimation. Terminates at the max token length.
        """

        # Initialize:
        enc = orts["encoder"].run(["output"], {"input": images})
        enc = self._encode_postprocess(enc[0]) # Output size is 1

        # Apply projection if applicable:
        if "projection" in orts:
            enc = orts["projection"].run(["output"], {"input": enc})
            enc = enc[0] # Take item
        
        B, P, E = enc.shape
        h, c = orts["decoder_init"].run(["output_h", "output_c"], 
                                        {"input": np.mean(enc, axis=1)})
        
        # Initial embedding:
        tokens_start = BOS * np.ones((B,), dtype=np.int32)
        emb = orts["decoder_embedding"].run(["output"],
                                            {"input": tokens_start})
        emb = emb[0]

        inds_inc = np.arange(B)
        inds = np.arange(B)

        # Create tensors to hold prediction scores and focal map (focal):
        predictions = np.zeros((B, max_length), dtype=np.int32)
        for t in range(max_length):
            preds, h, c = orts["decoder_iter"].run(
                ["output_pred", "output_h", "output_c"],
                {
                    "input_enc": enc[inds_inc],
                    "input_emb": emb,
                    "input_h": h[inds],
                    "input_c": c[inds]
                }
            )
            # Get token predictions:
            tokens = np.argmax(preds, axis=1).astype(np.int32)
            predictions[inds_inc, t] = tokens
            inds = np.where(tokens != EOS)[0]
            inds_inc = inds_inc[inds]

            # Terminate if there are no remaining indices:
            if inds_inc.size == 0:
                break

            # Update embeddings:
            emb = orts["decoder_embedding"].run(["output"],
                                                {"input": tokens[inds]})
            emb = emb[0]
        
        return predictions