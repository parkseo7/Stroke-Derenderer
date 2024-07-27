"""Module to implement stroke estimation for an entire image, given
detected bounding boxes for segmenting text.
"""

import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

from npi_demo.skeletonize import LineStrokeEstimation
from npi_demo.binarize import Binarization
from npi_demo.helper.helper_image import (
    expand_bbox,
    crop_rotated_bbox,
    upscale_image,
    get_centerpoint
)

EPS = 1e-6

class PageStrokeEstimation:

    def __init__(self, img, **params):
        self.img = img

        # PARAMETERS

        # The percentage by which the text bounding boxes are expanded by to get the segmented text.
        # dtype: positive float value between 0-1.
        self.expand_per = params.get("expand_per", 0.25)

        # The color value to fill in blank regions during tilt correction. Use an uncommon text color here.
        # dtype: 3-tuple of integers from 0-255.
        self.mask_value = params.get("mask_value", (1, 255, 1))

        # The upscaled pixel height for the segmented text image.
        # dtype: positive integer.
        self.upscale = params.get("upscale", 256)
        
        # The line width scaling coefficient to go from predicted NPI line width to value read by HWR engine.
        # dtype: Any float value between 0-1.
        self.lw_scale = params.get("lw_scale", 0.25) # Scaling between 0.15-0.2

        # The binarization method to convert pre-processed grayscale images.
        # dtype: string that determines the method. Default: "bradley".
        br_method = params.get("br_method", "bradley")
        
        # Binarizer, stroke estimator:
        self.br = Binarization(method=br_method)
        self.se = LineStrokeEstimation()

    
    def export_json(self, strokes, lws, filepath, dtype=float):
        """Given estimated strokes and line widths, saves as a json file
        to be read by the HWR engine.
        """

        trace_list_json = []
        for i, segment in enumerate(strokes):
            lw = lws[i]

            for j, trace in enumerate(segment):
                trace_json = []
                lw_trace = lw[j]
                lw_json = []

                stroke_lw = max(np.max(lw_trace), 1.0)
                stroke_pp = lw_trace / stroke_lw

                for x, y, z in zip(trace[0], trace[1], stroke_pp):
                    trace_json.append({
                        "eventType": "MOVE",
                        "x": dtype(x),
                        "y":dtype(y),
                        "p": dtype(z)
                        })
                    
                if trace_json:
                    trace_json[0]["eventType"]="DOWN"
                    trace_json[-1]["eventType"]="UP"
                    trace_list_json.append({
                        'mId':len(trace_list_json),
                        'mLineWidth': self.lw_scale * stroke_lw,
                        'mPoints':trace_json,
                        "pen": lw_json
                    })
        with open(filepath,'w') as out:
            json.dump(trace_list_json, out)

        return trace_list_json
    

    def estimate_strokes(self, bboxes, max_workers=0):
        """Given an image and its bounding boxes, implements skew correction
        and returns all segmented lines, alongside the perspective
        transform matrix and upscale ratio.
        """

        task_map = self.estimate_segment_strokes
        if len(bboxes) == 0:
            h, w = self.img.shape[0], self.img.shape[1]
            bbox = np.array([
                [0, w],
                [0, 0],
                [h, 0],
                [h, w]
            ])
            task_args = [bbox]
        else:
            task_args = bboxes
        

        # Parallel processing:
        if max_workers > 0:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = list(executor.map(task_map, task_args))
        # Stacked processing:
        else:
            N = len(task_args)
            futures = []
            for i in range(N):
                bbox = task_args[i]
                future = task_map(bbox)
                if future is not None:
                    futures.append(future)
        
        # Process futures:
        all_seg_strokes = [x[0] for x in futures]
        all_seg_lws = [x[1] for x in futures]
        all_seg_pos = [x[2] for x in futures]

        # Sort segmented text lines vertically, then horizontally:
        order = ('y', 'x')
        dtype = [('x', '<i4'), ('y', '<i4')]
        segs_pos = np.array(all_seg_pos, dtype=dtype)
        inds_sort = np.argsort(segs_pos, order=order)

        all_seg_strokes = [all_seg_strokes[i] for i in inds_sort]
        all_seg_lws = [all_seg_lws[i] for i in inds_sort]

        return all_seg_strokes, all_seg_lws
    

    def estimate_segment_strokes(self, bbox, img=None):
        """Segment an image using the bounding box.
        """

        if img is None:
            img = self.img

        bbox_exp = expand_bbox(bbox, per=self.expand_per)
        # Perspective transform:
        img_seg, mat = crop_rotated_bbox(img, bbox_exp, 
                                         mask_value=self.mask_value)
        # Inverse transform matrix (used for strokes):
        mat_inv = np.linalg.pinv(mat)

        # Upscale image:
        img_rs, ratio = upscale_image(img_seg, self.upscale)

        # Apply binarization, stroke estimation:
        img_pp = self.br.preprocess(img_rs)
        img_bin = self.br.binarize(img_pp)
        strokes, lws, img_new = self.se.estimate_strokes(img_bin)

        # Reverse transformation:
        strokes_trans = []
        for (X, Y) in strokes:
            # Downscale
            X_rs, Y_rs = X/ratio, Y/ratio
            # Use perspective transform matrix:
            ones = np.ones(X_rs.size)
            # Flip X, Y, since cv2 and numpy reverses height, width
            pad_stroke = np.stack([Y_rs, X_rs, ones]) # (L, 3) array
            stroke_trans = mat_inv @ pad_stroke
            
            stroke_div = stroke_trans[2]

            # Perspective strokes: First two rows as X, Y, divide each by 3rd row:
            X_trans = stroke_trans[0] / np.maximum(stroke_div, EPS)
            Y_trans = stroke_trans[1] / np.maximum(stroke_div, EPS)
            strokes_trans.append((X_trans, Y_trans))

        # Get center of bounding box (for top-bottom sorting):
        center = get_centerpoint(bbox)

        return strokes_trans, lws, center
    

    def segment_bboxes(self, img, bboxes, upscale=False):
        """Given an image and its bounding boxes, segments the image text
        at each bounding box. To be used to generate data for classification
        models.
        """

        img_segs = []
        # Apply future:
        for bbox in bboxes:
            bbox_exp = expand_bbox(bbox, per=self.expand_per)
            # Perspective transform:
            img_seg, mat = crop_rotated_bbox(img, bbox_exp, 
                                             mask_value=self.mask_value)
            if upscale:
                img_seg, ratio = upscale_image(img_seg, self.upscale)
            
            img_segs.append(img_seg)
        
        return img_segs