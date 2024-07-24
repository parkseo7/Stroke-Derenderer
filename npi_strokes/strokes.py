"""For a set of strokes (and preferably image), processes the strokes to simple bounding box
labels and Bezier curves, as well as retain the point-by-point format. If only strokes
are provided, generates a black-and-white background image.

Ensure that the processed input size is fixed.
"""

# from npi_demo.helper.helper_strokes import 
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from skimage.morphology import binary_dilation
from svgpathtools import polynomial2bezier, CubicBezier, QuadraticBezier
from npi_strokes.helper.helper_strokes import (
    fit_new_shape,
    douglas_peucker,
    cum_lengths,
    cubic_approximation,
    pixel_spline,
    pixelate,
    process_bbox,
    bezier_to_label,
    process_label
)

DFLT_SHAPE = (128, 128) # Character shape
DFLT_CUT_NUM = 1
DFLT_LINE_WIDTH = 6
DFLT_MARGINS = (4, 4)
DFLT_MAX_RESIZE = 2.0
DFLT_ERR_PER_RDP = 0.02
DFLT_BLUR_SIZE = (3, 3)
DFLT_MIN_LENGTH = 6
EPS = 1e-6

class ProcessStrokes:

    def __init__(self, **params):
        self.img_shape = params.get("img_shape", DFLT_SHAPE)
        self.cut_num = params.get("cut_num", DFLT_CUT_NUM)
        self.line_width = params.get("line_width", DFLT_LINE_WIDTH)
        self.margins = params.get("margins", DFLT_MARGINS)
        self.max_resize = params.get("max_resize", DFLT_MAX_RESIZE)

        self.err_per_rdp = params.get("err_per_rdp", DFLT_ERR_PER_RDP)
        self.poly_deg = 3
        self.err_fac = params.get("err_fac", 0.75)
        
        self.blur_size = params.get("blur_size", DFLT_BLUR_SIZE)
        self.min_stroke_length = params.get("min_stroke_length", DFLT_MIN_LENGTH)

    
    def _cut_strokes(self, point_strokes):
        # Cut and filter strokes:
        cut = self.cut_num
        cut_strokes = []
        for (X, Y) in point_strokes:
            N = min(X.size, Y.size)
            if N <= 2 + cut: # Needs at least 2 points remaining
                continue
            
            if cut > 0:
                X0 = X[cut:-cut]
                Y0 = Y[cut:-cut]
            else:
                X0 = X
                Y0 = Y
            cut_strokes.append((X0, Y0))

        return cut_strokes
    

    def generate_datapoint(self, inter_strokes):
        """Goes from POT point strokes to data points. Apply to
        interpolated strokes.
        """

        rs_paths, img = self.process_strokes(inter_strokes)
        img = self.augment_image(img)

        stroke_labels = []
        for p in rs_paths:
            poly = p.poly()
            if len(poly) == 1:
                p = CubicBezier(p[0], 2*p[0]/3+p[1]/3, p[0]/3+2*p[1]/3, p[1])
            elif len(poly) == 2:
                p = CubicBezier(p[0], (p[0]+p[1])/2, (p[1]+p[2])/2, p[2])

            if p.length() <= self.min_stroke_length:
                continue 

            # Get bounding box in format (xc, yc, h, w)
            bbox_c = process_bbox(p.bbox(), img.shape)

            # Configure bezier points to predict:
            tgt = bezier_to_label(p.bpoints())
            labels = process_label(bbox_c, tgt)

            stroke_labels.append((bbox_c, labels))
        
        return img, stroke_labels


    def interpolate_strokes(self, point_strokes):
        """Given point strokes, apply a quadratic or linear interpolation,
        to generate more points.
        """

        inter_strokes = []
        cut_strokes = self._cut_strokes(point_strokes)
        for (X, Y) in cut_strokes:
            Xint, Yint = pixel_spline(X, Y, self.err_fac)
            inter_strokes.append((Xint, Yint))

        return inter_strokes


    def process_strokes(self, point_strokes):
        """Processes the point strokes by applying RDP approximation to reduce
        the number of points, followed by Bezier curve approximation. Outputs 
        a list of Bezier curves approximating the list of point strokes.
        """
        
        # Translate and proportionally rescale set of strokes to fit in image shape
        # Only do this step for strokes not paired with images.
        bboxes = []
        for i, (X, Y) in enumerate(point_strokes):
            # Get bounding box corners:
            xmin, xmax = np.min(X), np.max(X)
            ymin, ymax = np.min(Y), np.max(Y)

            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
        
        # Get bounding box that wraps all bounding boxes:
        Xmin = np.min([bbox[0] for bbox in bboxes])
        Xmax = np.max([bbox[2] for bbox in bboxes])
        Ymin = np.min([bbox[1] for bbox in bboxes])
        Ymax = np.max([bbox[3] for bbox in bboxes])
        
        shift_strokes = []
        for i, (X, Y) in enumerate(point_strokes):
            shift_strokes.append((X-Xmin, Y-Ymin))

        shift_bboxes = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            bbox0 = [x1-Xmin, y1-Ymin, x2-Xmin, y2-Ymin]
            shift_bboxes.append(bbox0)
        
        # Rescale and pad the parent bounding box to fit in image
        Bbox = (Xmax - Xmin, Ymax - Ymin)
        ratio, offset = fit_new_shape(Bbox, 
                                      self.img_shape, 
                                      self.margins, 
                                      self.max_resize)
        # Scale down all strokes and apply offset:
        rs_strokes = []
        for i, (X, Y) in enumerate(shift_strokes):
            X0 = X * ratio + offset[0]
            Y0 = Y * ratio + offset[1]
            rs_strokes.append((X0, Y0))
        
        rs_bboxes = []
        for i, bbox in enumerate(shift_bboxes):
            x1, y1, x2, y2 = bbox
            
            bbox0 = [
                x1 * ratio + offset[0],
                y1 * ratio + offset[1],
                x2 * ratio + offset[0],
                y2 * ratio + offset[1]
            ]
            rs_bboxes.append(bbox0)
        
        # From rescaled strokes, get Bezier curves:
        rs_paths = []
        img_canvas = np.zeros(self.img_shape)
        for (X, Y) in rs_strokes:
            beziers = self.cubic_fitting(X, Y)
            for bezier in beziers:
                coeffs = bezier[2]
                p = polynomial2bezier(coeffs[0] + 1j*coeffs[1])
                if len(p) == 4:
                    path = CubicBezier(p[0], p[1], p[2], p[3])
                elif len(p) == 3:
                    path = QuadraticBezier(p[0], p[1], p[2])
                
                rs_paths.append(path)
            
            # From rescaled strokes, get pixel path and rasterize:
            Xpix, Ypix = pixelate(X, Y)
            Xpix = np.clip(Xpix, 0, self.img_shape[0])
            Ypix = np.clip(Ypix, 0, self.img_shape[1])
            img_canvas[Xpix, Ypix] = 1
        
        # Needs canvas image to be transposed
        return rs_paths, img_canvas.T
    

    def augment_image(self, img_canvas):
        """Given a skeleton binarized image of the strokes, adds line
        thickness and applies some median blur.
        """

        img_new = binary_dilation(img_canvas)
        img_bw = np.clip(255 * img_new, 0, 255).astype(np.uint8)
        img_blur = cv2.blur(img_bw, self.blur_size)

        return img_blur


    def cubic_fitting(self, X, Y):
        """For a single stroke, apply a series of cubic polynomials, 
        for Bezier curve approximation. Try using the Douglas-Peucker
        algorithm but using cubic fitting?
        """

        err = self.err_per_rdp * np.min(self.img_shape)
        # Xc, Yc = douglas_peucker(X, Y, err)

        # Cubic polynomial fitting on reduced curves:
        N = X.size
        # Null case:
        if N <= 1:
            return []
        
        beziers = []
        i_f = 1
        i_s = 0
        t_arr = cum_lengths(X, Y)
        prev_coeffs = None

        while i_f < N+2:
            Xsub = X[i_s:i_f]
            Ysub = Y[i_s:i_f]
            t_sub = t_arr[i_s:i_f]
            cond, coeffs = cubic_approximation(t_sub, Xsub, Ysub, err, 
                                               deg=self.poly_deg)
            # Within distance: Try next one:
            if cond:
                prev_coeffs = coeffs
            # Not within distance. Store bezier points:
            else:
                bezier = (Xsub[:-1], Ysub[:-1], prev_coeffs)
                beziers.append(bezier)
                i_s = i_f-2
            
            i_f += 1
        
        beziers.append((Xsub, Ysub, coeffs))
        return beziers


