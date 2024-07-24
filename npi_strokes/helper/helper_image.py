"""All image transformation functions using cv2.
"""
 
import numpy as np
import cv2
 
EPS = 1e-6
 
# CODE FROM SUB MODULE TEXT RECOGNITION
def order_points(pts):
    """
    Order points in the following order:
    top-left, top-right, bottom-right, bottom-left.
 
    :param pts: Array of four points
    :return: Ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
 
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    return rect
 
 
def crop_rotated_bbox(image, points, mask_value=(1, 255, 1)):
    """
    Crop a rotated bounding box from an image using the four corner points.
 
    :param image: Source image
    :param points: Four corner points of the bounding box
    :return: Cropped image
    """
    # Order the points
    rect = order_points(points)
   
    # Calculate the width and height of the new image
    width = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    ))
    height = int(max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    ))
 
    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
 
    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
   
    # Perform the perspective transformation
    warped = cv2.warpPerspective(image, M, (width, height),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=mask_value) # Green
    bg_color = np.median(warped, axis=[0,1])
 
    # Rotate again, fill border with median color:
    warped = cv2.warpPerspective(image, M, (width, height),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=bg_color)
   
    return warped, M
   
 
def expand_bbox(bbox, per=0.25):
    """Expands the bounding box in a way that is consistent with the bounding
    box tilt. Also re-orders the bbox.
    """
 
    # Identify the vertices:
    X = bbox[:,0]
    Y = bbox[:,1]
 
    # Two leftmost, two rightmost
    inds_y = np.argsort(Y)
 
    # Sort leftmost, rightmost:
    i1, i2 = inds_y[0], inds_y[1]
    i3, i4 = inds_y[2], inds_y[3]
 
    # Compare heights and get points:
    x1, x2 = X[i1], X[i2] # leftmost, second leftmost
    if x1 < x2:
        p_tl = np.array([x1, Y[i1]])
        p_bl = np.array([x2, Y[i2]])
    else:
        p_tl = np.array([x2, Y[i2]])
        p_bl = np.array([x1, Y[i1]])
 
    x3, x4 = X[i3], X[i4] # rightmost, second rightmost
    if x3 < x4:
        p_tr = np.array([x3, Y[i3]])
        p_br = np.array([x4, Y[i4]])
    else:
        p_tr = np.array([x4, Y[i4]])
        p_br = np.array([x3, Y[i3]])
 
    # Extend outward in the direction of right, top vectors:
    v_right = p_tr - p_tl
    v_top = p_tr - p_br
 
    norm_h = np.linalg.norm(v_top)
    norm_w = np.linalg.norm(v_right)
 
    v_right = v_right / (norm_w + EPS)
    v_top = v_top / (norm_h + EPS)
 
    # Equal margin:
    extend = int(per * min(norm_h, norm_w))
    p_tl = p_tl + extend * (v_top - v_right)
    p_tr = p_tr + extend * (v_top + v_right)
    p_bl = p_bl + extend * (-v_top - v_right)
    p_br = p_br + extend * (-v_top + v_right)
 
    bbox_exp = np.stack((p_tr, p_tl, p_bl, p_br))
    # Preserve order:
    bbox_exp = order_points(bbox_exp)
 
    return bbox_exp
 
 
def upscale_image(image, min_short_dim):
   """Given an image, resize it so that the shorter length is at lesat
   min_short_dim.
   """
 
   length = min(image.shape[0], image.shape[1])
   
   # No rescaling needed
   if length >= min_short_dim:
      return image, 1.0
   
   else:
      ratio = min_short_dim / length
      new_h = int(ratio * image.shape[0])
      new_w = int(ratio * image.shape[1])
      # print(new_h, new_w)
      new_image = cv2.resize(image, (new_w, new_h))
      return new_image, ratio
   
 
def normalize_image(image):
    """Normalizes the image by converting to float and centering by mean, std.
    """
 
    normalized_image = cv2.normalize(image, None, 0, 255,
                                     norm_type=cv2.NORM_MINMAX)
    return normalized_image
 
 
def get_centerpoint(bbox):
    """Get the center point of the bounding box.
    """
 
    X = bbox[:,0] # Horizontal
    Y = bbox[:,1] # Vertical
    return (np.mean(X), np.mean(Y))