"""Main algorithms for skeletonization and stroke estimation. Only dependencies
are Numpy and cv2.
"""

import numpy as np
import cv2


def get_binarized_islands(img_bin, margin=1, min_size=1, hole_per=0.04, inner_per=0.25):
    """From a binary image, return a list of all cropped binarized islands.
    Omit the island if its bounding box does not intersect with the inner box, 
    or if the size of the island is too small.
    """

    # Get inner bbox (reduce by inner percentage):
    bbox_inner = inner_bbox(img_bin.shape, per=inner_per)
    # New binarized image (for mdist processing):
    img_new = np.zeros(img_bin.shape, dtype=np.uint8)

    num_islands, img_islands, _ , _ = cv2.connectedComponentsWithStats(img_bin)
    islands = []
    for n in range(1, num_islands):
        img_island = (img_islands == n).astype(np.uint8)

        # Omit small-sized islands:
        island_size = np.count_nonzero(img_island)
        if island_size < min_size:
            continue

        # Crop out:
        x, y, w, h = cv2.boundingRect(img_island)

        x1 = np.max((x-margin, 0))
        y1 = np.max((y-margin, 0))
        x2 = np.min((x+w+margin, img_bin.shape[1]))
        y2 = np.min((y+h+margin, img_bin.shape[0]))
        
        bbox = [(y1, y2), (x1, x2)]
        
        # Omit any island that doesn't intersect with inner image bbox:
        if not is_intersect(bbox, bbox_inner):
            continue
        
        img_crop = img_island[y1:y2, x1:x2]
        # Process the island by removing small holes:
        img_crop = process_island(img_crop, hole_per=hole_per)
        
        # Add cropped image to new binarized image:
        img_new[y1:y2, x1:x2] += img_crop

        # Add island:
        island = {
            "img_crop": img_crop,
            "pos": (x1, y1),
            "bbox": bbox
        }

        islands.append(island)
    
    # Convert new binary image filetype to binary:
    img_new = (img_new > 0).astype(np.uint8)

    # Calculate mdists:
    mdists = cv2.distanceTransform(img_new, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    for island in islands:
        bbox = island["bbox"]
        img_crop = island["img_crop"]
        x1, x2 = bbox[1]
        y1, y2 = bbox[0]
        mdist_crop = mdists[y1:y2, x1:x2]
        island["img_mdist"] = mdist_crop * img_crop

    return islands, img_new


def neighborhood_mat(img):
    """Given an image, returns the submatrices that represent the neighbors
    at each point (x, y), in a clock-wise order.
    """

    # (x, y) -> (x_move, y_move)
    P2 = np.zeros(img.shape, dtype=np.uint8)
    P2[1:,:] = img[:-1,:] # Up
    P4 = np.zeros(img.shape, dtype=np.uint8)
    P4[:,:-1] = img[:,1:] # Right
    P6 = np.zeros(img.shape, dtype=np.uint8)
    P6[:-1,:] = img[1:,:] # Down
    P8 = np.zeros(img.shape, dtype=np.uint8)
    P8[:,1:] = img[:, :-1] # Left

    # Diagonals
    P3 = np.zeros(img.shape, dtype=np.uint8)
    P3[1:,:-1] = img[:-1, 1:] # Up right
    P5 = np.zeros(img.shape, dtype=np.uint8)
    P5[:-1,:-1] = img[1:,1:] # Down right
    P7 = np.zeros(img.shape, dtype=np.uint8)
    P7[:-1,1:] = img[1:,:-1] # Down left
    P9 = np.zeros(img.shape, dtype=np.uint8)
    P9[1:,1:] = img[:-1,:-1] # Up left

    N = [P2, P3, P4, P5, P6, P7, P8, P9]
    # Sum of all neighborhoods:
    S = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
    # Number of circular transitions:
    T = np.zeros(img.shape)
    for i in range(len(N)):
        P_now = N[i]
        P_next = N[(i+1) % len(N)]
        # 0 -> 1 transition
        T += (1 - P_now) * P_next

    mid = np.zeros(img.shape, dtype=np.bool_)
    mid[1:-1, 1:-1] = True
    return N, S, T, mid


def zhangSuen(img):
    """The Zhang-Suen Thinning Algorithm. For every iteration survived,
    add 1 to the medial axis distance. The surviving pixels is the skeleton.
    """
    
    img_thin = img.copy()
    is_change1 = True
    is_change2 = True

    while is_change1 or is_change2:
        mats, S, T, mid = neighborhood_mat(img_thin)
        P2, P3, P4, P5, P6, P7, P8, P9 = mats

        # Convert to 0 all points that survive all 4 conditions:
        cond0 = (img_thin == 1) * mid
        cond1 = (S >= 2) * (S <= 6)
        cond2 = (T == 1)
        cond3 = ((P2 * P4 * P6) == 0)
        cond4 = ((P4 * P6 * P8) == 0)
        
        cond = cond0 * cond1 * cond2 * cond3 * cond4
        is_change1 = np.any(cond)
        img_thin = (img_thin * (1 - cond)).astype(np.uint8)
        
        # Repeat, with slightly different conditions (cond3, cond4):
        mats, S, T, mid = neighborhood_mat(img_thin)
        P2, P3, P4, P5, P6, P7, P8, P9 = mats
        cond0 = (img_thin == 1) * mid
        cond1 = (S >= 2) * (S <= 6)
        cond2 = (T == 1)
        cond3 = ((P2 * P4 * P8) == 0)
        cond4 = ((P2 * P6 * P8) == 0)
        
        cond = cond0 * cond1 * cond2 * cond3 * cond4
        is_change2 = np.any(cond)
        img_thin = (img_thin * (1 - cond)).astype(np.uint8)

    return img_thin


# SUPPLEMENTARY FUNCTIONS:
def process_island(img_crop, hole_per=0.04):
    """Given a binarized island, fills in small holes.
    """

    # Get restructure kernel:
    H, W = img_crop.shape[0], img_crop.shape[1]
    length = min(H, W)
    min_size = int(length * hole_per)+1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(min_size, min_size))
    res = cv2.morphologyEx(1 - img_crop, cv2.MORPH_OPEN, kernel)
    img_crop = 1 - res

    # Cut out boundary:
    img_crop[:,0] = 0
    img_crop[0,:] = 0
    img_crop[:,-1] = 0
    img_crop[-1,:] = 0

    return img_crop


def inner_bbox(img_shape, per=0.25):
    """Given an image shape, returns the inner bounding box by reducing
    the smaller dimension by the given percentage.
    """

    H, W = img_shape[0], img_shape[1]
    margin = min(per*H, per*W)
    bbox = [(margin, H - margin), (margin, W - margin)]
    return bbox


def is_intersect(bbox1, bbox2):
    """Given two bounding boxes with 4 corners, return True if intersect
    and False otherwise.
    """

    t1, b1 = bbox1[0]
    l1, r1 = bbox1[1]
    t2, b2 = bbox2[0]
    l2, r2 = bbox2[1]

    # Get conditions:
    cond1 = (t1 <= t2) and (t2 <= b1)
    cond2 = (t2 <= t1) and (t1 <= b2)
    cond3 = (l1 <= l2) and (l2 <= r1)
    cond4 = (l2 <= l1) and (l1 <= r2)

    return (cond1 or cond2) and (cond3 or cond4)