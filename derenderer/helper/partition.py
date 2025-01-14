"""Helper functions to partition a binary image using connected binary islands.
To be used in preprocessing and postprocessing methods for the stroke estimation
model.
"""

import numpy as np
import cv2

def get_binarized_islands(img_bin, margin=2):
    """From a binary image, return a list of all cropped binarized islands.
    Get the top left position and the isolated image per island.
    """

    num_islands, img_islands, _, _ = cv2.connectedComponentsWithStats(img_bin)
    # Process each island. Get bounding boxes for the island.
    islands = []
    for n in range(1, num_islands):
        img_island = (img_islands == n).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(img_island)
        # Expanding the bounding rect:
        xs = max(x - margin, 0)
        ys = max(y - margin, 0)
        xf = min(x + w + margin + 1, img_bin.shape[1])
        yf = min(y + h + margin + 1, img_bin.shape[0]) 
        img_crop = img_island[ys:yf, xs:xf] # In reverse from cv2
        islands.append((img_crop, (ys, xs)))
    
    return islands, img_islands, num_islands


def group_islands(islands, target_shape):
    """Given an image shape, groups the islands by attempting to fit the island
    group into an image shape. Adds a new island to the group if the addition
    does not exceed the target width. Assumes these are islands with strokes.
    """

    tgt_h, tgt_w = target_shape[0], target_shape[1]
    # Sort islands by left position:
    islands = sort_islands(islands)

    # Gather intervals:
    intervals = []
    for i, island in enumerate(islands):
        img, (x, y) = island[0], island[1]
        h, w = img.shape[0], img.shape[1]
        intervals.append((y, y+w))
    
    # Get group indices:
    inds_groups = group_intervals(intervals, tgt_w)

    # Compile the groups into the a cropped image (both height, width):
    island_groups = []
    for inds_group in inds_groups:
        sub_islands = [islands[k] for k in inds_group]

        # Find left, right, top, bottom with applied margin:
        pos_left = []
        pos_top = []
        pos_right = []
        pos_bottom = []
        for island in sub_islands:
            img, (x, y) = island[0], island[1]
            h, w = img.shape[0], img.shape[1]
            pos_left.append(y)
            pos_top.append(x)
            pos_right.append(y+w)
            pos_bottom.append(x+h)
        
        left = np.min(pos_left)
        top = np.min(pos_top)
        right = np.max(pos_right)
        bottom = np.max(pos_bottom)

        # Define canvas:
        img_canvas = np.zeros((bottom-top, right-left)).astype(np.uint8)
        for island in sub_islands:
            img, (x, y) = island[0], island[1]
            h, w = img.shape[0], img.shape[1]
            x = x - top
            y = y - left
            img_canvas[x:x+h, y:y+w] += img.astype(np.uint8)

        img_canvas = (img_canvas > 0).astype(np.uint8)
        # Process canvas:
        island_groups.append((img_canvas, (top, left)))
    
    return island_groups


def sort_islands(islands):
    """Sorts islands by leftmost position. Returns the new list of islands.
    """

    x_pos = [island[1][1] for _, island in enumerate(islands)]
    inds_sort = np.argsort(x_pos)

    islands_sort = [islands[n] for n in inds_sort]
    return islands_sort


def resize_and_pad_image(image, new_dims, margin=0, pad_value=0):
    """Given an input image, resizes proportionately and pads the value so 
    that the output image has shape new_dims with pad_value. Returns the
    padded image and the resize ratio, offsets.
    """

    # Get proportional resize height, width
    height, width = image.shape[:2]
    # Apply margin:
    new_height = new_dims[0] - 2*margin
    new_width = new_dims[1] - 2*margin

    ratio_h = new_height / height
    ratio_w = new_width / width

    scale = min(ratio_h, ratio_w)
    # Resize image:
    rs_width = int(np.min((np.rint(scale * width), new_width)))
    rs_height = int(np.min((np.rint(scale * height), new_height)))
    image_rs = cv2.resize(image, (rs_width, rs_height))

    # Keep track of ratio:
    ratio = (rs_width / width + rs_height / height) / 2

    # Pad image evenly around the boundary to new dimensions:
    pad_h = np.max((new_dims[0] - image_rs.shape[0], 0))
    pad_w = np.max((new_dims[1] - image_rs.shape[1], 0))
    pad_h = get_pad_edges(pad_h)
    pad_w = get_pad_edges(pad_w)

    image_pad = cv2.copyMakeBorder(image_rs,
                                   pad_h[0], pad_h[1],
                                   pad_w[0], pad_w[1],
                                   cv2.BORDER_CONSTANT,
                                   value=pad_value)

    # Keep track of padding for translation.
    x_delta = (image_pad.shape[0] - image_rs.shape[0]) / 2
    y_delta = (image_pad.shape[1] - image_rs.shape[1]) / 2
    return image_pad, ratio, (y_delta, x_delta) # In reverse


def inverse_transform(strokes, trans1, ratio, trans2):
    """Given transformations, apply the transform onto the strokes to position
    back to the original image.
    """

    (x1, y1) = trans1
    (x2, y2) = trans2
    inv_strokes = []
    for i, (X, Y) in enumerate(strokes):
        Xinv = (X - x2) / ratio + x1
        Yinv = (Y - y2) / ratio + y1
        inv_strokes.append((Xinv, Yinv))
    
    return inv_strokes


def sort_strokes(strokes):
    """Given ground truth strokes in the form (X, Y), sort the strokes
    using the endpoints. Go left-to-right, top-to-bottom. Orient the strokes
    in this way.
    """

    order = ('x', 'y')
    dtype = [('x', '<i4'), ('y', '<i4')]
    # Collect endpoints:
    Xends = []
    Yends = []
    # Two endpoints of nth stroke are indexed 2*n, 2*n+1
    for i, (X, Y) in enumerate(strokes):
        xs, ys = X[0], Y[0]
        xf, yf = X[-1], Y[-1]
        Xends.append(xs)
        Xends.append(xf)
        Yends.append(ys)
        Yends.append(yf)
    
    # Sort the endpoints:
    values = [(Xends[k], Yends[k]) for k in range(len(Xends))]
    coords = np.array(values, dtype=dtype)
    inds_sort = np.argsort(coords, order=order)

    # Get the stroke order.
    inds_stroke_sort = []
    is_counted = [False for k in range(len(strokes))] # Hash list
    stroke_start_point = []
    for n in inds_sort:
        # Get stroke index:
        rem = n % 2
        if rem == 0:
            ind_stroke = n // 2
        else:
            ind_stroke = (n - 1) // 2

        # Only add the index if not counted:
        if not is_counted[ind_stroke]:
            inds_stroke_sort.append(ind_stroke)
            is_counted[ind_stroke] = True
            # Add the remainder: 0 if start on (xs, ys), 1 if start on (xf, yf)
            stroke_start_point.append(rem)
    
    # Orient the strokes using the endpoints.
    strokes_sorted = []
    for k in range(len(inds_stroke_sort)):
        ind = inds_stroke_sort[k]
        num_s = stroke_start_point[k]

        (X, Y) = strokes[ind]
        # Reverse the orientation if starting at (xf, yf):
        if num_s == 1:
            X = X[::-1]
            Y = Y[::-1]    
        strokes_sorted.append((X, Y))
    
    return strokes_sorted


def clip_strokes(strokes, img_shape):
    """Remove any part of the strokes that are not within the image
    boundaries.
    """

    strokes_clip = []
    for (X, Y) in strokes:
        X_clip = []
        Y_clip = []
        N = min(X.size, Y.size)
        for n in range(N):
            cond_x = X[n] >= 0 and X[n] <= img_shape[0]
            cond_y = Y[n] >= 0 and Y[n] <= img_shape[1]
            if cond_x and cond_y:
                X_clip.append(X[n])
                Y_clip.append(Y[n])
            
        strokes_clip.append((np.array(X_clip), np.array(Y_clip)))
    return strokes_clip
    

# SUPPLEMENTARY FUNCTIONS
def get_pad_edges(n):
    if n % 2 == 0:
        return (n // 2, n // 2)
    else:
        return (n // 2, n // 2 + 1)


def group_intervals(intervals, width):
    """Given a list of intervals (a, b) and a target width, returns groups
    of intervals by attempting to fit the group into the width. Assumes the
    intervals are sorted by left endpoint.
    """

    # First, get all intervals that exceed width.
    N = len(intervals)
    is_exceed = [(b - a) > width for _, (a, b) in enumerate(intervals)]
    # Group based on intervals that contain each other:
    groups = {n: [] for n in range(N)}
    is_contained = {n: False for n in range(N)}

    for n in range(N):
        if not is_exceed[n]:
            continue

        group = []
        a_o, b_o = intervals[n]
        # Gather all intervals contained in this interval:
        for k in range(N):
            a_i, b_i = intervals[k]
            # Do not count self:
            if k == n:
                continue
            # Passed the current interval. Remaining intervals is not contained.
            if a_i > b_o:
                break
            # Is contained: Add to group
            elif a_o <= a_i and b_o >= b_i:
                groups[n].append(k)
                groups[k].append(n)
                is_contained[n] = True
                is_contained[k] = True

    # Only keep the groups with a connection:
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    # Group the remaining intervals. None of their widths can exceed tgt width.
    groups_long = group_connections(groups)

    # Group the remaining intervals
    groups_short = []
    group = []
    w = 0
    left = 0
    for i, (a, b) in enumerate(intervals):

        # Interval is already a part of group. Proceed to next.
        if is_contained[i]:
            continue

        new_w = max(b-left, w)
        # Start new group. Add current group.
        if new_w > width:
            groups_short.append(group)
            group = [i]
            w = b - a
            left = a
        
        # Add to current group:
        else:
            group.append(i)
            w = new_w
        
    # Terminate by adding current group:
    groups_short.append(group)

    # Combine the groups:
    all_groups = groups_long + groups_short
    all_groups = [g for g in all_groups if len(g) > 0]
    return all_groups


def group_connections(edges):
    """Given a dictionary of entries fork: list of connected nodes,
    returns a list of all grouped nodes. Ungrouped nodes appear as singletons.
    """

    groups = []
    ungrouped = []

    is_done = {f: False for f in edges.keys()}

    for f, conns in edges.items():
        if is_done[f]:
            continue
        
        if len(conns) == 0:
            ungrouped.append(f)
        else:
            group = add_to_group([], f, edges)
            for _f in group:
                is_done[_f] = True
            groups.append(group)
        is_done[f] = True

    singletons = [[x] for x in ungrouped]
    return groups + singletons


def add_to_group(group, f, edges):
    """Update the group with new entries until exhausted.
    """

    conns = edges[f]
    for _f in conns:
        if _f not in group:
            group.append(_f)
            group = add_to_group(group, _f, edges)

    return group