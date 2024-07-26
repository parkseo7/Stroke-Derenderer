import numpy as np
EPS = 1e-6

def skeleton_to_edges(img_sk):
    """Given a binary skeletonized image with 1-pixel branches, converts it 
    into a bi-directional graph. Each node will be a single pixel.
    """

    h, w = img_sk.shape[0], img_sk.shape[1]

    # Get the graph directly:
    points_x, points_y = np.where(img_sk > 0)
    nodes_child = np.ravel_multi_index((points_x, points_y), img_sk.shape)
    nodes_child = nodes_child.astype(np.int32) # For .json compatibility

    edges = {}
    edge_types = {}

    for i in range(nodes_child.size):
        x_c, y_c = points_x[i], points_y[i]
        child = nodes_child[i]

        new_points_x = []
        new_points_y = []

        is_upbound = (x_c >= 1)
        is_downbound = (x_c+1 < h)
        is_leftbound = (y_c >= 1)
        is_rightbound = (y_c+1 < w)

        is_up = False
        is_down = False
        is_left = False
        is_right = False

        # First check direct neighbourhood.
        if is_upbound:
            is_up = img_sk[x_c-1, y_c]
            if is_up:
                new_points_x.append(x_c-1)
                new_points_y.append(y_c)
        if is_downbound:
            is_down = img_sk[x_c+1, y_c]
            if is_down:
                new_points_x.append(x_c+1)
                new_points_y.append(y_c)
        if is_leftbound:
            is_left = img_sk[x_c, y_c-1]
            if is_left:
                new_points_x.append(x_c)
                new_points_y.append(y_c-1)
        if is_rightbound:
            is_right = img_sk[x_c, y_c+1]
            if is_right:
                new_points_x.append(x_c)
                new_points_y.append(y_c+1)
        
        # Check diagonal. Only add diagonal if no edges nearby:
        if is_upbound and is_leftbound and not is_up and not is_left:
            is_upleft = img_sk[x_c-1, y_c-1]
            if is_upleft:
                new_points_x.append(x_c-1)
                new_points_y.append(y_c-1)
        if is_downbound and is_leftbound and not is_down and not is_left:
            is_downleft = img_sk[x_c+1, y_c-1] 
            if is_downleft:
                new_points_x.append(x_c+1)
                new_points_y.append(y_c-1)
        if is_upbound and is_rightbound and not is_up and not is_right:
            is_upright = img_sk[x_c-1, y_c+1]
            if is_upright:
                new_points_x.append(x_c-1)
                new_points_y.append(y_c+1)
        if is_downbound and is_rightbound and not is_down and not is_right:
            is_downright = img_sk[x_c+1, y_c+1] 
            if is_downright:
                new_points_x.append(x_c+1)
                new_points_y.append(y_c+1)

        if len(new_points_x) == 0:
            edges[child] = np.array([], dtype=np.int32)
        
        # Add all nodes:
        else:
            new_points = (new_points_x, new_points_y)
            new_nodes = np.ravel_multi_index(new_points, img_sk.shape)
            new_nodes = new_nodes.astype(np.int32) # Convert to int32 for C++

            if new_nodes.size == 1:
                edge_types[child] = 1
            # Fork node (more than 2 edges):
            elif new_nodes.size > 2:
                edge_types[child] = 3
            # Transition node (exactly 2 edges):
            elif new_nodes.size == 2:
                edge_types[child] = 2
            edges[child] = new_nodes

    return edges, edge_types


def nodes_to_curves(nodes, explored_nodes, edges, init_node=None, init_prev=None):
    """A sub-function for finding all pixel curves that start at the node list.
    To be applied to terminals, forks, then transition nodes in that order.
    Returns a list of continuous curves while updating the explored 
    dictionary.

    Args:
     - nodes (list of ints): Initial nodes to start at.
     - explored_nodes(dict int -> bool): Hash that tells which nodes have been explored.
     - edges (dict int -> list of ints): Hash that shows the connection between source node 
     to all target nodes.
     - init_node: An initial node point to start at. To be used if the initial node is a fork.
     - init_prev: Previous node point, to be used if the initial node is a transition point.
    """

    curves = []
    for x_s in nodes:
        
        # Skip if this terminal has already been explored:
        if explored_nodes[x_s]:
            continue
        
        if init_prev is None:
            x_prev = x_s
        else:
            x_prev = init_prev
        
        x_now = x_s
        if init_node is None:
            curve = []
        else:
            curve = [init_node]

        # Move along curve until we hit a fork or terminal:
        done = False
        while not done:
            next_nodes = [x for x in edges[x_now] if x != x_prev]
            # You've hit a fork. STOP.
            if len(next_nodes) > 1:
                done = True

            # You've hit a terminal point. STOP.
            elif len(next_nodes) == 0:
                done = True
                explored_nodes[x_now] = True
            
            # You're at a transition node. Keep going.
            elif len(next_nodes) == 1:
                explored_nodes[x_now] = True
            
            # Add node:
            curve.append(x_now)

            # Only update if not done:
            if not done:
                x_prev = x_now
                # Maybe choose the most parallel node?
                x_now = next_nodes[0]
                # Check if the next x_now node is already explored:
                done = explored_nodes[x_now]
        
        # Store curve (list of nodes) for later:
        curves.append(curve)

    return curves


def edges_to_graph(edges, edge_types):
    """Given a graph of edges and edge types, categorize all nodes into
    curved connections, from terminal to fork, terminal to terminal, and fork
    to fork. We need the shape of the skeleton image for ravel, unravel.
    Each segment is a 3-tuple consisting of x-pixel, y-pixel, node type.
    """

    # Nodes already explored (node -> bool)
    explored = {k: False for k in edges.keys()}
    curves = []

    terminals = [k for k, v in edge_types.items() if v == 1]
    transitions = [k for k, v in edge_types.items() if v == 2]
    forks = [k for k, v in edge_types.items() if v == 3]

    # Get all segments starting at terminals:
    curves_term = nodes_to_curves(terminals, explored, edges)
    curves += curves_term

    # Get line segments starting at every fork:
    for x_f in forks:
        # Get all unexplored adjacent nodes. Each of these is a line segment.
        init_nodes = [x for x in edges[x_f] if not explored[x]]
        curves_fork = nodes_to_curves(init_nodes, explored, edges,
                                      init_node=x_f, init_prev=x_f)
        curves += curves_fork
    
    # Get line segments starting at a random transition point:
    transitions_remain = [k for k in transitions if not explored[k]]
    done = (len(transitions_remain) == 0)
    while not done:
        # Choose an unexplored neighbor as the initial previous node:
        k0 = transitions_remain[0]
        init_prevs = [l for l in edges[k0]]
        init_prev = init_prevs[0]
        curves_trans = nodes_to_curves([k0], explored, edges,
                                        init_prev=init_prev)
        if len(curves_trans) > 0:
            curves.append(curves_trans[0])
        
        transitions_remain = [l for l in transitions if not explored[l]]
        done = (len(transitions_remain) == 0)
    
    # For each stroke segment, get endpoint types and organize as dict:
    return curves


def avg_stroke_width(mdists):
    """Obtain the average stroke width of an island, given a medial distance. 
    Use 0.5 * barw as the minimum branch length. Also returns the image boundary.
    """

    img_boundary = (mdists == 1)
    N_p = np.count_nonzero(mdists > 0)
    l_c = np.count_nonzero(img_boundary)
    sqrt_fac = l_c**2 / 16 - N_p
    if sqrt_fac < 0:
        avg_width = l_c / 4
    else:
        avg_width = l_c / 4 - np.sqrt(sqrt_fac)

    return avg_width, img_boundary


def index_curves(curves, forks):
    """Given an initial set of curves and fork nodes, return indexing
    dictionaries:
    - fork -> list of (index, node)
    - (index, node) -> fork
    - index -> fork-to-fork
    """

    is_ftf = [False for i in range(len(curves))]
    node_to_curve = {}
    curve_to_node = {}
    for i, curve in enumerate(curves):
        Ps = curve[0]
        Pf = curve[-1]
        if Ps in forks and Pf in forks:
            is_ftf[i] = True
        
        # Add to curve pointer:
        if Ps in node_to_curve:
            node_to_curve[Ps].append((i, True))
        else:
            node_to_curve[Ps] = [(i, True)]
        
        if Pf in node_to_curve:
            node_to_curve[Pf].append((i, False))
        else:
            node_to_curve[Pf] = [(i, False)]

        # Add to fork pointer:
        curve_to_node[(i, True)] = Ps
        curve_to_node[(i, False)] = Pf

    return node_to_curve, curve_to_node, is_ftf

def group_forks(edges):
    """Given a dictionary of entries fork: list of connected forks,
    returns a list of all grouped forks. Ungrouped forks appear as singletons.
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


def find_cutoff_index(node, curve, radius, img_shape):
    """Finds the closest point along the curve that is outside of the
    given radius. Here, curve is an array.
    """

    X, Y = np.unravel_index(curve, img_shape)
    px, py = np.unravel_index(node, img_shape)

    inds_out = np.where(((X-px)**2 + (Y-py)**2) >= radius**2)[0]
    
    if inds_out.size == 0:
        return curve.size
    
    else:
        return np.min(inds_out)
    

def nearest_straight_index(X, Y, thr_err):
    """Given a curve and corresponding error, obtain the nearest index
    such that a straight line is within the error margin.

    Args:
    - X, Y: Coordinate (float) arrays of the same size.
    - thr_err: Error threshold for cutting off the line index.
    """

    # Take the minimum length between the two arrays.
    N = min(X.size, Y.size)
    X, Y = X[:N], Y[:N]
    
    # Null condition:
    if X.size <= 2:
        return X.size-1
    
    xs, ys = X[0], Y[0]
    done = False
    # Keep moving until we hit endpoint or within err:
    ind = 2
    while not done:
        # If we have hit endpoint, we don't need to check:
        if ind >= X.size-1:
            done = True
        xf, yf = X[ind], Y[ind]

        # Assess straight line:
        dx = xf - xs
        dy = yf - ys
        nx, ny = -dy, dx
        C = -xs*nx - ys*ny
        norm = max(np.sqrt(nx**2 + ny**2), EPS)
        dists = np.abs(X[:ind]*nx + Y[:ind]*ny + C) / norm
        err_line = np.max(dists)

        # Cut off and return the current line if the error is above threshold:
        if err_line > thr_err:
            done = True

        ind += 1
    return ind


def median_node(nodes, img_shape):
    """Obtain the median node.
    """

    X, Y = np.unravel_index(nodes, img_shape)
    Xm = int(np.round(np.mean(X)))
    Ym = int(np.round(np.mean(Y)))
    
    # Ravel again:
    node_m = np.ravel_multi_index((Xm, Ym), img_shape)
    return node_m


def remove_repeating(X, Y, radius=0):
    """Given a 2d stroke array, filters out consecutive repeating points.
    Points are considered repeating if Euclidean distance is within radius.
    """

    diff = np.diff(X)**2 + np.diff(Y)**2
    inds = 1 + np.where(diff > radius**2)[0]
    Xc = np.concatenate(([X[0]], X[inds]))
    Yc = np.concatenate(([Y[0]], Y[inds]))

    # Recursively apply repeating:
    if Xc.size < X.size:
        Xc, Yc = remove_repeating(Xc, Yc, radius=radius)

    return Xc, Yc


def normalize_vector(xp, yp):
    """Normalizes the vector to the unit vector.
    """
    norm = max(np.sqrt(xp**2 + yp**2), EPS)
    xp, yp = xp / norm, yp / norm
    return xp, yp


def sort_ccwise(points, centerpoint=None):
    """Given an array of points and a centerpoint, returns
    a sorted array of points ordered in a counterclockwise
    fashion from the centerpoint, starting at angle = 0.
    """

    if centerpoint is None:
        centerpoint = np.mean(points, axis=0)

    vectors = points - centerpoint
    # Apply complex argument (for images y-axis is width)
    angles = np.angle(vectors[:,1] + vectors[:,0]*1j)

    # Mod to [0, 2*pi]:
    angles0 = angles * (angles >= 0) + \
        (angles < 0) * (2*np.pi + angles)

    # Sort angles:
    inds_sort = np.argsort(angles0)
    return inds_sort


# SUPPLEMENTARY FUNCTIONS
def add_to_group(group, f, edges):
    """Update the group with new entries until exhausted.
    """

    conns = edges[f]
    for _f in conns:
        if _f not in group:
            group.append(_f)
            group = add_to_group(group, _f, edges)

    return group