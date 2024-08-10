"""Implement stroke estimation on a binary picture. Use only cv2 and
numpy arrays to make it C++ convertible.
"""

import cv2
import numpy as np
# Scipy dependencies. Need to sort out C++ equivalency.
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splrep, splev

from npi_demo.helper.helper_sk import (
    get_binarized_islands,
    zhangSuen
)

from npi_demo.helper.helper_stroke import (
    skeleton_to_edges,
    edges_to_graph,
    avg_stroke_width,
    index_curves,
    group_forks,
    find_cutoff_index,
    nearest_straight_index,
    median_node,
    remove_repeating,
    normalize_vector,
    sort_ccwise,
    douglas_peucker,
    cum_lengths,
    sample_points,
    get_island_color
)

EPS = 1e-6

class LineStrokeEstimation:
    """Segment the binarized image into binarized islands, and obtain
    the initial skeleton and medial distances per island.
    """

    def __init__(self, **params):
        self.hole_div = params.get("hole_div", 24) # Hole is nth of image size
        self.max_size = params.get("max_size", 0.04)
        self.per_min_size = params.get("per_min_size", 0.02)
        self.inner_bbox_per = params.get("inner_bbox_per", 0.15)
        self.crop_margin = params.get("crop_margin", 1)


    def estimate_strokes(self, img_bin, img_color):
        """Given a binarized image, estimate the strokes. Returns the strokes,
        corresponding line widths, and a new binarized image where the 
        strokes were estimated.
        """

        islands, img_new = self.get_islands(img_bin)
        strokes, lws, colors = self.get_strokes(islands, img_color)
        inds_sort, oriented_strokes = self.orient_strokes(strokes, img_bin.shape)

        strokes = [oriented_strokes[i] for i in inds_sort]
        lws = [lws[i] for i in inds_sort]
        colors = [colors[i] for i in inds_sort]

        return strokes, lws, colors


    def get_islands(self, img_bin):
        """Get the binarized islands to implement stroke estimation.
        """

        H, W = img_bin.shape[0], img_bin.shape[1]
        length = min(H, W)
        min_size = int(length * self.per_min_size) ** 2
        islands, img_new = get_binarized_islands(img_bin,
                                                 margin=self.crop_margin,
                                                 min_size=min_size,
                                                 hole_per=1/self.hole_div,
                                                 inner_per=self.inner_bbox_per
                                                 )

        # Apply initial skeletonization to each cropped island:
        # Potential parallelization?
        for island in islands:
            # Fix the cropped island: Remove small holes
            img_sk = zhangSuen(island["img_crop"])
            island["img_sk"] = img_sk

        return islands, img_new
    

    def get_strokes(self, islands, img_color):
        """Given binarized islands, applies stroke estimation to each island.
        Can be parallelized.
        """

        all_strokes = []
        all_lws = []
        all_colors = []

        for island in islands:
            img_sk = island["img_sk"]
            img_mdist = island["img_mdist"]
            se = StrokeEstimation(img_sk, img_mdist)
            strokes, lws = se.estimate_strokes()

            # Translate and order the strokes:
            pos = island["pos"]
            translated_strokes = []
            for (X, Y) in strokes:
                translated_strokes.append((pos[1] + X, pos[0] + Y))
            
            # Get island color:
            color = get_island_color(img_mdist, pos, img_color)
            all_strokes += translated_strokes
            all_lws += lws
            all_colors += [color for i in range(len(all_strokes))]

        return all_strokes, all_lws, all_colors
        
    
    def orient_strokes(self, strokes, img_shape):
        """Orders the strokes using the image shape.
        """

        axis_long = np.argmax(img_shape)
        oriented_strokes = []
        for (X, Y) in strokes:
            xs, ys = X[0], Y[0]
            xf, yf = X[-1], Y[-1]
            
            flip = False
            if axis_long == 0:
                if xf < xs:
                    flip = True
            else:
                if yf < ys:
                    flip = True
            if flip:
                X = np.flip(X)
                Y = np.flip(Y)
            oriented_strokes.append((X, Y))
                    
        # Order the curves from left-to-right:
        dtype = [('x', '<i4'), ('y', '<i4')]
        if axis_long == 0:
            order = ('x', 'y')
        else:
            order = ('y', 'x')
        points = [(X[0], Y[0]) for (X, Y) in oriented_strokes]
        points = np.array(points, dtype=dtype)
        inds_sort = np.argsort(points, order=order)

        return inds_sort, oriented_strokes


class StrokeEstimation:
    """Implements stroke estimation per binarized island. Requires
    the initial skeleton and medial distance.
    """

    def __init__(self, img_sk, img_mdist, **params):

        # Store initial values:
        self.img_sk = img_sk
        self.img_mdists = img_mdist

        # Attributes:
        self.group_radius = params.get("group_radius", 1.0) # Larger = more zone groups
        self.nearby_fac = params.get("nearby_factor", 8) # Keep large for better tangent accuracy
        self.cut_fac = params.get("cut_factor", 4) # Keep non-ftf cut curves within this length
        self.sample_per_pixel = params.get("sample_per_pixel", 0.2) # Number of interpolated samples per pixel
        self.quint_angle_thr = params.get("quint_angle_thr", np.cos(np.pi/4))

        self.rdp_fac = params.get("rdp_fac", 4) # Divide avg line width by fac to get RDP error
        self.sample_fac = params.get("sample_fac", 8) # Divide avg line width by fac to get sample distance.

        # Initialize graph:
        lw, img_boundary = avg_stroke_width(img_mdist)
        edges, edge_types = skeleton_to_edges(img_sk)
        forks = [f for f, v in edges.items() if len(v) >= 3]
        init_curves = edges_to_graph(edges, edge_types)
        node_to_curve, curve_to_node, is_ftf = index_curves(init_curves, forks)

        # Store values:
        self.lw = lw
        self.edges = edges
        self.edge_types = edge_types
        self.init_curves = [np.array(c) for c in init_curves]
        self.forks = forks # Junction zone areas

        self.curve_to_node = curve_to_node
        self.node_to_curve = node_to_curve
        self.is_ftf = is_ftf # Is a fork-to-fork curve

        # Values to process:
        self.fork_groups = None
        self.is_conn_ftf = None
        self.zones = None


    @property
    def img_shape(self):
        return self.img_sk.shape
    

    def get_mdist(self, P):
        x, y = np.unravel_index(P, self.img_shape)
        D = self.img_mdists[x, y]
        return D
    

    # KEY METHODS
    def estimate_strokes(self, debug=True):
        """Estimates the strokes using the initial skeleton.
        """

        # Identify ambiguous zones and cut curves:
        self._group_forks()
        self.get_cut_curves()
        self._organize_zones()

        zone_conns = {}
        for zone in self.zones:
            conns = self.resolve_zone(zone)
            zone_conns.update(conns)

        # Get all nodes that need to be traversed:
        is_traversed = self.init_traverse_nodes()
        all_strokes = []

        done = False
        while not done:
            nodes_remain = [k for k, v in is_traversed.items() if not v]
            if len(nodes_remain) == 0:
                done = True
            else:
                node_s = nodes_remain[0]
                stroke, is_traversed = self.get_full_stroke(node_s, 
                                                            zone_conns, 
                                                            is_traversed)
                all_strokes.append(stroke)
            
        # NULL CASE: No strokes. Use initial skeletonization
        if len(all_strokes) == 0:
            all_strokes = self.init_curves

        rdp_curves = []
        rdp_lws = []
        for stroke in all_strokes:
            if debug:
                (X, Y), lws = self.refine_stroke_alt(stroke)
            else:
                (X, Y), lws = self.refine_stroke(stroke)
            rdp_curves.append((X, Y))
            rdp_lws.append(lws)

        return rdp_curves, rdp_lws
    

    def get_cut_curves(self):
        """Get all cut curves. Returns a status whether the curve was
        too short to cut.
        """
        curve_cutoffs = {k: [] for k in range(len(self.init_curves))}

        for fork in self.forks:
            curve_cutoffs = self._detect_cutoffs(fork, curve_cutoffs)

        cut_curves = []
        cut_status = []
        # Apply cut-offs:
        N = len(self.init_curves)
        for i in range(N):
            # Assess if it is a curve or a point based on mdist:

            pos_cutoffs = [k for k in curve_cutoffs[i] if k > 0]
            neg_cutoffs = [k for k in curve_cutoffs[i] if k < 0]

            if len(pos_cutoffs) == 0:
                ind_s = None
            else:
                ind_s = min(pos_cutoffs) # Minimize cutoff
            
            if len(neg_cutoffs) == 0:
                ind_f = None
            else:
                ind_f = max(neg_cutoffs)

            cut_curve, is_cut = self._cut_curve(i, ind_s, ind_f)
            cut_curves.append(cut_curve)
            cut_status.append(is_cut)

        # Store:
        self.cut_curves = cut_curves
        self.cut_status = cut_status

        return cut_curves, cut_status
    

    def resolve_zone(self, zone):
        """Given a defined zone as a list of tuples (node, is_curve),
        apply the connection criteria to each source -> target.
        Return all valid connections with minimal connection curvature.
        """

        num_curves = len(zone["curves"])
        num_points = len(zone["points"])
        num = num_curves + num_points
        
        # Tri-zone:
        if num == 3:
            if num_curves == 3:
                zone_conns = self._resolve_zone_tri_3curve(zone)
            elif num_curves == 2:
                zone_conns = self._resolve_zone_tri_2curve(zone)
            elif num_curves == 1:
                zone_conns = self._resolve_zone_tri_1curve(zone)
            else:
                zone_conns = {} # No connection between 3 points
        
        # Quad-zone:
        # elif num == 4:
            # zone_conns = self._resolve_zone_quad(zone)

        # Other cases (rarely used generalized method):
        else:
            zone_conns = self._resolve_zone_quint(zone)
        return zone_conns
    

    def init_traverse_nodes(self):
        """Get all nodes that need to be traversed. Compile into a dictionary.
        """

        N = len(self.cut_curves)
        is_traversed = {}
        for i in range(N):
            node_s = (i, True)
            node_f = (i, False)

            # Check if it is a valid target:
            is_conn_ftf = self.is_conn_ftf[i]
            is_cut = self.cut_status[i]

            # All cut curves must be traversed
            if is_cut:
                is_traversed[node_s] = False
                is_traversed[node_f] = False
            # Non-cut curves only serve as transitional nodes
            else:
                is_traversed[node_s] = True
                is_traversed[node_f] = True
            
            # Do not traverse through zone ftf curves
            if is_conn_ftf:
                is_traversed[node_s] = True
                is_traversed[node_f] = True
        
        return is_traversed
    

    def get_stroke(self, node_s, all_edges, is_traversed):
        """Complete the stroke starting at node_s using the edge connections.
        """

        node = node_s
        done = False
        stroke_curves = []
        while not done:
            tgt = all_edges.get(node)
            # No more target: Terminate at current curve.
            if tgt is None:
                done = True
            else:
                done = tgt["terminate"]

                # Add midpoint (might be terminal point):
                midnode = tgt.get("midnode")
                if midnode is not None:
                    stroke_curves.append([midnode])

                node_next = tgt.get("node")

                # Check if there is a terminal point:
                if node_next is not None:
                    if is_traversed[node_next]: 
                        break

                    is_traversed[node_next] = True
                    index, is_start = node_next
                    curve_next = self.cut_curves[index]
                    if not is_start:
                        curve_next = np.flip(curve_next)
                    stroke_curves.append(curve_next)
                    # Go to other side of curve
                    node = (index, not is_start)
                    is_traversed[node] = True
                
                else:
                    done = True
        
        # Stroke curve and updated is_traversed
        return stroke_curves, is_traversed
    

    def get_full_stroke(self, node_s, all_edges, is_traversed):
        """Get the full version of the stroke by going bi-directionally.
        """

        stroke_for, is_traversed = self.get_stroke(node_s, all_edges, is_traversed)
        index_s, is_start_s = node_s[0], node_s[1]
        node_f = (index_s, not is_start_s)
        stroke_back, is_traversed = self.get_stroke(node_f, all_edges, is_traversed)
        # Complete reversal.
        stroke_back.reverse()
        for i in range(len(stroke_back)):
            stroke_back[i] = np.flip(stroke_back[i])

        curve_mid = self.cut_curves[index_s]
        if is_start_s:
            curve_mid = np.flip(curve_mid)
        stroke_mid = [curve_mid]
        # Update traversed:
        is_traversed[node_s] = True
        is_traversed[node_f] = True

        # Connect:
        stroke = stroke_back + stroke_mid + stroke_for
        conn_stroke = np.concatenate(stroke)
        return conn_stroke, is_traversed
    

    def refine_stroke_alt(self, stroke):
        """An alternative way to refine the stroke, without using B-spline.
        """

        X, Y = np.unravel_index(stroke, self.img_shape)
        # Remove same endpoint (avoid full loops):
        if X[0] == X[-1] and Y[0] == Y[-1]:
            X, Y = X[:-1], Y[:-1]

        if X.size <= 1:
            return None, None
        
        # Remove repeating points in the curve:
        X, Y = remove_repeating(X, Y)
        
        # Apply Douglas-Peucker algorithm:
        X_rdp, Y_rdp = douglas_peucker(X, Y, self.lw/self.rdp_fac)

        # Sample points between RDP points:
        X_eval, Y_eval, inds_near = sample_points(X_rdp, Y_rdp, self.lw/self.sample_fac)
        X_near = X_rdp[inds_near]
        Y_near = Y_rdp[inds_near]
        line_widths = self.img_mdists[X_near, Y_near]
        return (X_eval, Y_eval), line_widths


    def refine_stroke(self, stroke):
        """Given a stroke, return the coordinates (x, y), alongside the line
        width, using B-spline interpolation. Returns a smoothened curve.
        """

        X, Y = np.unravel_index(stroke, self.img_shape)
        # Remove same endpoint (avoid full loops):
        if X[0] == X[-1] and Y[0] == Y[-1]:
            X, Y = X[:-1], Y[:-1]

        if X.size <= 1:
            return None, None

        X, Y = remove_repeating(X, Y)
        k = min(X.size-1, 3)
        s = X.size // 4
        
        # Use splrep on each coordinate:
        t_arr, arc_length = cum_lengths(X, Y)
        tck_x = splrep(t_arr, X, k=k, s=s)
        tck_y = splrep(t_arr, Y, k=k, s=s)

        # tck, t_arr = splprep((X, Y), k=k, s=s)

        # Also get the arc-length of the stroke:
        length = np.sum(np.sqrt(np.diff(X)**2 + np.diff(Y)**2))
        num_samples = int(length * self.sample_per_pixel) + 2

        # Evaluate at a sampling rate:
        t_inter = np.linspace(t_arr[0], t_arr[-1], 
                              num=num_samples, 
                              endpoint=True)
        
        Xint = splev(t_inter, tck_x)
        Yint = splev(t_inter, tck_y)
        # Xint, Yint = splev(t_inter, tck)

        # Get closest points to each float stroke point Xint, Yint:
        D = cdist(t_arr[:,None], t_inter[:,None])
        inds_t = np.argmin(D, axis=0)
        X_near, Y_near = X[inds_t], Y[inds_t]
        line_widths = self.img_mdists[X_near, Y_near]

        return (Xint, Yint), line_widths


    # SUPPLEMENTARY METHODS
    def _group_forks(self):
        """Using the initial curves and forks, group all nearby connected
        forks together. Get a new ftf status which is True if it the ftf
        curve is associated with the group using the curve index.
        Here, group_radius is a scaling coefficient. 
        Larger radius means more forks are grouped.
        """

        # Go through every fork-to-fork curve to get fork connections:
        N = len(self.is_ftf)
        inds_ftf = [i for i in range(N) if self.is_ftf[i]]
        conns = {P: [] for P in self.forks}

        # Zone fork-to-fork points, no need to traverse
        is_conn_ftf = [False for i in range(N)]

        for index in inds_ftf:
            curve = self.init_curves[index] # A fork-to-fork uncut curve
            fork_s = curve[0]
            fork_f = curve[-1]

            rad_s = self.get_mdist(fork_s)
            rad_f = self.get_mdist(fork_f)
            rad = self.group_radius * (rad_s + rad_f)
            length = curve.size

            # Merge zones + add to ftf curves to not use:
            if length < rad:
                conns[fork_s].append(fork_f)
                conns[fork_f].append(fork_s)
                is_conn_ftf[index] = True
        
        fork_groups = group_forks(conns)

        # Store:
        self.fork_groups = fork_groups
        self.is_conn_ftf = is_conn_ftf

        return fork_groups, is_conn_ftf
    

    def _detect_cutoffs(self, fork, curve_cutoffs):
        """Using each fork, use medial distances to cut off the initial
        curves to obtain cut curves. Curves that are too short to cut
        are reduced to target points.
        """

        fork_nodes = self.node_to_curve[fork]
        # Find medial distance at fork:
        mdist = self.get_mdist(fork)

        for n, node in enumerate(fork_nodes):
            index, is_start = node[0], node[1]
            curve = self.init_curves[index]
            if not is_start:
                curve = np.flip(curve)
            
            # Initial cutting index:
            ind_cut = find_cutoff_index(fork, curve, mdist, self.img_shape)
            # Add local line width to cutting index:
            node_cut = curve[ind_cut-1]
            mdist_c = self.get_mdist(node_cut)
            radius = mdist + mdist_c / 4 # Additional index cut.
            ind_cut = find_cutoff_index(fork, curve, radius, self.img_shape)
            
            if is_start:
                curve_cutoffs[index].append(ind_cut)
            else:
                curve_cutoffs[index].append(-ind_cut)
        
        return curve_cutoffs
    
    
    def _cut_curve(self, i, ind_s, ind_f):
        """Cuts the curve based on the status of the curve.
        """

        # Fork-to-fork zone curves:
        if self.is_conn_ftf[i]:
            return self.init_curves[i], False
        else:
            init_curve = self.init_curves[i]
            cut_curve = init_curve[ind_s:ind_f]
        
        # Always cut fork-to-fork curves:
        if self.is_ftf[i]:
            is_cut = True
            # Not enough points:
            if cut_curve.size <= 2:
                L = init_curve.size
                if L <= self.lw / 2 + 1:
                    cut_curve = init_curve
                else:
                    i_s = int(self.lw / 4)
                    i_f = -int(self.lw / 4)
                    cut_curve = init_curve[i_s:i_f]
                    if cut_curve.size <= 1:
                        cut_curve = init_curve  
        # Non fork-to-fork means its optional:
        else:
            if cut_curve.size <= max(self.lw // (2*self.cut_fac), 1):
                is_cut = False
                cut_curve = init_curve
            else:
                is_cut = True
        return cut_curve, is_cut
        

    def _organize_zones(self):
        """Compile zone group information and cut curves, statuses, to define
        all curve node attributes to look-up:
        - Node point
        - Curve or point
        - Whether it is a short fork-to-fork point (to ignore)
        - Associated fork (if any)
        - Associated fork group as a connection candidate (if any)

        Also for each fork group, obtain all candidate curve nodes.
        """

        # Update the attributes using fork groups:
        zones = []
        for n, group in enumerate(self.fork_groups):
            # In each group:
            zone = {
                "curves": [],
                "points": [],
                "forks": []
            }

            for fork in group:
                zone["forks"].append(fork)

                curve_nodes = self.node_to_curve[fork]
                for node in curve_nodes:
                    # Check if conn ftf:
                    index, is_start = node[0], node[1]
                    if self.is_conn_ftf[index]:
                        continue
                    is_cut = self.cut_status[index]
                    if is_cut:
                        zone["curves"].append(node)
                    else:
                        zone["points"].append(node)
            
            # Calculate zone midpoint:
            zone["center"] = median_node(zone["forks"], self.img_shape)
            zones.append(zone)

        self.zones = zones
        return zones
    

    def _get_nearby_segment(self, node):
        """Given a node, find the straightest segment of the cut curve
        approaching the node.
        """

        index, is_start = node[0], node[1]
        curve = self.cut_curves[index]

        if not is_start:
            curve = np.flip(curve)

        X, Y = np.unravel_index(curve, self.img_shape)

        err_thr = max(self.lw / (2*self.nearby_fac), np.sqrt(2))
        cut = nearest_straight_index(X, Y, err_thr)
        # You want it to approach the node:
        Xcut, Ycut = np.flip(X[:cut]), np.flip(Y[:cut])
        return Xcut, Ycut
    

    def _get_tangental_segment(self, node, return_point=False):
        """Given a (cut) curve, returns the tangental segment, up to a limit
        of line width. The tangent faces the direction of the point.
        """

        X, Y = self._get_nearby_segment(node)
        xp = X[-1] - X[0]
        yp = Y[-1] - Y[0]
        xp, yp = normalize_vector(xp, yp)

        if return_point:
            return (xp, yp), (X[-1], Y[-1])
        else:
            return (xp, yp)
    

    def _get_terminal_point(self, node, away=True):
        """Returns the terminal point at node = (index, is_start), the point
        away from the fork.
        """

        index, is_start = node[0], node[1]
        init_curve = self.cut_curves[index]
        # Get point away from fork:
        if is_start:
            ind = -1 if away else 0
            centerpoint = init_curve[ind]
        else:
            ind = 0 if away else -1
            centerpoint = init_curve[ind]
        return centerpoint


    def _resolve_zone_tri_3curve(self, zone):
        """Given a tri-zone with 3 curves, returns all segment connections.
        """

        tangents = []
        nodes_curve = zone["curves"]
        for i, node in enumerate(nodes_curve):
            xp, yp = self._get_tangental_segment(node)
            tangents.append((xp, yp))

        # Find most parallel tangents:
        N = len(tangents) # Should be 3
        angles = []
        for j in range(len(tangents)):
            xps, yps = tangents[j]
            xpf, ypf = tangents[(j+1) % N]
        
            dot = (-xps*xpf - yps*ypf)
            angle = np.arccos(np.clip(dot, -1, 1))
            angles.append(angle)
        
        # Connect pair with minimal angle:
        ind_min = np.argmin(angles)
        ind_next = (ind_min + 1) % N
        ind_omit = (ind_min + 2) % N
        node_s = nodes_curve[ind_min]
        node_f = nodes_curve[ind_next]
        node_omit = nodes_curve[ind_omit]

        centernode = zone["center"]

        all_conns = {}
        all_conns[node_s] = {
            "terminate": False,
            "node": node_f
        }
        all_conns[node_f] = {
            "terminate": False,
            "node": node_s
        }
        all_conns[node_omit] = {
            "terminate": True,
            "midnode": centernode
        }
        return all_conns
    

    def _resolve_zone_tri_2curve(self, zone):
        """Given a tri-zone with 2 curves, returns all segment connections.
        """

        nodes_curve = zone["curves"]
        node_point = zone["points"][0]

        # Get point away from fork:
        centerpoint = self._get_terminal_point(node_point)
        x0, y0 = np.unravel_index(centerpoint, self.img_shape)

        # Collect all points:
        node1, node2 = nodes_curve[0], nodes_curve[1]

        (xp1, yp1), (x1, y1) = self._get_tangental_segment(node1, return_point=True)
        (xp2, yp2), (x2, y2) = self._get_tangental_segment(node2, return_point=True)

        xp1_p, yp1_p = normalize_vector(x0 - x1, y0 - y1)
        xp2_p, yp2_p = normalize_vector(x0 - x2, y0 - y2)
        xp1_c, yp1_c = normalize_vector(x2 - x1, y2 - y1)

        # Determine more parallel connection:
        dot1_p = xp1*xp1_p + yp1*yp1_p
        dot1_c = xp1*xp1_c + yp1*yp1_c

        dot2_p = xp2*xp2_p + yp2*yp2_p
        dot2_c = -xp2*xp1_c - yp2*yp1_c

        # True: Connect to point. False: Connect to fork.
        conn1 = dot1_p > dot1_c
        conn2 = dot2_p > dot2_c

        all_conns = {}
        # Both points:
        if conn1 and conn2:
            all_conns[node1] = {
                "terminate": False,
                "node": node2,
                "midnode": centerpoint
            }
            all_conns[node2] = {
                "terminate": False,
                "node": node1,
                "midnode": centerpoint
            }
        
        # Ignore point:
        elif not conn1 and not conn2:
            all_conns[node1] = {
                "terminate": False,
                "node": node2
            }
            all_conns[node2] = {
                "terminate": False,
                "node": node1
            }
        
        # Separate points:
        else:
            forkpoint = zone["center"]
            all_conns[node1] = {
                "terminate": True,
                "midnode": centerpoint if conn1 else forkpoint
            }
            all_conns[node2] = {
                "terminate": True,
                "midnode": centerpoint if conn2 else forkpoint
            }

        return all_conns
    

    def _resolve_zone_tri_1curve(self, zone):
        """Given a tri-zone with 1 curve, returns all segment connections.
        This will be a single connnection where the single curve terminates
        at the more parallel connection to a point.
        """

        node_curve = zone["curves"][0]
        node_points = zone["points"]

        (xp0, yp0), (x0, y0) = self._get_tangental_segment(node_curve, 
                                                           return_point=True)
        # Get points away from fork:
        p1 = self._get_terminal_point(node_points[0])
        x1, y1 = np.unravel_index(p1, self.img_shape)

        p2 = self._get_terminal_point(node_points[1])
        x2, y2 = np.unravel_index(p2, self.img_shape)

        xp1, yp1 = normalize_vector(x1 - x0, y1 - y0)
        xp2, yp2 = normalize_vector(x2 - x0, y2 - y0)

        # Determine more parallel connection:
        dot1 = xp0*xp1 + yp0*yp1
        dot2 = xp0*xp2 + yp0*yp2

        all_conns = {}
        all_conns[node_curve] = {
            "terminate": True,
            "midnode": p1 if (dot1 > dot2) else p2
        }

        return all_conns
    

    def _resolve_zone_quad(self, zone):
        """Resolves a zone with 4 candidates. Prioritizes the cross-connection.
        """

        nodes_curves = zone["curves"]
        nodes_points = zone["points"]
        nodes = nodes_curves + nodes_points
        statuses = [True for i in range(len(nodes_curves))] \
            + [False for j in range(len(nodes_points))]

        # For each node, get the terminal point towards fork:
        points = []
        for i, node in enumerate(nodes):
            status = statuses[i]
            point = self._get_terminal_point(node, away=(not status))
            points.append(point)
        
        # Convert to coordinates:
        X, Y = np.unravel_index(points, self.img_shape)

        # Order in counter-clockwise direction:
        coords = np.stack([X, Y]).T
        inds_sort = sort_ccwise(coords)

        # Do a validity check. If not valid, use general resolution.

        # Establish connections:
        all_conns = {}
        for i in range(inds_sort.size):
            i_s = inds_sort[i]
            i_f = inds_sort[(i+2) % 4]
            node_s = nodes[i_s]
            node_f = nodes[i_f]

            status_s = statuses[i_s]
            status_f = statuses[i_f]

            # Only add to connections if curve:
            if not status_s:
                continue
            key_node = "node" if status_f else "midnode"
            if status_f:
                tgt = node_f
            else:
                tgt = self._get_terminal_point(node_f)
            all_conns[node_s] = {
                "terminate": not status_f,
                key_node: tgt
            }
        
        return all_conns
    

    def _connection_criteria(self, seg_s, seg_f):
        """Given two curve or point segments, returns the connectivity
        score by assessing the dot product of the tangents. Return None
        if it is an invalid connection.
        """

        node_s, is_curve_s = seg_s[0], seg_s[1]
        node_f, is_curve_f = seg_f[0], seg_f[1]

        # Two points: Do not connect
        if not is_curve_s and not is_curve_f:
            return None
        
        if is_curve_s:
            (xps, yps), (xs, ys) = self._get_tangental_segment(node_s, 
                                                               return_point=True)
        else:
            ps = self._get_terminal_point(node_s)
            xs, ys = np.unravel_index(ps, self.img_shape)
        
        if is_curve_f:
            (xpf, ypf), (xf, yf) = self._get_tangental_segment(node_f, 
                                                               return_point=True)
        else:
            pf = self._get_terminal_point(node_f)
            xf, yf = np.unravel_index(pf, self.img_shape)
        
        # Evaluate connection tangent:
        xpm, ypm = normalize_vector(xf - xs, yf - ys)

        # Get maximal dot product:
        if is_curve_s:
            dot_s = xps*xpm + yps*ypm
        else:
            dot_s = 0
        
        if is_curve_f:
            dot_f = -xpf*xpm - ypf*ypm
        else:
            dot_f = 0
        
        # Take maximum:
        dot = max(dot_s, dot_f)
        return dot
    

    def _resolve_zone_quint(self, zone):
        """General case to resolve an ambiguous zone.
        """

        nodes_curves = zone["curves"]
        nodes_points = zone["points"]
        nodes = nodes_curves + nodes_points
        statuses = [True for i in range(len(nodes_curves))] \
            + [False for j in range(len(nodes_points))]
        
        all_possible_conns = []
        for i_s, node_s in enumerate(nodes):
            is_curve_s = statuses[i_s]

            # Do not add points:
            if not is_curve_s:
                continue

            for i_f, node_f in enumerate(nodes):
                if i_s == i_f:
                    continue
                is_curve_f = statuses[i_f]
                kappa = self._connection_criteria((node_s, is_curve_s),
                                                  (node_f, is_curve_f))
                # Angle threshold:
                if kappa > self.quint_angle_thr:
                    all_possible_conns.append((i_s, i_f, kappa))
                
        # Go through connections one by one starting from lowest score:
        if len(all_possible_conns) == 0:
            return {}

        all_conns = {}
        scores = [x[2] for x in all_possible_conns]
        inds_sort = np.flip(np.argsort(scores)) # Highest to lowest
        for i in inds_sort:
            i_s, i_f, _ = all_possible_conns[i]
            node_s = nodes[i_s]
            is_curve_s = statuses[i_s]
            node_f = nodes[i_f]
            is_curve_f = statuses[i_f]

            # Already added this curve:
            if node_s in all_conns.keys():
                continue
            
            # ADD CONNECTIONS
            if is_curve_f:
                # Bi-directional connection:
                all_conns[node_s] = {
                    "terminate": False,
                    "node": node_f
                }
                all_conns[node_f] = {
                    "terminate": False,
                    "node": node_s
                }

            else:
                pf = self._get_terminal_point(node_f)
                all_conns[node_s] = {
                    "terminate": True,
                    "midnode": pf
                }

        # For any non-connected curves, terminate at fork.
        for k, node in enumerate(nodes):
            status = statuses[k]
            if status and node not in all_conns.keys():
                all_conns[node] = {
                    "terminate": True,
                    "midnode": zone["center"]
                }    

        return all_conns