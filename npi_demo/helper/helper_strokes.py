import numpy as np
from scipy.interpolate import splprep, splev

EPS = 1e-6


def fit_new_shape(old_shape, new_shape, margins, max_scale):

    # New shape with margin:
    new_shape0 = (new_shape[0]-2*margins[0], new_shape[1]-2*margins[1])
    h_ratio = new_shape0[0] / old_shape[0]
    w_ratio = new_shape0[1] / old_shape[1]

    ratio = min(h_ratio, w_ratio)
    if h_ratio >= 1 and w_ratio >= 1:
        ratio = min(ratio, max_scale)
    
    diff_h = int(max(new_shape[0] - ratio*old_shape[0], 0) / 2)
    diff_w = int(max(new_shape[1] - ratio*old_shape[1], 0) / 2)

    offset = (diff_h, diff_w)
    return ratio, offset


def cum_lengths(X, Y):
    """Evaluates the arc-length along X, Y, and returns
    the arc-length and corresponding time-step values.
    """

    dists = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
    lengths = np.cumsum(dists)
    lengths = np.concatenate(([0], lengths))
    return lengths


def perpendicular_dist(p0, pl1, pl2):
    """Returns the distance from point p0 to the line segment through
    points pl1, pl2.
    """

    d_x = pl2[0] - pl1[0]
    d_y = pl2[1] - pl1[1]

    n_x, n_y = -d_y, d_x
    C = -pl1[0]*n_x - pl1[1]*n_y
    norm = max(np.sqrt(n_x**2 + n_y**2), EPS)
    dist = np.abs(n_x * p0[0] + n_y * p0[1] + C) / norm
    return dist


def remove_repeating(X, Y, radius=EPS):
    """Given a 2d stroke array, filters out consecutive repeating points.
    Points are considered repeating if Euclidean distance is within radius.
    """

    diff = np.diff(X)**2 + np.diff(Y)**2
    inds = 1 + np.where(diff > radius**2)[0]
    Xc = np.concatenate(([X[0]], X[inds]))
    Yc = np.concatenate(([Y[0]], Y[inds]))

    if Xc.size < X.size:
        Xc, Yc = remove_repeating(Xc, Yc, radius=radius)
    return Xc, Yc 


def douglas_peucker(X, Y, eps):
    """Resample using the Douglas-Peucker algorithm. Do not apply to
    closed curves!
    """

    dmax = 0
    ind = 0
    end = X.size
    p_start = (X[0], Y[0])
    p_end = (X[end-1], Y[end-1])
    for i in range(1, end-1):
        p0 = (X[i], Y[i])
        D = perpendicular_dist(p0, p_start, p_end)
        if D > dmax:
            ind = i
            dmax = D
        
    if dmax > eps:
        X1, Y1 = douglas_peucker(X[:ind], Y[:ind], eps)
        X2, Y2 = douglas_peucker(X[ind:], Y[ind:], eps)
        X_rs = np.concatenate((X1, X2))
        Y_rs = np.concatenate((Y1, Y2))
    
    else:
        X_rs = np.array([X[0], X[-1]])
        Y_rs = np.array([Y[0], Y[-1]])
    
    X_out, Y_out = remove_repeating(X_rs, Y_rs, radius=eps)
    return X_out, Y_out
    

def cubic_approximation(t_arr, X, Y, err, deg=3):
    """Given a stroke, approximate using a series of polynomials of fixed
    degrees. Each polynomial will be a separate Bezier curve.
    """

    # Normalize time array:
    t_arr = t_arr - t_arr[0]
    t_arr = t_arr / (t_arr[-1]+ EPS)

    # Skip over if not enough points:
    if t_arr.size < 2:
        return True, None

    deg0 = min(deg, t_arr.size-1)

    # Cubic approximation:
    coeffs_x = np.polyfit(t_arr, X, deg=deg0)
    coeffs_y = np.polyfit(t_arr, Y, deg=deg0)

    # Evaluate polynomial:
    X_poly = np.polyval(coeffs_x, t_arr)
    Y_poly = np.polyval(coeffs_y, t_arr)

    # Get error. Cut-off if error is too large.
    dists = np.sqrt((X_poly - X)**2 + (Y_poly - Y)**2)
    return np.max(dists) <= err, (coeffs_x, coeffs_y)


def pixel_spline(X, Y, fac_curve, deg=2):
    """Given a sequence of unique x, y points, applies a cubic
    interpolation and returns all pixel-points in-between.
    """

    X, Y = remove_repeating(X, Y)
    k = min(X.size-1, deg)
    s = 1 # max(1, X.size // 4)
    tck, t_inter = splprep((X, Y), k=k, s=s)
    
    Xints, Yints = [], []
    for i in range(t_inter.size-1):
        dist_line = (X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2
        num_points = max(int(deg * np.sqrt(dist_line)), deg)
        t_arr = np.linspace(t_inter[i], t_inter[i+1], 
                            num=num_points, endpoint=False)
        
        Xint, Yint = splev(t_arr, tck)
        # Test for overshoot:
        Xline = np.linspace(X[i], X[i+1], num=Xint.size, endpoint=False)
        Yline = np.linspace(Y[i], Y[i+1], num=Yint.size, endpoint=False)
        dists = (Xint - Xline)**2 + (Yint - Yline)**2
        dist_max = np.max(dists)
        if dist_max > fac_curve**2 * dist_line:
            Xint, Yint = Xline, Yline 
        
        Xints.append(Xint)
        Yints.append(Yint)
    
    Xout = np.concatenate(Xints)
    Yout = np.concatenate(Yints)

    Xout, Yout = remove_repeating(Xout, Yout)
    return Xout, Yout


def pixelate(X, Y):
    """Given a curve with X, Y positions, returns the integer versions of
    the curve by rounding and removing duplicates.
    """

    X_pixels = np.rint(X).astype(np.int32)
    Y_pixels = np.rint(Y).astype(np.int32)
    
    X_pixels, Y_pixels = remove_repeating(X_pixels, Y_pixels)
    return X_pixels, Y_pixels