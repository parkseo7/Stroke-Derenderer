from xml.dom import minidom
from svgpathtools import (
    Path, 
    Line, 
    CubicBezier,
    parse_path, 
    svg2paths, 
    wsvg, 
    smoothed_path
)

import matplotlib.pyplot as plt
import numpy as np
import cv2


def read_clipstudio_svg(filepath):
    """Reads an SVG file exported by Clip Studio. Returns all strokes in
    the form of Bezier paths, along with the stroke width and color.
    """

    doc = minidom.parse(filepath)  # parseString also exists

    # Needs to scale image by viewbox:
    svg = doc.getElementsByTagName('svg')[0]
    svg_width = float(svg.getAttribute('width').replace('pt', ''))
    svg_height = float(svg.getAttribute('height').replace('pt', ''))

    # Get translation:
    transf = doc.getElementsByTagName('g')[0]
    str_tf = transf.getAttribute('transform')

    if len(str_tf.strip()) > 0:
        # Process transform
        str_tf = str_tf.replace("translate", "")
        str_tf = str_tf.replace("(", "")
        str_tf = str_tf.replace(")", "")

        # Split:
        tf_coords = str_tf.split(",")
        x, y = tf_coords[0].strip(), tf_coords[1].strip()
        x = float(x)
        y = float(y)
    
    else:
        x = 0
        y = 0

    # Compile each strokes:
    all_strokes = []
    for path in doc.getElementsByTagName('path'):
        str_coords = path.getAttribute('d')

        str_color = path.getAttribute("stroke")
        str_width = path.getAttribute("stroke-width")
        str_opacity = path.getAttribute("stroke-opacity")

        # Process the string coordinates to Bezier curves:
        bezier_path = parse_path(str_coords)
        if len(bezier_path) > 0:
            bezier_path = smoothed_path(bezier_path)
        
        # Transform bezier path:
        bezier_path = bezier_path.translated(x + 1j*y)

        # Process stroke attributes:
        str_colors = str_color.split(',')
        colors = [''.join(c for c in x if c.isdigit()) for x in str_colors]
        colors = [int(x) for x in colors]

        width = float(str_width)
        opacity = float(str_opacity)
        
        stroke = {
            "bezier":bezier_path,
            "color": colors,
            "width": width,
            "opacity": opacity
        }
        all_strokes.append(stroke)

    doc.unlink()

    # Compiile as processed SVG file:
    svg_dict = {
        "strokes": all_strokes,
        "dim": (svg_height, svg_width),
        "translate": (x, y)
    }

    return svg_dict


def update_svg_with_coords(svg_dict, img_shape, num_samples=3):
    """Given an imported SVG dictionary containing all stroke information
    and the underlying image shape, returns the stroke coordinates for all
    Bezier paths per stroke. These points can be used for NPI, HWR. Updates
    the SVG dictionary.
    """

    height, width = img_shape[0], img_shape[1]
    svg_height, svg_width = svg_dict["dim"]
    ratio = np.mean([height / svg_height, width / svg_width])

    all_strokes = svg_dict["strokes"]
    new_strokes = []

    for k, stroke in enumerate(all_strokes):
        p = stroke["bezier"]
        points = bezier_to_points(p, ratio, num_samples=num_samples)
        if points is None:
            continue

        (X, Y) = points
        stroke["coords"] = (X, Y)
        new_strokes.append(stroke)

    svg_dict["strokes"] = new_strokes
    return svg_dict


def bezier_to_points(stroke, ratio, num_samples=3):
    """Given a sequence of Bezier paths, and the corresponding shapes,
    samples points between each Bezier segment, and returns the
    stroke coordinates per stroke. Here, the ratio is the mean ratio
    image_dims / svg_dims.
    """

    p = stroke
    p = p.scaled(ratio) # To match background image

    if len(p) == 0:
        return None

    X_arr = []
    Y_arr = []
    for sub_p in p:
        # Exclude endpoint:
        samples = [sub_p.point(i/num_samples) for i in range(num_samples)]
        # Decipher complex number:
        X = np.array([samples[i].real for i in range(len(samples))])
        Y = np.array([samples[i].imag for i in range(len(samples))])

        X_arr.append(X)
        Y_arr.append(Y)

    X_arr = np.concatenate(X_arr)
    Y_arr = np.concatenate(Y_arr)

    return (X_arr, Y_arr)
    