import csv
import os
from glob import glob
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Text that should not affect evaluation: https://rrc.cvc.uab.es/?ch=4&com=tasks
DO_NOT_CARE = "###"


def load_annotation(annotation_filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the quadrilaterals' coordinates and the transcription

    Arguments:
        annotation_filepath:

    Returns:

    """
    quads = []
    texts = []
    with open(annotation_filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # the annotation files aren't fully cleaned, so sometimes there are more
            # than 9 values in some rows
            x1, y1, x2, y2, x3, y3, x4, y4, text = [
                elem.strip("\ufeff") for elem in row[:9]
            ]
            quads.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            if text == DO_NOT_CARE:
                texts.append(False)
            else:
                texts.append(True)
    quads = np.array(quads, dtype=np.float)
    texts = np.array(texts, dtype=np.bool)
    return quads, texts


def generate_maps(
    quads: np.ndarray,
    texts: np.ndarray,
    input_shape: Tuple[int, int],
    min_text_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the score map

    Arguments:
        quads:
        texts:
        input_shape:
        min_text_size:

    Returns:
    """
    h, w = input_shape
    score_map = np.zeros((h, w), dtype=np.uint8)
    geometry_map = np.zeros((h, w), dtype=np.float32)
    train_ignore_mask = np.ones((h, w), dtype=np.uint8)
    train_boundary_mask = np.ones((h, w), dtype=np.uint8)
    original_quad_mask = np.zeros((h, w), dtype=np.uint8)
    shrinked_quad_mask = np.zeros((h, w), dtype=np.uint8)
    for i, (quad, text) in enumerate(zip(quads, texts)):
        # Calculate the reference lengths of the quadrilateral
        # https://arxiv.org/pdf/1704.03155.pdf#page=4
        ref_len = np.zeros(4, dtype=np.float32)
        for j in range(4):
            ref_len[j] = min(
                np.linalg.norm(quad[j] - quad[(j + 1) % 4]),
                np.linalg.norm(quad[j] - quad[(j - 1) % 4]),
            )
        shrinked_quad = shrink_quad(quad.copy(), ref_len)
        cv2.fillPoly(score_map, shrinked_quad, 1)
        cv2.fillPoly(original_quad_mask, quad[np.newaxis, ...].astype(np.int32), 1)
        cv2.fillPoly(score_map, shrinked_quad, i + 1)
        shrinked_quad_points = np.argwhere(shrinked_quad_mask == i + 1)
        dists = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        top, right, bottom, left = dists
        if not text or min(top, right, bottom, left) < min_text_size:
            cv2.fillPoly(train_ignore_mask, quad[np.newaxis, ...].astype(np.int32), 0)
        mbr, angle = minimum_bounding_rectangle(quad)
        for y, x in shrinked_quad_points:
            point = np.array([x, y])
            geometry_map[y, x, 0] = dist_to_boundary(mbr[0], mbr[1], point)
            geometry_map[y, x, 1] = dist_to_boundary(mbr[1], mbr[2], point)
            geometry_map[y, x, 2] = dist_to_boundary(mbr[2], mbr[3], point)
            geometry_map[y, x, 3] = dist_to_boundary(mbr[3], mbr[0], point)
            geometry_map[y, x, 4] = angle
    shrinked_quad_mask = (shrinked_quad_mask > 0).astype(np.uint8)
    train_boundary_mask = 1 - (original_quad_mask - shrinked_quad_mask)
    return score_map, geometry_map, train_ignore_mask, train_boundary_mask


def shrink_quad(
    quad: np.ndarray, ref_len: np.ndarray, shrink_ratio: float = 0.3
) -> np.ndarray:
    """
    Shrinks the edges of quadrilateral

    Arguments:
        quad:
        ref_len:
        shrink_ratio:

    Returns:

    Reference:
        - https://arxiv.org/pdf/1704.03155.pdf#page=4
    """
    top_len = np.linalg.norm(quad[0] - quad[1])
    bottom_len = np.linalg.norm(quad[2] - quad[3])
    left_len = np.linalg.norm(quad[3] - quad[0])
    right_len = np.linalg.norm(quad[1] - quad[2])
    shrink_order = []
    if top_len + bottom_len > left_len + right_len:
        shrink_order = [(1, 0), (2, 3), (3, 0), (2, 1)]
    else:
        shrink_order = [(3, 0), (2, 1), (1, 0), (2, 3)]
    for i, j in shrink_order[:2]:
        dx, dy = quad[i] - quad[j]
        theta = np.arctan2(dy, dx)
        quad[j][0] += ref_len[j] * np.cos(theta) * 0.3
        quad[j][1] += ref_len[j] * np.sin(theta) * 0.3
        quad[i][0] -= ref_len[i] * np.cos(theta) * 0.3
        quad[i][1] -= ref_len[i] * np.sin(theta) * 0.3
    for i, j in shrink_order[2:]:
        dx, dy = quad[i] - quad[j]
        theta = np.arctan2(dx, dy)
        quad[j][0] += ref_len[j] * np.sin(theta) * 0.3
        quad[j][1] += ref_len[j] * np.cos(theta) * 0.3
        quad[i][0] -= ref_len[i] * np.sin(theta) * 0.3
        quad[i][1] -= ref_len[i] * np.cos(theta) * 0.3
    quad = quad[np.newaxis, ...].astype(np.int32)
    return quad


def minimum_bounding_rectangle(quad: np.ndarray) -> Tuple[np.ndarray, float]:
    *rect, angle = cv2.minAreaRect(quad.astype(np.int32))
    rect = cv2.boxPoints((*rect, angle))
    angle *= -1
    if angle > 45:
        angle -= 90
    rect = sort_vertices(rect)
    return rect, angle


def sort_vertices(quad: np.ndarray) -> np.ndarray:
        # Calculate the centroid and use the centroid to determine left and right
    # vertices. For each side, compare the y value to determine top and bottom.
    center = np.mean(quad, axis=0)
    left = np.argwhere(quad[:, 0] < center[0]).squeeze()
    tl, bl = left[np.argsort(quad[left, 1])]
    right = np.argwhere(quad[:, 0] > center[0]).squeeze()
    tr, br = right[np.argsort(quad[right, 1])]
    quad = quad[[tl, tr, br, bl]]
    return quad

def dist_to_boundary(p0: np.ndarray, p1: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Calculates the distance between a line (formed by a pair of coordinates) and a
    point.

    Arguments:
        p0: the first coordinate that forms the line
        p1: the second coordinate that forms the line
        p: the point of interest for calculating distance

    Returns:

    """
    return np.linalg.norm(np.cross(p1 - p0, p0 - p)) / np.linalg.norm(p1 - p0)

