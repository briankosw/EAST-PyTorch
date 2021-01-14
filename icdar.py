import csv
import os
from glob import glob
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# Text that should not affect evaluation: https://rrc.cvc.uab.es/?ch=4&com=tasks
DO_NOT_CARE = "###"


class ICDARDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        input_shape: Union[int, Tuple[int, int]],
        min_text_size: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(ICDARDataset, self).__init__(root, transforms, transform, target_transform)
        image_pathname = os.path.join(root, "images/*.jpg")
        self.images = sorted(glob(image_pathname))
        annotation_pathname = os.path.join(root, "annotations/*.txt")
        self.annotations = sorted(glob(annotation_pathname))
        if isinstance(input_shape, tuple):
            self.input_shape = input_shape
        else:
            self.input_shape = (input_shape, input_shape)
        self.min_text_size = min_text_size

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        quads, texts = load_annotation(self.annotations[idx])
        # Crop
        image, quads, texts = self.transforms(image, quads, texts)
        # Generate maps
        score_map, geometry_map, train_ignore_mask, train_boundary_mask = generate_maps(
            quads, texts, input_shape=self.input_shape, min_text_size=self.min_text_size
        )
        # Image augmentation
        if self.transform:
            image = self.transform(image)
        return image, quads, texts

    def __len__(self) -> int:
        return len(self.images)


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, image, quads, texts):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
        width, height = F._get_image_size(image)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
        top, left, height, width = self.get_params(image, self.size)
        image = F.crop(image, top, left, height, width)
        bottom, right = top + height, left + width
        indices = []
        for i in range(len(quads)):
            quads[i] = sort_vertices(quads[i])
            quad_left, quad_top = quads[i][0]
            quad_right, quad_bottom = quads[i][2]
            if (
                left < quad_left and top < quad_top and quad_right < right and quad_bottom < bottom
            ):
                # quads[i, :, 0] = np.minimum(quads[i, :, 0], right)
                # quads[i, :, 0] = np.maximum(quads[i, :, 0], left)
                # quads[i, :, 1] = np.minimum(quads[i, :, 1], bottom)
                # quads[i, :, 1] = np.maximum(quads[i, :, 1], top)
                indices.append(i)
        quads = quads[indices] - (left, top)
        texts = texts[indices]
        return image, quads, texts



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
    min_text_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates the score map

    Arguments:
        quads:
        texts:
        input_shape:
        min_text_size:

    Returns:
        score_map
    """
    h, w = input_shape
    score_map = np.zeros((h, w), dtype=np.uint8)
    geometry_map = np.zeros((h, w, 5), dtype=np.float32)
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
    score_map = torch.as_tensor(score_map, dtype=torch.float32)
    geometry_map = torch.as_tensor(geometry_map, dtype=torch.float32)
    train_ignore_mask = torch.as_tensor(train_ignore_mask, dtype=torch.float32)
    train_boundary_mask = torch.as_tensor(train_boundary_mask, dtype=torch.float32)
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

