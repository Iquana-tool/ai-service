"""Pure array ops for hierarchy-aware SAM 3 -- no torch, no model, no HF.

Hierarchical SAM 3 searches for a child concept (e.g. coral polyps) only *inside* the
existing objects of its parent label (e.g. the coral fragments). For each parent it crops the
image to the parent's padded bbox and blurs everything outside the parent mask, so SAM 3's
full input resolution and attention land on the parent. Detections are then pasted back to
full-image coordinates and clipped to the parent.

This module is the geometry half -- cropping, focusing, assigning exemplars to parents, and
mapping crop-space masks back -- kept free of torch and gated weights so it is unit-testable.
The SAM 3 forward passes live in :mod:`models.sam3_hierarchical`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from models.embedding_ops import mask_bbox


@dataclass(frozen=True)
class FocusCrop:
    """One parent region cut out of the image.

    ``box`` is ``(y0, y1, x0, x1)`` in full-image pixels (end-exclusive); ``image`` is the
    focused crop (outside-parent pixels blurred) and ``parent`` the parent mask in crop space.
    """

    box: tuple[int, int, int, int]
    image: np.ndarray
    parent: np.ndarray

    def to_crop(self, mask: np.ndarray) -> np.ndarray:
        """Cut a full-image mask down to this crop."""
        y0, y1, x0, x1 = self.box
        return np.asarray(mask)[y0:y1, x0:x1]


def containment(mask: np.ndarray, region: np.ndarray) -> float:
    """Fraction of ``mask``'s foreground that lies inside ``region`` (0 for an empty mask)."""
    m = np.asarray(mask).astype(bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    return int(np.logical_and(m, np.asarray(region).astype(bool)).sum()) / total


def assign_to_regions(
    masks: list[np.ndarray], regions: list[np.ndarray], min_frac: float = 0.5
) -> list[int | None]:
    """Index of the region containing each mask best (at least ``min_frac``), else ``None``."""
    assigned: list[int | None] = []
    for mask in masks:
        best, best_frac = None, min_frac
        for i, region in enumerate(regions):
            frac = containment(mask, region)
            if frac >= best_frac and (best is None or frac > best_frac):
                best, best_frac = i, frac
        assigned.append(best)
    return assigned


def focus_crop(
    image: np.ndarray,
    region: np.ndarray,
    padding: float = 0.05,
    blur: float = 0.03,
) -> FocusCrop:
    """Crop ``image`` to ``region``'s bbox (padded by ``padding`` of its side) and blur the rest.

    ``blur`` is the Gaussian sigma as a fraction of the crop's longer side; ``0`` disables it.
    Relative, so a small fragment and a large one get the same amount of defocus.
    """
    region = np.asarray(region).astype(bool)
    box = mask_bbox(region, pad_frac=padding)
    y0, y1, x0, x1 = box
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    parent = region[y0:y1, x0:x1]

    sigma = blur * max(crop.shape[0], crop.shape[1])
    if sigma > 0 and not parent.all():
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
        keep = parent[..., None] if crop.ndim == 3 else parent
        crop = np.where(keep, crop, blurred).astype(image.dtype)
    return FocusCrop(box=box, image=crop, parent=parent)


def bbox_xyxy(mask: np.ndarray, x_off: int = 0, y_off: int = 0) -> list[float] | None:
    """Pixel ``[xmin, ymin, xmax, ymax]`` of a mask, shifted by an offset; ``None`` if empty."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    y0, y1, x0, x1 = mask_bbox(m)
    return [float(x0 + x_off), float(y0 + y_off), float(x1 + x_off), float(y1 + y_off)]


def paste_back(
    crop_masks: list[np.ndarray],
    crop: FocusCrop,
    image_hw: tuple[int, int],
    min_inside_frac: float = 0.5,
    max_parent_iou: float = 0.8,
) -> tuple[list[np.ndarray], list[int]]:
    """Map crop-space detections to full-image masks clipped to the parent.

    A detection is dropped when less than ``min_inside_frac`` of it lies on the parent (it
    latched onto the blurred surroundings) or when it is essentially the parent itself
    (IoU above ``max_parent_iou`` -- SAM 3 re-segmenting the fragment, not a child).
    Returns the full-size boolean masks and their indices into ``crop_masks``.
    """
    y0, y1, x0, x1 = crop.box
    parent = crop.parent
    parent_area = int(parent.sum())
    kept: list[np.ndarray] = []
    indices: list[int] = []
    for i, mask in enumerate(crop_masks):
        mask = np.asarray(mask).astype(bool)
        if containment(mask, parent) < min_inside_frac:
            continue
        clipped = np.logical_and(mask, parent)
        area = int(clipped.sum())
        if area == 0:
            continue
        if parent_area and area / (parent_area + int(mask.sum()) - area) > max_parent_iou:
            continue
        full = np.zeros(image_hw, dtype=bool)
        full[y0:y1, x0:x1] = clipped
        kept.append(full)
        indices.append(i)
    return kept, indices
