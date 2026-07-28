"""Pure array ops for building DINOv3 embeddings -- no torch, no model, no HuggingFace.

Split out from :mod:`models.dinov3_embedder` so the geometry and pooling math (crop to a
padded mask bbox, resize a mask onto the patch grid, mean-pool the foreground patches,
L2-normalize) is unit-testable with plain NumPy, independent of the heavy backbone and of
gated model weights. The embedder is then a thin wrapper: run the backbone, hand its dense
grid / CLS token to these functions.
"""
from __future__ import annotations

import cv2
import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm; a (near-)zero vector is returned unchanged."""
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def mask_bbox(mask: np.ndarray, pad_frac: float = 0.0) -> tuple[int, int, int, int]:
    """Padded ``(y0, y1, x0, x1)`` bounds of a binary mask.

    The padding is ``pad_frac`` of each side length, clamped to the image. An all-zero mask
    has no foreground to bound, so the full frame is returned (the caller then embeds the
    whole crop -- a sensible, low-signal fallback rather than a crash).
    """
    m = np.asarray(mask).astype(bool)
    h, w = m.shape[:2]
    ys, xs = np.where(m)
    if ys.size == 0:
        return 0, h, 0, w
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray, pad_frac: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both ``image`` and ``mask`` to the mask's padded bbox (see :func:`mask_bbox`)."""
    y0, y1, x0, x1 = mask_bbox(mask, pad_frac)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def resize_mask_to_grid(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a binary mask onto the ``(Hp, Wp)`` patch grid.

    Nearest-neighbour keeps the result strictly binary (no interpolated edge values), so a
    patch is foreground iff the mask covers its centre.
    """
    hp, wp = grid_hw
    resized = cv2.resize(np.asarray(mask).astype(np.uint8), (wp, hp), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def masked_mean(grid_chw: np.ndarray, mask_grid: np.ndarray) -> np.ndarray:
    """L2-normalized mean of the foreground patch features.

    ``grid_chw`` is the dense feature grid ``(C, Hp, Wp)``; ``mask_grid`` is the ``(Hp, Wp)``
    boolean foreground on that grid. If the mask selects no patch (thinner than a patch) the
    mean is taken over the whole grid, matching how :mod:`foveate.reid` degrades -- a weak
    descriptor beats an empty one.
    """
    c = grid_chw.shape[0]
    flat = grid_chw.reshape(c, -1).T  # (Hp*Wp, C)
    sel = np.asarray(mask_grid).reshape(-1).astype(bool)
    patches = flat[sel] if sel.any() else flat
    return l2_normalize(patches.mean(axis=0))
