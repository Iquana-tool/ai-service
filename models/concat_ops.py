"""Pure array ops for the SAM 3 cross-image concat trick -- no torch, no model, no HF.

SAM 3 segments a *concept* only within one image. To transfer a concept from annotated
exemplars (in other images) onto a new target image, we composite the exemplar image(s)
beside the target on one canvas, give SAM 3 the exemplars' boxes as positive visual prompts,
run it over the whole canvas, then keep only the detections that land on the target region
and map them back to target coordinates.

This module is the geometry half of that -- laying out the canvas, pasting images, shifting
exemplar boxes onto the canvas, and pulling target-side masks back out -- kept free of torch
and gated weights so it is unit-testable. The SAM 3 forward pass lives in
:mod:`models.sam3`.

Layout: the target sits at the canvas origin ``(0, 0)``; exemplars stack in a column to its
right. So "on the target side" is simply "inside the ``target_xywh`` rectangle", and mapping a
target-side mask back to the target image is a crop of that rectangle -- no scaling, because
everything is pasted at native resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.embedding_ops import mask_bbox


@dataclass(frozen=True)
class LayoutPlan:
    """Where each image sits on the composite canvas.

    ``target_xywh`` and each ``exemplar_xywh`` are ``(x, y, w, h)`` in canvas pixels.
    """

    canvas_h: int
    canvas_w: int
    target_xywh: tuple[int, int, int, int]
    exemplar_xywh: list[tuple[int, int, int, int]]


def plan_layout(
    target_hw: tuple[int, int], exemplar_hws: list[tuple[int, int]]
) -> LayoutPlan:
    """Place the target at the origin and stack the exemplars in a column to its right."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    col_x = tw
    y = 0
    exemplar_xywh: list[tuple[int, int, int, int]] = []
    for eh, ew in exemplar_hws:
        exemplar_xywh.append((col_x, y, int(ew), int(eh)))
        y += int(eh)
    col_h = y
    col_w = max((ew for _, ew in exemplar_hws), default=0)
    canvas_h = max(th, col_h)
    canvas_w = tw + col_w
    return LayoutPlan(canvas_h=canvas_h, canvas_w=canvas_w,
                      target_xywh=(0, 0, tw, th), exemplar_xywh=exemplar_xywh)


def composite_image(
    target: np.ndarray,
    exemplars: list[np.ndarray],
    plan: LayoutPlan,
    fill: int = 0,
) -> np.ndarray:
    """Paste the target and exemplars onto one canvas per ``plan``; gaps stay ``fill``."""
    channels = target.shape[2] if target.ndim == 3 else 1
    shape = (plan.canvas_h, plan.canvas_w, channels) if target.ndim == 3 else (plan.canvas_h, plan.canvas_w)
    canvas = np.full(shape, fill, dtype=target.dtype)

    tx, ty, tw, th = plan.target_xywh
    canvas[ty:ty + th, tx:tx + tw] = target
    for img, (x, y, w, h) in zip(exemplars, plan.exemplar_xywh):
        canvas[y:y + h, x:x + w] = img
    return canvas


def exemplar_boxes_on_canvas(
    exemplar_masks: list[np.ndarray], plan: LayoutPlan
) -> list[list[float]]:
    """Canvas-space ``[xmin, ymin, xmax, ymax]`` box for each exemplar's mask.

    The box is the exemplar mask's bbox shifted by that exemplar's tile offset -- the positive
    visual prompt SAM 3 uses to push the concept.
    """
    boxes: list[list[float]] = []
    for mask, (x_off, y_off, _, _) in zip(exemplar_masks, plan.exemplar_xywh):
        y0, y1, x0, x1 = mask_bbox(mask)
        boxes.append([float(x0 + x_off), float(y0 + y_off), float(x1 + x_off), float(y1 + y_off)])
    return boxes


def extract_target_masks(
    canvas_masks: np.ndarray,
    target_xywh: tuple[int, int, int, int],
    min_target_frac: float = 0.5,
) -> tuple[list[np.ndarray], list[int]]:
    """Keep the detections that lie mostly on the target, cropped back to target size.

    ``canvas_masks`` is ``(N, canvas_h, canvas_w)`` boolean. A detection is kept when at least
    ``min_target_frac`` of its foreground area falls inside ``target_xywh`` -- this drops the
    echo SAM 3 produces back on the exemplar tile while keeping genuine target-side instances.
    Returns the cropped ``(target_h, target_w)`` masks and their indices into ``canvas_masks``.
    """
    tx, ty, tw, th = target_xywh
    kept: list[np.ndarray] = []
    indices: list[int] = []
    for i, mask in enumerate(canvas_masks):
        mask = np.asarray(mask).astype(bool)
        total = int(mask.sum())
        if total == 0:
            continue
        sub = mask[ty:ty + th, tx:tx + tw]
        if int(sub.sum()) / total >= min_target_frac:
            kept.append(sub.copy())
            indices.append(i)
    return kept, indices
