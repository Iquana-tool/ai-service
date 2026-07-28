"""Unit tests for the pure concat geometry (models.concat_ops).

NumPy-only -- no torch, no SAM 3 weights, no HuggingFace -- so the layout, compositing, box
shifting, and target-side extraction behind the cross-image concat trick are verified without
the model.
"""
import numpy as np
import pytest

from models import concat_ops as cc


def test_plan_layout_dimensions_and_offsets():
    plan = cc.plan_layout((10, 20), [(4, 6), (8, 3)])
    # Target at origin; exemplars stack in a column to its right (x = target_w = 20).
    assert plan.target_xywh == (0, 0, 20, 10)
    assert plan.exemplar_xywh == [(20, 0, 6, 4), (20, 4, 3, 8)]
    # Width = target_w + widest exemplar; height = max(target_h, stacked exemplar height).
    assert plan.canvas_w == 20 + 6
    assert plan.canvas_h == max(10, 4 + 8)


def test_plan_layout_no_exemplars():
    plan = cc.plan_layout((7, 9), [])
    assert plan.canvas_h == 7 and plan.canvas_w == 9
    assert plan.exemplar_xywh == []


def test_composite_places_images_at_their_tiles():
    target = np.full((10, 20, 3), 100, dtype=np.uint8)
    ex0 = np.full((4, 6, 3), 200, dtype=np.uint8)
    ex1 = np.full((8, 3, 3), 50, dtype=np.uint8)
    plan = cc.plan_layout((10, 20), [(4, 6), (8, 3)])

    canvas = cc.composite_image(target, [ex0, ex1], plan, fill=0)
    assert canvas.shape == (12, 26, 3)
    assert (canvas[0:10, 0:20] == 100).all()   # target block
    assert (canvas[0:4, 20:26] == 200).all()   # exemplar 0
    assert (canvas[4:12, 20:23] == 50).all()   # exemplar 1
    # A gap the tiles don't cover keeps the fill value (target is 10 tall, canvas 12).
    assert (canvas[10:12, 0:20] == 0).all()


def test_exemplar_boxes_shifted_onto_canvas():
    plan = cc.plan_layout((10, 20), [(4, 6)])  # exemplar tile at (x=20, y=0)
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True  # local bbox: y0=1,y1=3,x0=2,x1=5

    boxes = cc.exemplar_boxes_on_canvas([mask], plan)
    # Shifted by the tile offset (20, 0): [xmin, ymin, xmax, ymax].
    assert boxes == [[22.0, 1.0, 25.0, 3.0]]


def test_extract_target_masks_keeps_target_side_and_crops():
    plan = cc.plan_layout((10, 20), [(10, 6)])  # canvas 10x26, target rect (0,0,20,10)

    # Mask A: entirely on the target side.
    a = np.zeros((10, 26), dtype=bool)
    a[2:5, 3:8] = True
    # Mask B: entirely on the exemplar tile (x >= 20) -> dropped.
    b = np.zeros((10, 26), dtype=bool)
    b[1:4, 21:24] = True
    # Mask C: split -- 6 px on target, 6 px on exemplar -> exactly 0.5, kept at default frac.
    c = np.zeros((10, 26), dtype=bool)
    c[0, 17:23] = True

    kept, idx = cc.extract_target_masks(np.stack([a, b, c]), plan.target_xywh, min_target_frac=0.5)
    assert idx == [0, 2]                       # A and C kept, B dropped
    assert kept[0].shape == (10, 20)           # cropped to target size
    assert kept[0][2:5, 3:8].all()
    assert int(kept[1].sum()) == 3             # only the 3 target-side pixels of C survive the crop


def test_extract_target_masks_strict_fraction_drops_split():
    plan = cc.plan_layout((10, 20), [(10, 6)])
    c = np.zeros((10, 26), dtype=bool)
    c[0, 17:23] = True  # half on target, half off
    kept, idx = cc.extract_target_masks(np.stack([c]), plan.target_xywh, min_target_frac=0.75)
    assert kept == [] and idx == []


def test_extract_target_masks_skips_empty():
    plan = cc.plan_layout((5, 5), [])
    empty = np.zeros((1, 5, 5), dtype=bool)
    kept, idx = cc.extract_target_masks(empty, plan.target_xywh)
    assert kept == [] and idx == []
