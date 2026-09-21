"""Tests for hierarchy-aware SAM 3: the pure geometry (models.hierarchy_ops) and the
per-parent standard-vs-concat dispatch in SAM3Hierarchical, with the SAM 3 forward stubbed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from models import hierarchy_ops as ho


def _rect(hw, y0, y1, x0, x1):
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# -- geometry ------------------------------------------------------------------ #
def test_containment_and_assignment():
    parents = [_rect((100, 100), 0, 50, 0, 50), _rect((100, 100), 0, 50, 50, 100)]
    inside_second = _rect((100, 100), 10, 20, 60, 70)
    straddling = _rect((100, 100), 10, 20, 45, 55)  # half on each -> first match at 0.5
    outside = _rect((100, 100), 80, 90, 80, 90)

    assert ho.containment(inside_second, parents[1]) == 1.0
    assert ho.containment(np.zeros((100, 100)), parents[0]) == 0.0
    assert ho.assign_to_regions([inside_second, straddling, outside], parents) == [1, 0, None]


def test_focus_crop_pads_and_blurs_only_outside_parent():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[::2] = 255  # stripes: blur visibly changes them
    parent = _rect((100, 100), 20, 60, 30, 70)

    crop = ho.focus_crop(image, parent, padding=0.25, blur=0.05)
    assert crop.box == (10, 70, 20, 80)
    assert crop.image.shape == (60, 60, 3)
    assert crop.parent.shape == (60, 60) and crop.parent[10:50, 10:50].all()
    # Inside the parent the pixels are untouched; outside they are smoothed.
    assert np.array_equal(crop.image[crop.parent], image[10:70, 20:80][crop.parent])
    assert not np.array_equal(crop.image[~crop.parent], image[10:70, 20:80][~crop.parent])

    sharp = ho.focus_crop(image, parent, padding=0.25, blur=0.0)
    assert np.array_equal(sharp.image, image[10:70, 20:80])


def test_bbox_xyxy_shifts_and_handles_empty():
    assert ho.bbox_xyxy(_rect((10, 10), 2, 4, 3, 6), x_off=10, y_off=1) == [13.0, 3.0, 16.0, 5.0]
    assert ho.bbox_xyxy(np.zeros((10, 10))) is None


def test_paste_back_clips_filters_and_places():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    parent = _rect((100, 100), 20, 60, 30, 70)
    crop = ho.focus_crop(image, parent, padding=0.25, blur=0.0)  # box (10, 70, 20, 80)
    ch, cw = crop.parent.shape

    child = _rect((ch, cw), 12, 18, 12, 18)          # fully on the parent
    edge = _rect((ch, cw), 5, 15, 12, 18)            # 50% on the parent -> clipped
    background = _rect((ch, cw), 0, 8, 0, 8)         # entirely on the blurred margin
    whole = crop.parent.copy()                       # SAM 3 re-segmenting the parent

    masks, kept = ho.paste_back([child, edge, background, whole], crop, (100, 100))
    assert kept == [0, 1]
    assert masks[0].shape == (100, 100)
    assert np.array_equal(masks[0], _rect((100, 100), 22, 28, 32, 38))
    assert np.array_equal(masks[1], _rect((100, 100), 20, 25, 32, 38))  # clipped to the parent


# -- dispatch (SAM 3 stubbed) -------------------------------------------------- #
torch = pytest.importorskip("torch")
from models.sam3_hierarchical import SAM3Hierarchical  # noqa: E402


class _StubSAM3(SAM3Hierarchical):
    """Records each forward and 'detects' a fixed square at the first prompt box."""

    def __init__(self):  # no weights
        self.threshold, self.mask_threshold = 0.3, 0.5
        self.calls = []

    def _forward(self, image, boxes, labels, text, threshold, mask_threshold):
        self.calls.append(SimpleNamespace(shape=image.shape[:2], boxes=boxes, labels=labels))
        h, w = image.shape[:2]
        # One detection in the top-left corner of the (target part of the) input.
        det = np.zeros((1, h, w), dtype=bool)
        det[0, 2:6, 2:6] = True
        return det, np.array([0.9])


def _request(image, positives, parents, negatives=()):
    return SimpleNamespace(
        image=image, concept=None,
        positive_exemplar_masks=list(positives),
        negative_exemplar_masks=list(negatives),
        parent_region_masks=list(parents),
    )


def test_one_pass_per_parent_standard_where_exemplar_lies_concat_elsewhere():
    hw = (200, 200)
    image = np.zeros((*hw, 3), dtype=np.uint8)
    parent_a = _rect(hw, 0, 60, 0, 60)
    parent_b = _rect(hw, 100, 160, 100, 160)
    polyp = _rect(hw, 20, 30, 20, 30)  # the one exemplar, on parent A

    model = _StubSAM3()
    masks, scores = model.suggest_instances(
        _request(image, [polyp], [parent_a, parent_b]), {"crop_padding": 0.0, "background_blur": 0.0}
    )

    assert len(model.calls) == 2
    standard, concat = model.calls
    # Parent A: its own crop only, prompted with the exemplar in crop coordinates.
    assert standard.shape == (60, 60)
    assert standard.boxes == [[20.0, 20.0, 30.0, 30.0]] and standard.labels == [1]
    # Parent B: its crop with parent A's crop pasted to the right; box shifted onto that tile.
    assert concat.shape == (60, 120)
    assert concat.boxes == [[80.0, 20.0, 90.0, 30.0]]

    # Both detections are mapped back into their parent in full-image coordinates.
    assert masks.shape == (2, *hw) and masks.dtype == np.uint8
    assert np.array_equal(masks[0].astype(bool), _rect(hw, 2, 6, 2, 6))
    assert np.array_equal(masks[1].astype(bool), _rect(hw, 102, 106, 102, 106))
    assert scores.tolist() == [0.9, 0.9]


def test_without_parents_runs_once_on_the_whole_image():
    hw = (50, 80)
    model = _StubSAM3()
    masks, _ = model.suggest_instances(
        _request(np.zeros((*hw, 3), np.uint8), [_rect(hw, 10, 20, 10, 20)], []), {}
    )
    assert [c.shape for c in model.calls] == [hw]
    assert model.calls[0].boxes == [[10.0, 10.0, 20.0, 20.0]]
    assert masks.shape == (1, *hw)


def test_orphan_exemplar_becomes_the_concat_reference():
    hw = (200, 200)
    parent = _rect(hw, 100, 160, 100, 160)
    orphan = _rect(hw, 20, 30, 20, 30)  # on no parent

    model = _StubSAM3()
    model.suggest_instances(
        _request(np.zeros((*hw, 3), np.uint8), [orphan], [parent]), {"crop_padding": 0.0}
    )
    (call,) = model.calls
    # Reference = orphan bbox padded by its own size (10 px each side) -> 30x30 tile at x=60.
    assert call.shape == (60, 90)
    assert call.boxes == [[70.0, 10.0, 80.0, 20.0]]


def test_model_advertises_instance_suggestion_only():
    tasks = [t.name for t in SAM3Hierarchical.supported_tasks()]
    assert tasks == ["instance-suggestion"]
    assert SAM3Hierarchical.model_info.registry_key == "sam3_hierarchical"
