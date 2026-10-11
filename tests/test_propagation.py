"""Propagation (#121): the task, SAM 2's handler, and its behaviour on a real scan.

The loop is checked against a stand-in for the video model, so these run without
weights: which frames get results, that drawn frames are left alone, the range,
the direction, early stopping once the object is gone, and which pass wins
between two drawn frames.

``test_real_oct_volume`` runs SAM 2.1 Tiny on real slices when
``IQUANA_TEST_OCT_FRAMES`` points at a folder of ``frame_*.png`` -- patient scans
do not belong in the repository.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.networking.http.services import (
    FrameMask,
    PropagationObject,
    PropagationRequest,
)

from models.base import PROPAGATION, get_task
from models.sam2 import SAM2Prompted

H, W = 8, 10


def _mask(value=1):
    array = np.zeros((H, W), dtype=np.uint8)
    array[2:5, 3:7] = value
    return BinaryMask.from_numpy_array(array)


def _object(*frames, object_id=1):
    return PropagationObject(object_id=object_id,
                             frame_masks=[FrameMask(frame_index=f, mask=_mask()) for f in frames])


class FakeVideo:
    """Stands in for Sam2VideoProcessor + Sam2VideoModel.

    ``presence[i]`` is the object score on session frame ``i``; the iterator walks
    from the start frame to the end of the session (or back to 0 in reverse), as
    the real one does.
    """

    def __init__(self, presence):
        self.presence = presence
        self.conditioned = []
        self.passes = []

    # processor
    def init_video_session(self, video, **kwargs):
        return SimpleNamespace(num_frames=len(video))

    def add_inputs_to_inference_session(self, session, frame_idx, obj_ids, input_masks):
        self.conditioned.append(frame_idx)

    def post_process_masks(self, masks, original_sizes, binarize=True):
        return [torch.ones((1, 1, *original_sizes[0]), dtype=torch.bool)]

    # model
    def __call__(self, inference_session, frame_idx):
        return None

    def propagate_in_video_iterator(self, session, start_frame_idx, reverse=False):
        self.passes.append((start_frame_idx, reverse))
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, session.num_frames)
        for index in order:
            logit = torch.logit(torch.tensor([self.presence[index]]).clamp(1e-4, 1 - 1e-4))
            yield SimpleNamespace(frame_idx=index, pred_masks=torch.zeros(1, 1, 4, 4),
                                  object_score_logits=logit, object_ids=[1])


def _model(presence):
    model = SAM2Prompted("sam2-1-tiny", device="cpu")
    fake = FakeVideo(presence)
    model.video_processor = fake
    model.video_model = fake
    return model, fake


def _run(model, obj, frames=None, start=0, end=None, direction="both", threshold=0.5):
    frames = frames or [np.zeros((H, W, 3), dtype=np.uint8)] * len(model.video_model.presence)
    end = len(frames) - 1 if end is None else end
    return model._propagate_object(frames, obj, start, end, direction, threshold)


# --------------------------------------------------------------------------- #
# The task and how SAM 2 declares it
# --------------------------------------------------------------------------- #
def test_propagation_task_is_registered():
    assert get_task("propagation") is PROPAGATION
    assert PROPAGATION.request_type is PropagationRequest


def test_sam2_serves_propagation_but_stays_a_prompted_model():
    model = SAM2Prompted("sam2-1-tiny", device="cpu")
    tags = model.model_info.tags
    assert tags["task"] == "prompted-segmentation"
    assert tags["task_propagation"] == "true"
    contract = model.get_input_contract("propagation")
    assert [p.key for p in contract.parameters] == ["presence_threshold"]


def test_prompted_use_does_not_load_the_video_model():
    model = SAM2Prompted("sam2-1-tiny", device="cpu")
    model.model, model.processor = object(), object()
    loads = []
    model._load_weights = lambda: loads.append(1)
    model._ensure_weights()
    assert loads == []


def test_video_weights_stay_out_of_the_pickle():
    model, _ = _model([1.0])
    state = pickle.loads(pickle.dumps(model)).__dict__
    assert "video_model" not in state and "video_processor" not in state


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def test_fills_every_frame_but_the_drawn_one():
    model, fake = _model([0.9] * 5)
    results = _run(model, _object(2))
    assert [r.frame_index for r in results] == [0, 1, 3, 4]
    assert all(r.mask is not None and r.object_id == 1 for r in results)
    assert fake.conditioned == [2]
    assert fake.passes == [(2, False), (2, True)]


def test_respects_range_and_direction():
    model, fake = _model([0.9] * 6)
    assert [r.frame_index for r in _run(model, _object(2), start=1, end=4)] == [1, 3, 4]
    model, fake = _model([0.9] * 6)
    assert [r.frame_index for r in _run(model, _object(2), direction="forward")] == [3, 4, 5]
    assert fake.passes == [(2, False)]


def test_stops_after_two_absent_frames_and_reports_them_empty():
    model, _ = _model([0.9, 0.9, 0.9, 0.2, 0.1, 0.9, 0.9])
    results = _run(model, _object(0), direction="forward")
    assert [(r.frame_index, r.mask is None) for r in results] == [(1, False), (2, False), (3, True), (4, True)]
    assert results[-1].presence == pytest.approx(0.1, abs=1e-3)


def test_one_low_frame_does_not_end_it():
    model, _ = _model([0.9, 0.3, 0.9, 0.9])
    results = _run(model, _object(0), direction="forward")
    assert [r.frame_index for r in results if r.mask is not None] == [2, 3]


def test_several_drawn_frames_condition_one_session():
    model, fake = _model([0.9] * 6)
    results = _run(model, _object(1, 4))
    assert fake.conditioned == [1, 4]
    assert fake.passes == [(1, False), (4, True)]
    assert [r.frame_index for r in results] == [0, 2, 3, 5]


def test_drawn_frame_outside_the_range_still_conditions():
    model, fake = _model([0.9] * 6)
    results = _run(model, _object(5), start=0, end=2, direction="backward")
    assert fake.conditioned == [5]
    assert [r.frame_index for r in results] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# A real scan
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("IQUANA_TEST_OCT_FRAMES"), reason="needs IQUANA_TEST_OCT_FRAMES")
def test_real_oct_volume():
    """The retina band, drawn on the middle slice, reaches every other slice."""
    import cv2

    paths = sorted(Path(os.environ["IQUANA_TEST_OCT_FRAMES"]).glob("frame_*.png"))
    middle = len(paths) // 2
    image = cv2.imread(str(paths[middle]))
    model = SAM2Prompted("sam2-1-tiny")
    model._ensure_weights()
    from iquana_toolbox.schemas.prompts import BoxPrompt, Prompts
    height, width = image.shape[:2]
    seed, score = model._segment(image, Prompts(box_prompt=BoxPrompt(
        min_x=0.0, min_y=65 / height, max_x=1.0, max_y=190 / height)))
    assert score > 0.8

    request = PropagationRequest(
        image_url=str(paths[middle]), user_id="test", model_registry_key="sam2-1-tiny",
        frame_urls=[str(p) for p in paths],
        objects=[PropagationObject(object_id=1, frame_masks=[
            FrameMask(frame_index=middle, mask=BinaryMask.from_numpy_array((seed > 0).astype(np.uint8)))])],
    )
    results = model.propagate(request, {"presence_threshold": 0.5})
    assert len(results) == len(paths) - 1
    assert all(r.mask is not None and r.presence > 0.9 for r in results)
    areas = [int(r.mask.mask.sum()) for r in results]
    seed_area = int((seed > 0).sum())
    assert all(0.7 * seed_area < area < 1.3 * seed_area for area in areas)
