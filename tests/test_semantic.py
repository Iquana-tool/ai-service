"""The generic semantic -> instance adapter (models/semantic.py)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from models.base import CapabilityModel
from models.semantic import (
    SemanticClass,
    SemanticInstanceSegmentation,
    component_contours,
    one_hot,
    semantic_to_instances,
)

CELL = SemanticClass("cell", "Cell")
NUCLEUS = SemanticClass("nucleus", "Nucleus", part_of="cell")
NUCLEOLUS = SemanticClass("nucleolus", "Nucleolus", part_of="nucleus")
LANDMARK = SemanticClass("landmark", "Landmark", 0.1, single_instance=True)


def _blob(scores: np.ndarray, y0: int, y1: int, x0: int, x1: int, value: float) -> None:
    scores[y0:y1, x0:x1] = value


def test_components_become_instances_and_speckle_is_dropped():
    scores = np.zeros((40, 40), dtype=np.float32)
    _blob(scores, 2, 12, 2, 12, 0.9)
    _blob(scores, 20, 30, 20, 30, 0.6)
    _blob(scores, 35, 37, 35, 37, 0.9)  # 4 px
    _, contours = component_contours(scores, 0.5, min_area=10, added_by="m", model_class="cell")
    contours = list(contours.values())
    assert len(contours) == 2
    assert sorted(round(c.confidence, 2) for c in contours) == [0.6, 0.9]
    assert all(0.0 <= v <= 1.0 for c in contours for v in c.x + c.y)
    assert {(c.added_by, c.model_class) for c in contours} == {("m", "cell")}


def test_single_instance_keeps_the_strongest_component_and_reports_its_peak():
    scores = np.zeros((40, 40), dtype=np.float32)
    _blob(scores, 2, 20, 2, 20, 0.3)
    _blob(scores, 25, 30, 25, 30, 0.6)
    scores[27, 27] = 0.95
    _, contours = component_contours(scores, 0.1, min_area=0, single_instance=True)
    (only,) = contours.values()
    assert only.confidence == pytest.approx(0.95)


def test_part_of_nests_children_in_the_containing_parent():
    scores = np.zeros((2, 60, 60), dtype=np.float32)
    _blob(scores[0], 0, 30, 0, 60, 0.9)      # two cells: top band ...
    _blob(scores[0], 40, 60, 0, 60, 0.9)     # ... and bottom band
    _blob(scores[1], 5, 15, 5, 15, 0.9)      # nucleus in the top cell
    _blob(scores[1], 45, 55, 5, 15, 0.9)     # nucleus in the bottom cell
    _blob(scores[1], 31, 39, 30, 40, 0.9)    # nucleus between the cells: stays top-level
    top, bottom, loose = semantic_to_instances(scores, (CELL, NUCLEUS), min_area=0)
    assert (top.model_class, bottom.model_class, loose.model_class) == ("cell", "cell", "nucleus")
    assert [len(top.children), len(bottom.children), len(loose.children)] == [1, 1, 0]
    assert top.children[0].model_class == "nucleus"


def test_nesting_chains_through_several_levels_in_any_class_order():
    scores = np.zeros((3, 40, 40), dtype=np.float32)
    _blob(scores[0], 0, 40, 0, 40, 0.9)
    _blob(scores[1], 10, 30, 10, 30, 0.9)
    _blob(scores[2], 15, 20, 15, 20, 0.9)
    (cell,) = semantic_to_instances(scores[::-1].copy(), (NUCLEOLUS, NUCLEUS, CELL), min_area=0)
    (nucleus,) = cell.children
    (nucleolus,) = nucleus.children
    assert (cell.model_class, nucleus.model_class, nucleolus.model_class) == ("cell", "nucleus", "nucleolus")


def test_thresholds_override_per_class():
    scores = np.zeros((1, 20, 20), dtype=np.float32)
    _blob(scores[0], 0, 5, 0, 5, 0.6)
    assert len(semantic_to_instances(scores, (CELL,), min_area=0)) == 1
    assert semantic_to_instances(scores, (CELL,), thresholds={"cell": 0.7}, min_area=0) == []


def test_scores_must_match_the_classes():
    with pytest.raises(ValueError, match="Expected scores of shape"):
        semantic_to_instances(np.zeros((2, 5, 5)), (CELL,))


def test_one_hot_turns_label_maps_into_scores():
    scores = one_hot(np.array([[0, 1], [2, 1]]), 3)
    assert scores.shape == (3, 2, 2)
    assert scores[1].tolist() == [[0, 1], [0, 1]]


class _Toy(SemanticInstanceSegmentation, CapabilityModel):
    """The smallest possible wrapped semantic model: an argmax over two intensity bands."""

    semantic_classes = (CELL, LANDMARK)
    model_info = InstanceSegmentationModelInfo(
        registry_key="toy", name="Toy", description="d", usage_tip=None,
    )

    def class_scores(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return one_hot((gray > 100).astype(int) + (gray > 200).astype(int), 3)[1:]


def test_wrapping_declares_label_space_and_contract():
    info = _Toy.model_info
    assert [(c.key, c.name) for c in info.classes] == [("cell", "Cell"), ("landmark", "Landmark")]
    (contract,) = info.input_contracts
    assert [p.key for p in contract.parameters] == ["cell_threshold", "landmark_threshold", "min_area"]
    assert info.tags["task_instance_segmentation"] == "true"


def test_wrapped_model_predicts_through_the_capability_interface(tmp_path):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = 150
    image[25:35, 25:35] = 250
    path = tmp_path / "toy.png"
    cv2.imwrite(str(path), image)
    request = InstanceSegmentationRequest(image_url=str(path), user_id="u", model_registry_key="toy")
    contours = _Toy().predict(None, request, {"task": "instance-segmentation", "min_area": 4})
    assert sorted(c.model_class for c in contours) == ["cell", "landmark"]
