"""Focused tests for flat Mask2Former validation splits and mask metrics."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from models import mask2former as mask2former_module
from models.mask2former import (
    Mask2Former,
    compute_instance_ap_metrics,
    compute_mask_metrics,
)
from models.mask2former_dataset import CocoSample, LabelMapping, split_dataset_indices


def test_split_is_deterministic_disjoint_and_keeps_a_training_image() -> None:
    first = split_dataset_indices(10, seed=17)
    second = split_dataset_indices(10, seed=17)

    assert first == second
    train_indices, validation_indices = first
    assert set(train_indices).isdisjoint(validation_indices)
    assert sorted(train_indices + validation_indices) == list(range(10))
    assert len(validation_indices) == 2


def test_small_datasets_train_on_all_without_validation() -> None:
    assert split_dataset_indices(0) == ([], [])
    assert split_dataset_indices(1) == ([0], [])
    train_indices, validation_indices = split_dataset_indices(2)
    assert len(train_indices) == 1
    assert len(validation_indices) == 1


def test_perfect_masks_have_one_iou_f1_precision_and_recall() -> None:
    mask = np.array([[True, False], [True, True]])

    metrics = compute_mask_metrics([{7: mask}], [{7: mask.copy()}])

    assert metrics["val_mask_iou_label_7"] == 1.0
    assert metrics["val_mask_f1_label_7"] == 1.0
    assert metrics["val_mask_precision_label_7"] == 1.0
    assert metrics["val_mask_recall_label_7"] == 1.0
    assert metrics["val_mask_iou_macro"] == 1.0
    assert metrics["val_mask_f1_macro"] == 1.0
    assert metrics["val_mask_precision_macro"] == 1.0
    assert metrics["val_mask_recall_macro"] == 1.0


def test_partial_and_empty_predictions_have_expected_finite_values() -> None:
    target = np.array([[True, True], [False, False]])
    partial_prediction = np.array([[True, False], [False, False]])
    empty_prediction = np.zeros_like(target)

    partial = compute_mask_metrics([{7: partial_prediction}], [{7: target}])
    empty = compute_mask_metrics([{7: empty_prediction}], [{7: target}])

    assert partial["val_mask_iou_label_7"] == 0.5
    assert partial["val_mask_f1_label_7"] == 2 / 3
    assert partial["val_mask_precision_label_7"] == 1.0
    assert partial["val_mask_recall_label_7"] == 0.5
    assert empty["val_mask_iou_label_7"] == 0.0
    assert empty["val_mask_f1_label_7"] == 0.0
    assert empty["val_mask_precision_label_7"] == 0.0
    assert empty["val_mask_recall_label_7"] == 0.0


def test_absent_ground_truth_labels_do_not_inflate_macro_or_emit_nonfinite_values() -> None:
    target = np.array([[True, False], [False, False]])
    prediction = np.array([[True, False], [False, False]])
    predicted_only = np.array([[False, True], [False, False]])
    absent = np.zeros_like(target)

    metrics = compute_mask_metrics(
        [{7: prediction, 42: predicted_only, 99: absent}],
        [{7: target, 42: absent, 99: absent}],
    )

    assert metrics["val_mask_iou_label_7"] == 1.0
    assert metrics["val_mask_f1_label_7"] == 1.0
    assert metrics["val_mask_precision_label_7"] == 1.0
    assert metrics["val_mask_recall_label_7"] == 1.0
    assert metrics["val_mask_iou_label_42"] == 0.0
    assert metrics["val_mask_f1_label_42"] == 0.0
    assert metrics["val_mask_precision_label_42"] == 0.0
    assert metrics["val_mask_recall_label_42"] == 0.0
    assert "val_mask_iou_label_99" not in metrics
    assert "val_mask_f1_label_99" not in metrics
    assert "val_mask_precision_label_99" not in metrics
    assert "val_mask_recall_label_99" not in metrics
    assert metrics["val_mask_iou_macro"] == 1.0
    assert metrics["val_mask_f1_macro"] == 1.0
    assert metrics["val_mask_precision_macro"] == 1.0
    assert metrics["val_mask_recall_macro"] == 1.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_validation_uses_thresholded_instances_and_unions_database_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation must score decoded instances, not a semantic argmax map."""

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def __call__(self, **_inputs: object) -> object:
            return object()

    class FakeProcessor:
        def __init__(self) -> None:
            self.thresholds: list[float] = []
            self.semantic_calls = 0
            self._segmentation = np.array(
                [
                    [10, 10, 0, 0],
                    [10, 11, 0, 0],
                    [0, 11, 13, 13],
                ]
            )
            self._segments = [
                {"id": 10, "label_id": 0, "score": 0.5},
                {"id": 11, "label_id": 0, "score": 0.75},
                {"id": 12, "label_id": 1, "score": 0.49},
                {"id": 13, "label_id": 1, "score": 0.8},
            ]

        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {}

        def post_process_semantic_segmentation(
            self, *_args: object, **_kwargs: object
        ) -> None:
            self.semantic_calls += 1
            raise AssertionError("validation must not use semantic post-processing")

        def post_process_instance_segmentation(
            self,
            _outputs: object,
            *,
            target_sizes: list[tuple[int, int]],
            threshold: float,
            return_binary_maps: bool = False,
        ) -> list[dict[str, object]]:
            assert target_sizes == [(3, 4)]
            self.thresholds.append(threshold)
            selected_segments = [
                segment for segment in self._segments if segment["score"] >= threshold
            ]
            if return_binary_maps:
                segmentation = np.stack([
                    self._segmentation == segment["id"] for segment in selected_segments
                ])
                selected_segments = [
                    {**segment, "id": index}
                    for index, segment in enumerate(selected_segments)
                ]
            else:
                segmentation = self._segmentation
            return [
                {
                    "segmentation": segmentation,
                    "segments_info": selected_segments,
                }
            ]

    label_mapping = LabelMapping(
        database_to_model={7: 0, 42: 1},
        model_to_database={0: 7, 1: 42},
        id2label={0: "cell", 1: "core"},
    )
    sample = CocoSample(
        image=np.zeros((3, 4, 3), dtype=np.uint8),
        instance_mask=np.array(
            [
                [1, 1, 255, 255],
                [2, 2, 255, 255],
                [255, 255, 3, 255],
            ],
            dtype=np.int32,
        ),
        instance_id_to_semantic_id={1: 0, 2: 0, 3: 1},
    )
    processor = FakeProcessor()
    model = Mask2Former(checkpoint="unused")
    model._model = FakeModel()
    model._processor = processor
    model._label_mapping = label_mapping

    captured: dict[str, list[dict[int, np.ndarray]]] = {}

    def capture_metrics(
        predicted_masks: list[dict[int, np.ndarray]],
        target_masks: list[dict[int, np.ndarray]],
    ) -> dict[str, float]:
        captured["predicted"] = predicted_masks
        captured["target"] = target_masks
        return compute_mask_metrics(predicted_masks, target_masks)

    monkeypatch.setattr(mask2former_module, "compute_mask_metrics", capture_metrics)
    validation_metrics = model._evaluate_validation(
        [[sample]], label_mapping, is_cancelled=lambda: False
    )

    predicted = captured["predicted"][0]
    target = captured["target"][0]
    assert processor.semantic_calls == 0
    assert processor.thresholds == [0.5, 0.0]
    assert processor._segments[2]["score"] < processor.thresholds[0]
    assert set(predicted) == {7, 42}
    assert np.array_equal(
        predicted[7],
        np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, True, False, False],
            ]
        ),
    )
    assert not predicted[7][processor._segmentation == 0].any()
    assert not predicted[42][processor._segmentation == 0].any()
    assert np.array_equal(
        target[7],
        np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
            ]
        ),
    )
    assert np.array_equal(
        target[42],
        np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, False],
            ]
        ),
    )

    assert validation_metrics["val_mask_iou_label_7"] == 4 / 5
    assert validation_metrics["val_mask_f1_label_7"] == 8 / 9
    assert validation_metrics["val_mask_precision_label_7"] == 4 / 5
    assert validation_metrics["val_mask_recall_label_7"] == 1.0
    assert validation_metrics["val_mask_iou_label_42"] == 1 / 2
    assert validation_metrics["val_mask_f1_label_42"] == 2 / 3
    assert validation_metrics["val_mask_precision_label_42"] == 1 / 2
    assert validation_metrics["val_mask_recall_label_42"] == 1.0
    assert validation_metrics["val_mask_iou_macro"] == pytest.approx(13 / 20)
    assert validation_metrics["val_mask_f1_macro"] == pytest.approx(7 / 9)
    assert validation_metrics["val_mask_precision_macro"] == pytest.approx(13 / 20)
    assert validation_metrics["val_mask_recall_macro"] == 1.0

    inference_thresholds: list[float] = []
    original_instance_post_process = processor.post_process_instance_segmentation

    def record_inference_threshold(
        outputs: object,
        *,
        target_sizes: list[tuple[int, int]],
        threshold: float,
        return_binary_maps: bool = False,
    ) -> list[dict[str, object]]:
        inference_thresholds.append(threshold)
        return original_instance_post_process(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            return_binary_maps=return_binary_maps,
        )

    processor.post_process_instance_segmentation = (  # type: ignore[method-assign]
        record_inference_threshold
    )
    model.segment_instances(
        SimpleNamespace(
            image=np.zeros((3, 4, 3), dtype=np.uint8),
            label=None,
            model_registry_key="test",
        )
    )
    assert inference_thresholds == [0.5]
    assert processor.thresholds == [0.5, 0.0, 0.5]


def test_validation_instance_masks_handle_empty_results() -> None:
    label_mapping = LabelMapping(
        database_to_model={7: 0},
        model_to_database={0: 7},
        id2label={0: "coral"},
    )

    assert Mask2Former._instance_masks_by_label(
        {"segmentation": None, "segments_info": []}, label_mapping
    ) == {}
    assert Mask2Former._instance_masks_by_label(
        {"segmentation": np.zeros((2, 2), dtype=np.int64), "segments_info": []},
        label_mapping,
    ) == {}


@pytest.mark.parametrize(
    "segments_info",
    [None, [{"id": 1, "label_id": "0"}], [{"id": 0.9, "label_id": 0}]],
)
def test_validation_instance_masks_reject_malformed_segments(
    segments_info: object,
) -> None:
    label_mapping = LabelMapping(
        database_to_model={7: 0},
        model_to_database={0: 7},
        id2label={0: "coral"},
    )

    with pytest.raises(RuntimeError, match="malformed segments"):
        Mask2Former._instance_masks_by_label(
            {
                "segmentation": np.ones((2, 2), dtype=np.int64),
                "segments_info": segments_info,
            },
            label_mapping,
        )


def _rectangular_mask(
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> np.ndarray:
    mask = np.zeros((4, 4), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _instance(label_id: int, score: float, mask: np.ndarray) -> dict[str, object]:
    return {"label_id": label_id, "score": score, "mask": mask}


def test_instance_ap_ranks_predictions_by_confidence() -> None:
    target_mask = _rectangular_mask(0, 0, 2, 2)
    false_positive_mask = _rectangular_mask(2, 2, 4, 4)

    high_confidence_false_positive = compute_instance_ap_metrics(
        [[
            _instance(7, 0.2, target_mask),
            _instance(7, 0.9, false_positive_mask),
        ]],
        [[_instance(7, 1.0, target_mask)]],
    )
    high_confidence_true_positive = compute_instance_ap_metrics(
        [[
            _instance(7, 0.9, target_mask),
            _instance(7, 0.2, false_positive_mask),
        ]],
        [[_instance(7, 1.0, target_mask)]],
    )

    assert high_confidence_false_positive["val_mask_ap50"] == pytest.approx(0.5)
    assert high_confidence_true_positive["val_mask_ap50"] == pytest.approx(1.0)
    assert high_confidence_false_positive["val_mask_ap"] == pytest.approx(0.5)
    assert high_confidence_true_positive["val_mask_ap"] == pytest.approx(1.0)


def test_instance_ap_matches_same_label_targets_one_to_one() -> None:
    first_target = _rectangular_mask(0, 0, 2, 2)
    second_target = _rectangular_mask(2, 2, 4, 4)

    metrics = compute_instance_ap_metrics(
        [[
            _instance(7, 0.9, first_target),
            _instance(7, 0.8, first_target),
        ]],
        [[
            _instance(7, 1.0, first_target),
            _instance(7, 1.0, second_target),
        ]],
    )

    expected = 51 / 101
    assert metrics["val_mask_ap50"] == pytest.approx(expected)
    assert metrics["val_mask_ap75"] == pytest.approx(expected)
    assert metrics["val_mask_ap"] == pytest.approx(expected)


def test_duplicate_prediction_lowers_average_precision() -> None:
    first_target = _rectangular_mask(0, 0, 2, 2)
    second_target = _rectangular_mask(2, 2, 4, 4)
    targets = [[
        _instance(7, 1.0, first_target),
        _instance(7, 1.0, second_target),
    ]]

    perfect = compute_instance_ap_metrics(
        [[
            _instance(7, 0.95, first_target),
            _instance(7, 0.85, second_target),
        ]],
        targets,
    )
    duplicate = compute_instance_ap_metrics(
        [[
            _instance(7, 0.95, first_target),
            _instance(7, 0.90, first_target),
            _instance(7, 0.85, second_target),
        ]],
        targets,
    )

    expected_duplicate_ap = 253 / 303
    assert perfect["val_mask_ap"] == pytest.approx(1.0)
    assert duplicate["val_mask_ap50"] == pytest.approx(expected_duplicate_ap)
    assert duplicate["val_mask_ap75"] == pytest.approx(expected_duplicate_ap)
    assert duplicate["val_mask_ap"] == pytest.approx(expected_duplicate_ap)
    assert duplicate["val_mask_ap"] < perfect["val_mask_ap"]


def test_instance_ap_penalizes_wrong_label_false_positive_and_missed_object() -> None:
    label_one_target = _rectangular_mask(0, 0, 2, 2)
    label_two_target = _rectangular_mask(0, 2, 2, 4)
    false_positive = _rectangular_mask(2, 0, 4, 2)

    metrics = compute_instance_ap_metrics(
        [[
            _instance(2, 0.99, label_one_target),
            _instance(1, 0.95, false_positive),
            _instance(2, 0.80, label_two_target),
        ]],
        [[
            _instance(1, 1.0, label_one_target),
            _instance(2, 1.0, label_two_target),
        ]],
    )

    assert metrics["val_mask_ap50"] == pytest.approx(0.25)
    assert metrics["val_mask_ap75"] == pytest.approx(0.25)
    assert metrics["val_mask_ap"] == pytest.approx(0.25)


def test_instance_ap_reports_ap50_ap75_and_coco_iou_average() -> None:
    target_mask = _rectangular_mask(0, 0, 2, 2)
    three_quarters_iou_mask = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    metrics = compute_instance_ap_metrics(
        [[_instance(7, 1.0, three_quarters_iou_mask)]],
        [[_instance(7, 1.0, target_mask)]],
    )

    assert metrics["val_mask_ap50"] == pytest.approx(1.0)
    assert metrics["val_mask_ap75"] == pytest.approx(1.0)
    assert metrics["val_mask_ap"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("predicted_instances", "target_instances"),
    [
        ([], []),
        ([[]], [[]]),
        ([[]], [[_instance(7, 1.0, _rectangular_mask(0, 0, 2, 2))]]),
        ([[_instance(7, 1.0, _rectangular_mask(0, 0, 2, 2))]], [[]]),
    ],
)
def test_instance_ap_empty_cases_are_finite(
    predicted_instances: list[list[dict[str, object]]],
    target_instances: list[list[dict[str, object]]],
) -> None:
    metrics = compute_instance_ap_metrics(predicted_instances, target_instances)

    assert set(metrics) == {"val_mask_ap", "val_mask_ap50", "val_mask_ap75"}
    assert all(math.isfinite(value) for value in metrics.values())
