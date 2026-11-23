"""M2C: hyperparameters, matcher, loss and the token-offset hook. No SAM 3 weights needed."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from models.base import INSTANCE_SEGMENTATION
from models.m2c import M2C, ConceptDataset, m2c_loss, match, validate_hyperparameters
from models.mask2former_dataset import LabelMapping


def _outputs(num_queries=6, size=16, *, logits=None, boxes=None, masks=None):
    return SimpleNamespace(
        pred_logits=torch.zeros(1, num_queries) if logits is None else logits,
        pred_boxes=torch.tensor([[0.1, 0.1, 0.5, 0.5]]).repeat(1, num_queries, 1) if boxes is None else boxes,
        pred_masks=torch.zeros(1, num_queries, size, size) if masks is None else masks,
        presence_logits=torch.zeros(1, 1),
        semantic_seg=torch.zeros(1, 1, size, size),
    )


def test_model_is_a_trainable_instance_segmenter():
    tags = M2C.model_info.tags
    assert M2C.supported_tasks() == [INSTANCE_SEGMENTATION]
    assert tags["trainable"] == "true" and tags[INSTANCE_SEGMENTATION.tag_key] == "true"


def test_hyperparameters_defaults_and_bounds():
    assert validate_hyperparameters({}) == (30, 5e-4, 4, "A {label} in an image")
    with pytest.raises(ValueError):
        validate_hyperparameters({"prompt_template": "no placeholder"})
    with pytest.raises(ValueError):
        validate_hyperparameters({"epochs": 0})


def test_match_pairs_each_box_with_the_query_on_it():
    gt = torch.tensor([[0.6, 0.6, 0.9, 0.9], [0.0, 0.0, 0.2, 0.2]])
    boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.4, 0.4, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
    rows, cols = match(torch.zeros(3), boxes, gt)
    assert dict(zip(cols.tolist(), rows.tolist())) == {0: 2, 1: 0}


def test_loss_without_instances_only_pushes_queries_and_presence_down():
    terms = m2c_loss(_outputs(), torch.zeros(0, 16, 16))
    assert set(terms) == {"ce", "presence", "semantic", "semantic_dice"}
    assert all(torch.isfinite(value) for value in terms.values())


def test_loss_drops_when_a_query_predicts_the_instance():
    gt = torch.zeros(1, 16, 16)
    gt[0, 2:8, 2:8] = 1
    wrong = _outputs()
    right_masks = torch.full((1, 6, 16, 16), -10.0)
    right_masks[0, 0, 2:8, 2:8] = 10.0
    right = _outputs(
        logits=torch.tensor([[5.0, -5, -5, -5, -5, -5]]),
        boxes=torch.tensor([[[2 / 16, 2 / 16, 8 / 16, 8 / 16]]]).repeat(1, 6, 1),
        masks=right_masks,
    )
    wrong_terms, right_terms = m2c_loss(wrong, gt), m2c_loss(right, gt)
    assert {"bbox", "giou", "mask", "dice"} <= set(wrong_terms)
    for key in ("ce", "mask", "dice", "giou"):
        assert right_terms[key] < wrong_terms[key], key


def test_dataset_keeps_a_parent_whole_under_its_child(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((40, 40, 3), np.uint8))
    square = lambda x0, x1: [[x0, x0, x1, x0, x1, x1, x0, x1]]
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 40, "height": 40}],
        "categories": [{"id": 7, "name": "fragment"}, {"id": 9, "name": "polyp"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "segmentation": square(0, 39)},
            {"id": 2, "image_id": 1, "category_id": 9, "segmentation": square(10, 20)},
        ],
    }
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    labels = [SimpleNamespace(id=7, name="fragment"), SimpleNamespace(id=9, name="polyp")]
    dataset = ConceptDataset(tmp_path / "coco.json", tmp_path, LabelMapping.from_selected_labels(labels),
                             mask_size=20)
    image, masks = dataset[0]
    assert image.shape == (40, 40, 3) and masks[7].shape == (1, 20, 20)
    assert masks[7][0, 5:10, 5:10].all(), "the polyp must not cut a hole into the fragment"
    assert masks[9][0].sum() < masks[7][0].sum()


def test_token_offset_hook_changes_embeddings_only_inside_the_block():
    embedding = torch.nn.Embedding(10, 4)
    model = M2C(device="cpu")
    model.model = SimpleNamespace(text_encoder=SimpleNamespace(
        text_model=SimpleNamespace(embeddings=SimpleNamespace(token_embedding=embedding))))
    ids = torch.tensor([[1, 2, 3]])
    phi = torch.ones(1, 3, 4)
    before = embedding(ids)
    with model._token_offset(phi):
        assert torch.allclose(embedding(ids), before + 1)
    assert torch.allclose(embedding(ids), before)
