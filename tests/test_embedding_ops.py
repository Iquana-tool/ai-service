"""Unit tests for the pure embedding ops (models.embedding_ops).

These need only NumPy + OpenCV -- no torch, no DINOv3 weights, no HuggingFace login -- so the
geometry/pooling math behind region embeddings is verified independently of the heavy backbone.
"""
import numpy as np
import pytest

from models import embedding_ops as ops


def test_l2_normalize_unit_norm():
    v = ops.l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v.tolist() == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_unchanged():
    z = np.zeros(5)
    assert np.array_equal(ops.l2_normalize(z), z)


def test_mask_bbox_tight_and_padded():
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 3:7] = True  # ys 4..5, xs 3..6

    assert ops.mask_bbox(mask) == (4, 6, 3, 7)

    # 50% padding: height 2 -> pad 1 each side; width 4 -> pad 2 each side.
    assert ops.mask_bbox(mask, pad_frac=0.5) == (3, 7, 1, 9)


def test_mask_bbox_clamps_to_frame():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 6:8] = True
    # Padding would push past the edges; result is clamped to [0, 8].
    assert ops.mask_bbox(mask, pad_frac=1.0) == (0, 4, 4, 8)


def test_mask_bbox_empty_returns_full_frame():
    assert ops.mask_bbox(np.zeros((6, 9), dtype=bool)) == (0, 6, 0, 9)


def test_crop_to_mask_returns_aligned_crops():
    image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:8] = True

    crop, crop_mask = ops.crop_to_mask(image, mask)
    assert crop.shape == (3, 5, 3)
    assert crop_mask.shape == (3, 5)
    assert crop_mask.all()  # the tight crop is entirely foreground


def test_resize_mask_to_grid_is_binary_and_shaped():
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16, :] = True  # top half foreground

    grid = ops.resize_mask_to_grid(mask, (4, 4))
    assert grid.shape == (4, 4)
    assert grid.dtype == bool
    # Top two grid rows map to the foreground half, bottom two to background.
    assert grid[:2, :].all()
    assert not grid[2:, :].any()


def test_masked_mean_selects_foreground_patches():
    # Two channels; craft the grid so foreground and background have distinct values.
    # Foreground patch feature = [2, 0]; background = [0, 2].
    grid = np.zeros((2, 2, 2), dtype=np.float64)  # (C=2, Hp=2, Wp=2)
    grid[0, 0, :] = 2.0  # top row -> channel-0 energy (foreground)
    grid[1, 1, :] = 2.0  # bottom row -> channel-1 energy (background)

    mask_grid = np.zeros((2, 2), dtype=bool)
    mask_grid[0, :] = True  # select the top row only

    out = ops.masked_mean(grid, mask_grid)
    # Mean of foreground patches is [2, 0] -> normalized [1, 0].
    assert out.tolist() == pytest.approx([1.0, 0.0])
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_masked_mean_empty_mask_falls_back_to_whole_grid():
    grid = np.ones((3, 2, 2), dtype=np.float64)
    empty = np.zeros((2, 2), dtype=bool)
    out = ops.masked_mean(grid, empty)
    # All patches equal -> normalized mean is the unit vector along the equal components.
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out.tolist() == pytest.approx([1 / np.sqrt(3)] * 3)
