"""The provisional 640-pixel Cellpose-SAM polyp inference provider.

The weights are intentionally loaded into Cellpose's low-level CPSAM network,
not through its stock-model constructor: that constructor can download and
silently substitute ``cpsam_v2`` when a path is missing.  The pinned Cellpose
``CellposeModel.eval`` implementation is reused after the complete fine-tuned
state has been loaded strictly.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.special import expit

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo, ModelClass
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import HyperParameter

from models.base import CapabilityModel, INSTANCE_SEGMENTATION, InstanceSegmentation
from models.registry import register_model

logger = logging.getLogger(__name__)

MODEL_KEY = "cpsam_polyp_v3_b_seed17"
MODEL_NAME = "Cellpose-SAM · Polyp"
CHECKPOINT_ENV = "IQUANA_CELLPOSE_POLYP_CHECKPOINT"
CHECKPOINT_SHA256_ENV = "IQUANA_CELLPOSE_POLYP_CHECKPOINT_SHA256"
EXPECTED_CHECKPOINT_SHA256 = "8d17682741428d0c71b93dc6112de9ff9d6595fc586fe797bf4566423b788eba"
DEVICE_ENV = "IQUANA_CELLPOSE_POLYP_DEVICE"
CELLPOSE_VERSION = "4.2.1.1"

TILE_SIZE = 640
TILE_OVERLAP = 128
MODEL_TILE_OVERLAP = 0.1
MODEL_BATCH_SIZE = 1
CANDIDATE_SCORE_FLOOR = 0.001
DEFAULT_APPLICATION_THRESHOLD = 0.05
PARENT_MARGIN = 0
MIN_SIZE = 15
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

# Cellpose's 640-pixel eval path mutates the backbone marker around a call.
# Serialize inference in each worker process so concurrent requests cannot race
# that temporary mutation or allocate multiple 640-pixel activations at once.
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.BoundedSemaphore(1)


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    core: tuple[int, int, int, int]


def axis_windows(length: int, tile: int = TILE_SIZE, overlap: int = TILE_OVERLAP) -> list[tuple[int, int, int]]:
    """Return exact reference ``(start, core_lo, core_hi)`` windows."""
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("Tile size must be positive and overlap must be in [0, tile size).")
    if length < tile:
        raise ValueError(f"image side {length} is smaller than the required {tile}-pixel tile")
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    cuts = [(left + tile + right) // 2 for left, right in zip(starts, starts[1:])]
    return list(zip(starts, [0] + cuts, cuts + [length]))


def tile_grid(height: int, width: int) -> list[Tile]:
    """The reference's row-major 640/128 tile grid and half-open ownership cores."""
    columns = axis_windows(width)
    return [
        Tile(x0, y0, (x_lo, y_lo, x_hi, y_hi))
        for y0, y_lo, y_hi in axis_windows(height)
        for x0, x_lo, x_hi in columns
    ]


def centre_owned(masks: np.ndarray, tile: Tile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ownership and native-pixel centroids using the reference rule."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or tuple(masks.shape[1:]) != (TILE_SIZE, TILE_SIZE):
        raise ValueError("Expected tile masks with shape (instances, 640, 640).")
    columns = masks.sum(axis=1)
    rows = masks.sum(axis=2)
    area = columns.sum(axis=1)
    axis = np.arange(TILE_SIZE, dtype=np.float64)
    x = (columns * axis).sum(axis=1) / np.maximum(area, 1) + tile.x0
    y = (rows * axis).sum(axis=1) / np.maximum(area, 1) + tile.y0
    x_lo, y_lo, x_hi, y_hi = tile.core
    owned = (area > 0) & (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
    return owned, x, y


def decode_parent_regions(
    parent_regions: Any, height: int, width: int
) -> np.ndarray | None:
    """Decode and union native-coordinate COCO RLEs; empty input is unrestricted."""
    if parent_regions is None or (isinstance(parent_regions, list) and not parent_regions):
        return None
    if not isinstance(parent_regions, list):
        raise ValueError("parent_regions must be a list of native-size COCO RLE objects.")

    union = np.zeros((height, width), dtype=bool)
    for index, rle in enumerate(parent_regions):
        if not isinstance(rle, Mapping) or rle.get("size") != [height, width]:
            raise ValueError(
                f"parent_regions[{index}] must have native size [{height}, {width}]."
            )
        counts = rle.get("counts")
        if isinstance(counts, str):
            encoded: dict[str, Any] = {"size": [height, width], "counts": counts.encode("ascii")}
        elif isinstance(counts, bytes):
            encoded = {"size": [height, width], "counts": counts}
        elif isinstance(counts, list):
            encoded = mask_utils.frPyObjects(
                {"size": [height, width], "counts": counts}, height, width
            )
        else:
            raise ValueError(f"parent_regions[{index}].counts must be compressed or uncompressed COCO RLE.")
        try:
            decoded = mask_utils.decode(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parent_regions[{index}] is not a valid COCO RLE.") from exc
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        if decoded.shape != (height, width):
            raise ValueError(f"parent_regions[{index}] decoded to a non-native shape.")
        union |= np.asarray(decoded, dtype=bool)
    return union


def select_tiles(tiles: Sequence[Tile], parent: np.ndarray | None) -> list[Tile]:
    """Keep all tiles without an ROI, otherwise only windows touching its union."""
    if parent is None:
        return list(tiles)
    return [tile for tile in tiles if parent[tile.y0:tile.y0 + TILE_SIZE, tile.x0:tile.x0 + TILE_SIZE].any()]


def parent_centroid_owned(parent: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Test rounded, clipped centroid pixels against the undilated ROI union."""
    height, width = parent.shape
    xs = np.clip(np.rint(x).astype(np.int64), 0, width - 1)
    ys = np.clip(np.rint(y).astype(np.int64), 0, height - 1)
    return parent[ys, xs]


def native_rle(mask: np.ndarray, x0: int, y0: int, height: int, width: int) -> dict[str, Any]:
    """Encode a tile mask as compressed native-size COCO RLE without a full canvas."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Only non-empty 2D masks can be encoded as native RLE.")
    tile_h, tile_w = mask.shape
    if x0 < 0 or y0 < 0 or x0 + tile_w > width or y0 + tile_h > height:
        raise ValueError("Tile mask extends beyond the native image bounds.")

    xs, ys = np.nonzero(mask.T)
    indices = (xs + x0).astype(np.int64) * height + (ys + y0)
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]])) + 1
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[0::2] = np.concatenate(
        ([starts[0]], starts[1:] - ends[:-1], [height * width - ends[-1]])
    )
    counts[1::2] = ends - starts
    if counts[-1] == 0:
        counts = counts[:-1]
    encoded = mask_utils.frPyObjects(
        {"size": [height, width], "counts": counts.tolist()}, height, width
    )
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def resize_cpsam_positional_embedding(net: Any, tile_size: int = TILE_SIZE) -> None:
    """Resize only the newly constructed architecture grid before checkpoint load."""
    position = getattr(getattr(net, "encoder", None), "pos_embed", None)
    patch_size = getattr(net, "ps", None)
    if not isinstance(position, torch.nn.Parameter):
        raise RuntimeError("Cellpose CPSAM encoder has no trainable positional embedding parameter.")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise RuntimeError("Cellpose CPSAM network has an invalid patch size.")
    if tile_size % patch_size:
        raise RuntimeError(f"Tile size {tile_size} is not divisible by CPSAM patch size {patch_size}.")
    if position.ndim != 4 or position.shape[0] != 1 or position.shape[1] != position.shape[2]:
        raise RuntimeError(f"Unexpected CPSAM positional embedding shape {tuple(position.shape)}.")
    target_grid = tile_size // patch_size
    if tuple(position.shape[1:3]) == (target_grid, target_grid):
        return
    resized = torch.nn.functional.interpolate(
        position.detach().permute(0, 3, 1, 2).float(),
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).to(device=position.device, dtype=position.dtype)
    net.encoder.pos_embed = torch.nn.Parameter(resized, requires_grad=position.requires_grad)


def strict_load_cpsam_checkpoint(net: Any, checkpoint_path: Path) -> None:
    """Load the complete CP4 state dict, rejecting incomplete or shape-mismatched weights."""
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:  # ``mmap`` was added after early supported PyTorch releases.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("The Cellpose-SAM polyp checkpoint is not a non-empty state dictionary.")
    if "W2" not in state and "module.W2" not in state:
        raise RuntimeError("The configured checkpoint is not a Cellpose 4 model (missing W2).")
    keys = list(state)
    if all(key.startswith("module.") for key in keys):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    elif any(key.startswith("module.") for key in keys):
        raise RuntimeError("The checkpoint mixes prefixed and unprefixed state-dictionary keys.")
    try:
        net.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "The Cellpose-SAM polyp checkpoint does not strictly match the pinned CPSAM architecture."
        ) from exc


def _prepare_cpsam_for_inference(net: Any, device: torch.device, dtype: torch.dtype) -> None:
    """Move the fully built and loaded CPSAM to one consistent inference dtype.

    Cellpose 4.2.1.1 constructs a few layers (including its replacement patch
    projection) with the default float32 dtype even when CPSAM was initialized
    with bfloat16. Casting the complete module after checkpoint load keeps those
    layers, buffers, and the net's input dtype aligned.
    """
    net.to(device=device, dtype=dtype)
    mixed_tensors = [
        f"parameter {name} ({value.dtype})"
        for name, value in net.named_parameters()
        if value.is_floating_point() and value.dtype != dtype
    ]
    mixed_tensors.extend(
        f"buffer {name} ({value.dtype})"
        for name, value in net.named_buffers()
        if value.is_floating_point() and value.dtype != dtype
    )
    if mixed_tensors:
        raise RuntimeError(
            f"CPSAM has floating tensors outside requested {dtype} inference precision: "
            + ", ".join(mixed_tensors[:5])
        )
    if getattr(net, "dtype", dtype) != dtype:
        raise RuntimeError(f"CPSAM reports dtype {net.dtype}, expected {dtype}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_checkpoint_sha256() -> str:
    """Return the configured exact artifact digest, defaulting to the published one."""
    value = os.environ.get(CHECKPOINT_SHA256_ENV, EXPECTED_CHECKPOINT_SHA256)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise RuntimeError(f"{CHECKPOINT_SHA256_ENV} must contain exactly 64 hexadecimal characters.")
    return value.lower()


def _device_name_from_environment() -> str:
    value = os.environ.get(DEVICE_ENV, "cuda:0").strip().lower()
    if value == "cpu":
        return value
    match = re.fullmatch(r"cuda(?::(\d+))?", value)
    if not match:
        raise RuntimeError(f"{DEVICE_ENV} must be 'cpu' or 'cuda:N' (got {value!r}).")
    return f"cuda:{int(match.group(1) or 0)}"


def configured_checkpoint(env: str) -> Path | None:
    """The checkpoint path set in ``env``, or None. There is no default location."""
    value = os.environ.get(env, "").strip()
    return Path(value).expanduser() if value else None


def runtime_status(env: str, package: str) -> str:
    """``ready`` when the optional extra is installed and the checkpoint is configured.

    Evaluated when the model class is defined, i.e. at service start, and synced to the
    registered model's ``status`` tag, so an unconfigured model is not offered at all
    instead of failing on its first request. The SHA-256 is still checked on load.
    """
    path = configured_checkpoint(env)
    if importlib.util.find_spec(package) is None or path is None or not path.is_file():
        return "not_ready"
    return "ready"


def _configured_model_cache_identity() -> tuple[str, str, str]:
    checkpoint = configured_checkpoint(CHECKPOINT_ENV)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Set {CHECKPOINT_ENV} to the verified cpsam_best artifact; there is no default path."
        )
    return (
        str(checkpoint.resolve()),
        configured_checkpoint_sha256(),
        _device_name_from_environment(),
    )


def _device_from_name(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    index = int(value.split(":", maxsplit=1)[1])
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{DEVICE_ENV} requests {value}, but CUDA is unavailable. Set it to 'cpu' only for an explicit CPU run."
        )
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{DEVICE_ENV} requests unavailable CUDA device {index}.")
    return torch.device("cuda", index)


def _active_polyp_label_id(request: InstanceSegmentationRequest) -> int:
    label = getattr(request, "label", None)
    raw = getattr(label, "id", None) if label is not None else None
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise ValueError(
            "Cellpose-SAM polyp inference requires the active IQuana label id on the request."
        )
    return int(raw)


def _cellpose_eval_model(checkpoint: Path, device: torch.device) -> tuple[Any, str]:
    """Build CPSAM without a stock checkpoint, then reuse the pinned eval methods."""
    try:
        import cellpose
        from cellpose.models import CellposeModel
        from cellpose.vit import CPSAM
    except ImportError as exc:
        raise RuntimeError(
            f"Cellpose-SAM polyp inference requires the pinned cellpose=={CELLPOSE_VERSION} extra."
        ) from exc
    try:
        version = importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(cellpose, "__version__", "unknown")
    if version != CELLPOSE_VERSION:
        raise RuntimeError(
            f"Cellpose-SAM polyp requires cellpose=={CELLPOSE_VERSION}; found {version}."
        )

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    net = CPSAM(dtype=dtype)
    resize_cpsam_positional_embedding(net)
    strict_load_cpsam_checkpoint(net, checkpoint)
    _prepare_cpsam_for_inference(net, device, dtype)
    net.eval()

    # The constructor rejects False and downloads cpsam_v2 for missing paths.
    # This pinned class uses only these attributes in eval/_run_net/_compute_masks.
    model = object.__new__(CellposeModel)
    model.device = device
    model.gpu = device.type == "cuda"
    model.pretrained_model = str(checkpoint)
    model.backbone = "sam_vitl"
    model.net = net
    return model, version


def eval_full_640(model: Any, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one native 640×640 RGB tile with Cellpose's guarded full-resolution path."""
    if model.backbone != "sam_vitl":
        raise RuntimeError(f"Expected a SAM-ViT-L CPSAM model, got backbone {model.backbone!r}.")
    previous_backbone = model.backbone
    model.backbone = "sam_vitl_v2_full640"
    try:
        masks, flows, _styles = model.eval(
            image_rgb,
            batch_size=MODEL_BATCH_SIZE,
            resample=True,
            channels=None,
            channel_axis=2,
            z_axis=None,
            normalize=True,
            rescale=None,
            diameter=None,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            do_3D=False,
            anisotropy=None,
            flow3D_smooth=0,
            stitch_threshold=0.0,
            min_size=MIN_SIZE,
            max_size_fraction=0.4,
            niter=None,
            augment=False,
            tile_overlap=MODEL_TILE_OVERLAP,
            bsize=TILE_SIZE,
            compute_masks=True,
            progress=None,
        )
    finally:
        model.backbone = previous_backbone
    labels = np.asarray(masks)
    logits = np.asarray(flows[2])
    if labels.shape != (TILE_SIZE, TILE_SIZE) or logits.shape != (TILE_SIZE, TILE_SIZE):
        raise RuntimeError(
            f"Cellpose returned non-native tile outputs: masks={labels.shape}, logits={logits.shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"Cellpose returned non-integer instance labels ({labels.dtype}).")
    return labels, logits


def _contour_dicts(
    mask: np.ndarray,
    *,
    score: float,
    label_id: int,
    model_key: str,
    source_instance_id: str,
    tile: Tile,
    image_height: int,
    image_width: int,
) -> list[dict[str, Any]]:
    """Serialize float-normalized polygons in the native image coordinate frame.

    ``Contour.from_binary_mask`` in the pinned toolbox normalizes OpenCV's int32
    contours in place, truncating every point to zero. Keep its contour schema
    serializer while normalizing the extracted float points here.
    """
    pixel_contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = []
    for pixel_contour in pixel_contours:
        normalized = pixel_contour.astype(np.float64)
        normalized[..., 0] = (tile.x0 + normalized[..., 0]) / image_width
        normalized[..., 1] = (tile.y0 + normalized[..., 1]) / image_height
        contour = Contour.from_normalized_cv_contour(
            normalized_cv_contour=normalized,
            confidence=score,
            label_id=label_id,
            added_by=model_key,
        )
        contour.compute_path(image_width=image_width, image_height=image_height)
        if hasattr(contour, "model_dump"):
            payload = contour.model_dump(mode="json")
        else:
            payload = contour.dict()
        payload["source_instance_id"] = source_instance_id
        result.append(payload)
    return result


@register_model
class CellposePolyp(InstanceSegmentation, CapabilityModel):
    """Fine-tuned Cellpose-SAM v3/B/seed17 full-resolution polyp detector."""

    model_info = InstanceSegmentationModelInfo(
        registry_key=MODEL_KEY,
        name=MODEL_NAME,
        description=(
            "Provisional experimental Cellpose-SAM polyp model, trained as run B with seed 17. "
            "Predictions are editable suggestions and are returned in native image coordinates."
        ),
        usage_tip=(
            "Runs full-resolution 640-pixel tiles. Saved ROI outlines restrict tile selection and "
            "candidate-centroid ownership; no ROI means whole-image inference."
        ),
        classes=[ModelClass(key="polyp", name="Polyp")],
        tags={
            # Discovery filters on the registered-model `status` tag (ModelInfo.status is
            # not copied into it), so both carry the runtime check.
            "status": runtime_status(CHECKPOINT_ENV, "cellpose"),
            # Accepts saved outlines as `parent_regions` (native COCO RLEs) to bound
            # inference; the backend and frontend read this instead of model keys.
            "roi_input": "parent_regions",
            "maturity": "provisional",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "colonoscopy",
            "architecture": "Cellpose-SAM",
            "model_version": "v3_B_seed17",
            "default_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        status=runtime_status(CHECKPOINT_ENV, "cellpose"),
        trainable=False,
        input_contracts=[
            InputContract(
                task=INSTANCE_SEGMENTATION.name,
                conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                parameters=[
                    HyperParameter(
                        key="threshold",
                        label="Application confidence threshold",
                        type="float",
                        default_value=DEFAULT_APPLICATION_THRESHOLD,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        description=(
                            "Suggestions below this mean-sigmoid cell-probability score are hidden. "
                            "The Cellpose candidate floor remains 0.001."
                        ),
                    )
                ],
            )
        ],
    )

    _unpicklable_attrs = ("_cellpose_model",)
    _passthrough_input_params = ("parent_regions",)

    def __init__(self) -> None:
        super().__init__()
        self._cellpose_model: Any = None
        self._checkpoint_sha256: str | None = None
        self._cellpose_version: str | None = None
        self._model_cache_identity: tuple[str, str, str] | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._cellpose_model = None
        self._checkpoint_sha256 = None
        self._cellpose_version = None
        self._model_cache_identity = None

    def _load_weights(self) -> None:
        with _MODEL_LOAD_LOCK:
            cache_identity = _configured_model_cache_identity()
            if self._cellpose_model is not None:
                if self._model_cache_identity != cache_identity:
                    raise RuntimeError(
                        "Cellpose-SAM polyp checkpoint or device configuration changed after model load; "
                        "restart the worker to apply it."
                    )
                return
            checkpoint = Path(cache_identity[0])
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cellpose-SAM polyp checkpoint is missing at {checkpoint}; set {CHECKPOINT_ENV} "
                    "to the verified cpsam_best artifact. Stock Cellpose weights are never substituted."
                )
            checkpoint_sha256 = _sha256(checkpoint)
            expected_sha256 = cache_identity[1]
            if checkpoint_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Cellpose-SAM polyp checkpoint SHA-256 mismatch at {checkpoint}: "
                    f"expected {expected_sha256}, found {checkpoint_sha256}."
                )
            model, version = _cellpose_eval_model(
                checkpoint, _device_from_name(cache_identity[2])
            )
            self._cellpose_model = model
            self._checkpoint_sha256 = checkpoint_sha256
            self._cellpose_version = version
            self._model_cache_identity = cache_identity
            logger.info(
                "Loaded %s on %s with strict checkpoint %s",
                MODEL_KEY,
                model.device,
                checkpoint_sha256,
            )

    def segment_instances(
        self, request: InstanceSegmentationRequest, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return serialized suggestions and raw per-instance provenance metadata."""
        params = params or {}
        try:
            threshold = float(params.get("threshold", DEFAULT_APPLICATION_THRESHOLD))
        except (TypeError, ValueError) as exc:
            raise ValueError("Application confidence threshold must be numeric.") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Application confidence threshold must be finite and in [0, 1].")

        image_bgr = np.asarray(request.image)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Cellpose-SAM polyp inference expects a three-channel BGR image.")
        height, width = image_bgr.shape[:2]
        tiles_all = tile_grid(height, width)
        parent = decode_parent_regions(params.get("parent_regions"), height, width)
        tiles = select_tiles(tiles_all, parent)
        label_id = _active_polyp_label_id(request)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self._ensure_weights()
        if self._cellpose_model is None or self._checkpoint_sha256 is None:
            raise RuntimeError("Cellpose-SAM polyp model did not initialize.")
        if (
            self._model_cache_identity is not None
            and self._model_cache_identity != _configured_model_cache_identity()
        ):
            raise RuntimeError(
                "Cellpose-SAM polyp checkpoint or device configuration changed after model load; "
                "restart the worker to apply it."
            )

        contour_items: list[dict[str, Any]] = []
        instance_metadata: list[dict[str, Any]] = []
        try:
            with _INFERENCE_LOCK, torch.inference_mode():
                for tile_index, tile in enumerate(tiles):
                    crop = image_rgb[tile.y0:tile.y0 + TILE_SIZE, tile.x0:tile.x0 + TILE_SIZE]
                    labels, cellprob_logits = eval_full_640(self._cellpose_model, crop)
                    instance_labels = [int(value) for value in np.unique(labels) if value > 0]
                    if not instance_labels:
                        continue
                    masks = np.stack([labels == value for value in instance_labels], axis=0)
                    owned, centroid_x, centroid_y = centre_owned(masks, tile)
                    if parent is not None:
                        owned &= parent_centroid_owned(parent, centroid_x, centroid_y)

                    for local_index, instance_label in enumerate(instance_labels):
                        if not owned[local_index]:
                            continue
                        mask = masks[local_index]
                        score = float(expit(cellprob_logits[mask]).mean(dtype=np.float64))
                        if not math.isfinite(score) or score < CANDIDATE_SCORE_FLOOR:
                            continue
                        if score < threshold:
                            continue

                        rle = native_rle(mask, tile.x0, tile.y0, height, width)
                        rle_digest = hashlib.sha256(
                            f"{rle['size'][0]}x{rle['size'][1]}:{rle['counts']}".encode("ascii")
                        ).hexdigest()[:16]
                        source_instance_id = f"{MODEL_KEY}:t{tile_index:04d}:{rle_digest}"
                        contour_items.extend(
                            _contour_dicts(
                                mask,
                                score=score,
                                label_id=label_id,
                                model_key=request.model_registry_key,
                                source_instance_id=source_instance_id,
                                tile=tile,
                                image_height=height,
                                image_width=width,
                            )
                        )
                        instance_metadata.append(
                            {
                                "source_instance_id": source_instance_id,
                                "rle": rle,
                                "score": score,
                                "cellpose_label": instance_label,
                            }
                        )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "Cellpose-SAM polyp inference ran out of device memory. Reduce worker concurrency "
                    "or use a GPU with more memory; this provider serializes inference within each worker."
                ) from exc
            raise

        return {
            "contours": contour_items,
            "prediction_metadata": {
                "model_key": MODEL_KEY,
                "model_name": MODEL_NAME,
                "checkpoint": getattr(configured_checkpoint(CHECKPOINT_ENV), "name", None),
                "checkpoint_sha256": self._checkpoint_sha256,
                "device": str(self._cellpose_model.device),
                "precision": str(self._cellpose_model.net.dtype),
                "cellpose_version": self._cellpose_version,
                "class_key": "polyp",
                "label_id": label_id,
                "coordinate_space": "native_image_rle; Contour x/y follow the shared normalized-coordinate schema",
                "roi_mode": "parent_regions" if parent is not None else "whole_image",
                "roi_margin": PARENT_MARGIN,
                "tiles_total": len(tiles_all),
                "tiles_run": len(tiles),
                "candidate_score_floor": CANDIDATE_SCORE_FLOOR,
                "application_threshold": threshold,
                "inference_config": {
                    "tile_size": TILE_SIZE,
                    "tile_overlap": TILE_OVERLAP,
                    "model_batch_size": MODEL_BATCH_SIZE,
                    "cellpose_tile_overlap": MODEL_TILE_OVERLAP,
                    "channel_axis": 2,
                    "normalize": True,
                    "diameter": None,
                    "flow_threshold": FLOW_THRESHOLD,
                    "cellprob_threshold": CELLPROB_THRESHOLD,
                    "min_size": MIN_SIZE,
                    "max_size_fraction": 0.4,
                    "augment": False,
                    "resample": True,
                    "bsize": TILE_SIZE,
                    "do_3D": False,
                },
                "instances": instance_metadata,
            },
        }
