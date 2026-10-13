"""Serve any semantic segmentation model as full instance segmentation.

A semantic model scores every pixel for a fixed set of classes; the instance
segmentation surface wants objects. The step between them is the same for every such
model, so it lives here once:

* each class's score map is thresholded and split into connected components, one
  object per component, small specks dropped;
* a class that occurs once per image (``single_instance``) keeps only its strongest
  component;
* a class that is ``part_of`` another is nested: each of its objects becomes a child
  of the parent-class object holding most of its pixels;
* every object names its class in ``Contour.model_class``, so the gateway can map it
  onto a dataset's labels.

Two ways to use it, pick the lighter one:

**Wrap a model** -- mix in :class:`SemanticInstanceSegmentation`, declare the classes and
implement one method returning per-class scores::

    @register_model
    class MySegmenter(SemanticInstanceSegmentation, CapabilityModel):
        semantic_classes = (
            SemanticClass("cell", "Cell"),
            SemanticClass("nucleus", "Nucleus", part_of="cell"),
        )
        model_info = InstanceSegmentationModelInfo(registry_key="my-segmenter", ...)

        def class_scores(self, image):            # BGR (H, W, 3) uint8
            return my_network(image)              # (2, H, W) floats in [0, 1]

``model_info.classes`` and the instance-segmentation input contract (one threshold per
class plus a minimum object size) are filled in from ``semantic_classes`` unless the
model declares its own.

**Post-process only** -- call :func:`semantic_to_instances` on score maps you already
have. A model that outputs a label map (argmax) rather than scores goes through
:func:`one_hot` first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Sequence

import cv2
import numpy as np

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelClass
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import HyperParameter

from models.base import INSTANCE_SEGMENTATION, InstanceSegmentation

MIN_AREA_PARAM = "min_area"
DEFAULT_MIN_AREA = 20


@dataclass(frozen=True)
class SemanticClass:
    """One output class of a semantic model, and how to turn it into objects.

    Attributes:
        key: Stable identifier; becomes ``ModelClass.key`` and ``Contour.model_class``.
        name: Name shown to people, and the default name of a dataset label made for it.
        threshold: Scores at or above this belong to the class.
        single_instance: The class occurs once per image; keep only its strongest component.
        part_of: Key of the class this one is part of; its objects are nested under that
            class's objects.
        description: What the class covers.
    """

    key: str
    name: str
    threshold: float = 0.5
    single_instance: bool = False
    part_of: Optional[str] = None
    description: Optional[str] = None

    @property
    def threshold_param(self) -> str:
        return f"{self.key}_threshold"

    def model_class(self) -> ModelClass:
        return ModelClass(
            key=self.key, name=self.name, description=self.description, part_of=self.part_of
        )


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def one_hot(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    """Turn an ``(H, W)`` map of class indices into ``(num_classes, H, W)`` 0/1 scores."""
    return (label_map[None] == np.arange(num_classes)[:, None, None]).astype(np.float32)


def component_contours(
    scores: np.ndarray,
    threshold: float,
    *,
    min_area: int,
    single_instance: bool = False,
    added_by: str = "",
    model_class: Optional[str] = None,
) -> tuple[np.ndarray, dict[int, Contour]]:
    """Split one class's score map into one contour per connected component.

    Args:
        scores: ``(H, W)`` scores of one class.
        threshold: Pixels at or above this belong to the class.
        min_area: Components smaller than this many pixels are dropped as speckle.
        single_instance: The map locates something that occurs once (a landmark, say):
            keep only the component with the highest peak, and report that peak as its
            confidence. Such maps are often soft heatmaps whose mean is low by construction.
        added_by: Written to every contour's ``added_by``.
        model_class: Written to every contour's ``model_class``.

    Returns:
        The component label map, and the kept contours keyed by their component's index
        in it. Contours are in normalised coordinates; ``confidence`` is the mean score
        inside the component, or its peak with ``single_instance``.
    """
    height, width = scores.shape
    mask = (scores >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    found: dict[int, tuple[float, Contour]] = {}
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < min_area:
            continue
        component = labels[y:y + h, x:x + w] == index
        outlines, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y)
        )
        if not outlines:
            continue
        outline = max(outlines, key=cv2.contourArea).astype(np.float64)
        outline[..., 0] /= width
        outline[..., 1] /= height
        inside = scores[y:y + h, x:x + w][component]
        peak = float(inside.max())
        confidence = peak if single_instance else float(inside.mean())
        found[index] = (peak, Contour.from_normalized_cv_contour(
            outline, confidence=confidence, added_by=added_by, model_class=model_class,
        ))

    if single_instance and found:
        strongest = max(found, key=lambda index: found[index][0])
        found = {strongest: found[strongest]}
    return labels, {index: contour for index, (_, contour) in found.items()}


def nest_in_parents(
    parent_labels: np.ndarray,
    parents: Mapping[int, Contour],
    child_labels: np.ndarray,
    children: Mapping[int, Contour],
) -> list[Contour]:
    """Attach each child component to the parent component holding most of its pixels.

    A child goes under a parent only when more than half of it lies inside that parent;
    the others are returned, to stay where they are.
    """
    loose: list[Contour] = []
    for index, child in children.items():
        under = parent_labels[child_labels == index]
        votes = np.bincount(under, minlength=int(parent_labels.max()) + 1)
        votes[0] = 0  # background
        for dropped in set(np.flatnonzero(votes).tolist()) - set(parents):
            votes[dropped] = 0  # a speck below min_area is not an object to nest under
        best = int(votes.argmax())
        if votes[best] * 2 > under.size:
            parents[best].add_child(child)
        else:
            loose.append(child)
    return loose


def semantic_to_instances(
    scores: np.ndarray,
    classes: Sequence[SemanticClass],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    min_area: int = DEFAULT_MIN_AREA,
    added_by: str = "",
) -> list[Contour]:
    """Turn ``(C, H, W)`` per-class scores into instance contours.

    Args:
        scores: One score map per entry of ``classes``, in the same order.
        classes: The model's classes.
        thresholds: Per-class threshold overrides by class key; the class default otherwise.
        min_area: Components smaller than this many pixels are dropped.
        added_by: Written to every contour's ``added_by``.

    Returns:
        The top-level objects, in class order. Objects of a class that is ``part_of``
        another are children of the parent object containing them, unless they lie
        mostly outside every parent object, in which case they stay top-level.
    """
    if scores.ndim != 3 or scores.shape[0] != len(classes):
        raise ValueError(
            f"Expected scores of shape ({len(classes)}, H, W) for {len(classes)} classes, "
            f"got {scores.shape}."
        )
    thresholds = thresholds or {}
    found = {
        cls.key: component_contours(
            channel,
            float(thresholds.get(cls.key, cls.threshold)),
            min_area=min_area,
            single_instance=cls.single_instance,
            added_by=added_by,
            model_class=cls.key,
        )
        for cls, channel in zip(classes, scores)
    }
    # add_child links the objects themselves, so a chain (cell > nucleus > nucleolus)
    # nests correctly whatever order the classes come in. What changes per class is only
    # which of its objects are still top-level.
    top_level = {key: dict(contours) for key, (_, contours) in found.items()}
    for cls in classes:
        if cls.part_of is None:
            continue
        parent_labels, parents = found[cls.part_of]
        child_labels, children = found[cls.key]
        loose = nest_in_parents(parent_labels, parents, child_labels, children)
        top_level[cls.key] = {id(contour): contour for contour in loose}
    return [contour for cls in classes for contour in top_level[cls.key].values()]


def semantic_contract(
    classes: Sequence[SemanticClass], *, min_area: int = DEFAULT_MIN_AREA
) -> InputContract:
    """The instance-segmentation contract of a semantic model: a threshold per class and
    a minimum object size."""
    return InputContract(
        task=INSTANCE_SEGMENTATION.name,
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            *(
                HyperParameter(
                    key=cls.threshold_param,
                    label=f"{cls.name} threshold",
                    type="float",
                    default_value=cls.threshold,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    description=f"Pixels scoring at or above this count as {cls.name.lower()}.",
                )
                for cls in classes
            ),
            HyperParameter(
                key=MIN_AREA_PARAM,
                label="Minimum object size (px)",
                type="int",
                default_value=min_area,
                min_value=0,
                max_value=100_000,
                step=10,
                description="Connected areas smaller than this are dropped as noise.",
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Model mixin
# --------------------------------------------------------------------------- #
class SemanticInstanceSegmentation(InstanceSegmentation):
    """Instance segmentation for a model that scores pixels per class.

    Declare ``semantic_classes`` and implement :meth:`class_scores`; the instance
    handler, ``model_info.classes`` and the input contract come from here.
    """

    semantic_classes: ClassVar[tuple[SemanticClass, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        # Before CapabilityModel's own hook (further up the MRO), which validates the
        # contracts and stamps task tags from them.
        info = cls.__dict__.get("model_info")
        if info is not None and cls.semantic_classes:
            if not info.classes:
                info.classes = [c.model_class() for c in cls.semantic_classes]
            if not any(c.task == INSTANCE_SEGMENTATION.name for c in info.input_contracts):
                info.input_contracts.append(semantic_contract(cls.semantic_classes))
        super().__init_subclass__(**kwargs)

    def class_scores(self, image: np.ndarray) -> np.ndarray:
        """Score every pixel of a BGR ``(H, W, 3)`` image: ``(C, H, W)`` in ``[0, 1]``,
        one map per entry of ``semantic_classes``, in that order."""
        raise NotImplementedError

    def segment_instances(
        self, request: InstanceSegmentationRequest, params: dict[str, Any] | None = None
    ) -> list[Contour]:
        params = params or {}
        return semantic_to_instances(
            self.class_scores(request.image),
            self.semantic_classes,
            thresholds={
                cls.key: params[cls.threshold_param]
                for cls in self.semantic_classes
                if params.get(cls.threshold_param) is not None
            },
            min_area=int(params.get(MIN_AREA_PARAM, DEFAULT_MIN_AREA)),
            added_by=request.model_registry_key,
        )
