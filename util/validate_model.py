from typing import Union

from fastapi import HTTPException
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from app.state import MODEL_REGISTRY


def validate_model(request: Union[InstanceSegmentationRequest, InstanceSegmentationTrainingRequest]):
    """Validate that the requested model exists and is usable for the request.

    Tags are read straight off the registered model (string values) rather than
    going through ``get_model_info`` / ``parse_tags_to_model_info``, which rebuilds
    a full ``ModelInfo`` and fails when the registered tags only carry the
    filterable subset (task/status/...).
    """
    try:
        registered_model = MODEL_REGISTRY.client.get_registered_model(request.model_registry_key)
    except Exception:
        raise HTTPException(status_code=404,
                            detail=f"Model '{request.model_registry_key}' is not registered.")
    tags = registered_model.tags or {}

    # Accept the legacy single ``task`` tag and the new per-task boolean tag
    # (``task.instance-segmentation`` == "true") so this keeps working across the
    # tag-encoding migration.
    is_instance_seg = (
        tags.get("task") == "instance-segmentation"
        or str(tags.get("task.instance-segmentation", "")).lower() == "true"
    )
    if not is_instance_seg:
        raise HTTPException(status_code=400,
                            detail=f"Model {request.model_registry_key} is not an instance segmentation model.")

    is_training = isinstance(request, InstanceSegmentationTrainingRequest)

    if is_training:
        # Training *adds* the classes in ``request.labels``; do not require the base
        # model to already predict them. Only enforce that the model is trainable.
        if str(tags.get("trainable", "")).lower() != "true":
            raise HTTPException(status_code=400,
                                detail=f"Model {request.model_registry_key} is not trainable!")
        return

    # Inference: if a label filter is given and the model declares its class set,
    # make sure the requested label is one the model can predict.
    if request.label is not None and tags.get("label_ids"):
        if str(request.label.id) not in tags.get("label_ids"):
            raise HTTPException(status_code=400,
                                detail=f"Model {request.model_registry_key} does not predict "
                                       f"label {request.label.name} (id {request.label.id}).")
