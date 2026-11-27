from types import SimpleNamespace

import pytest

from app.routes import models as routes
from models.cpsam_polyp import CellposePolyp
from models.yolo_polyp import YoloPolyp


@pytest.mark.parametrize("model", [CellposePolyp, YoloPolyp])
def test_catalog_uses_current_display_text_for_existing_registration(monkeypatch, model):
    current = model.model_info
    old = {"registry_key": current.registry_key, "name": "Old polyp (v3, seed 1)",
           "description": "Old configuration B seed 17 description"}
    monkeypatch.setattr(routes.mlflow.models, "get_model_info",
                        lambda _: SimpleNamespace(metadata=old))
    registry = SimpleNamespace(client=SimpleNamespace(get_registered_model=lambda _:
        SimpleNamespace(tags=current.tags, description=current.description)))
    result = routes._full_model_info(current.registry_key, registry)
    assert result["name"] == current.name
    assert result["description"] == current.description
    assert result["registry_key"] == current.registry_key
    assert "seed" not in result["name"].lower() + result["description"].lower()
    assert "v3" not in result["name"]
    assert old["name"] == "Old polyp (v3, seed 1)"
