"""Embedders loaded by Hugging Face link: preset parsing and the embed handler's shape."""
import numpy as np

from models import pretrained_embedders as pe


class _Request:
    def __init__(self, kinds):
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_kinds = kinds


def test_extra_embedders_need_only_the_hf_id():
    first, second = pe.extra_presets("facebook/dinov2-base, imageomics/bioclip@open_clip")
    assert (first.key, first.hf_id, first.loader) == ("hf-facebook-dinov2-base", "facebook/dinov2-base", "transformers")
    assert (second.key, second.loader) == ("hf-imageomics-bioclip", "open_clip")
    assert pe.extra_presets("") == []


def test_every_preset_is_registered_with_a_known_loader():
    keys = {preset.key for preset in pe.PRESETS}
    assert {"ijepa", "bioclip-2", "biomedclip", "curia", "retfound-oct"} <= keys
    assert all(preset.loader in pe.LOADERS for preset in pe.PRESETS)
    assert all(hasattr(pe, "Embedder_" + p.key.replace("-", "_")) for p in pe.PRESETS)


def test_embed_returns_one_normalized_whole_image_vector():
    model = pe.Embedder_biomedclip()
    model.encoder = lambda image: np.array([3.0, 4.0])  # stands in for the loaded model
    [vector] = model.embed(_Request(["image_cls"]))
    assert (vector.kind, vector.dim, vector.model_id) == ("image_cls", 2, pe.PRESETS[2].hf_id)
    assert np.allclose(vector.vector, [0.6, 0.8])
    assert model.embed(_Request(["region_mean"])) == []
