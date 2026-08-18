"""Frozen DINOv3 feature extractor.

DINOv3 (Meta, 2025) is a self-supervised ViT whose *dense* patch features are
strong enough to drive detection/segmentation with the backbone **frozen** — only
a lightweight task head is trained. That frozen-backbone setup is the whole point:
a tiny number of trainable parameters means fast convergence on little data,
versus fine-tuning a full backbone end-to-end.

This wrapper loads a DINOv3 checkpoint via HuggingFace ``transformers`` (the
``facebook/dinov3-*`` models, available since transformers 4.56), freezes it, and
exposes the per-patch features as a spatial grid ``(B, C, Hp, Wp)`` ready to feed a
segmentation/detection head.

Notes
-----
* DINOv3 weights are **gated** on the HuggingFace Hub. The hosting process must be
  logged in (``huggingface-cli login`` / ``HF_TOKEN``) and have accepted the model
  terms. The unified service handles this through its application lifespan.
* DINOv3 ships under Meta's own *DINOv3 License* (not Apache-2.0), which carries
  commercial-use restrictions. Fine for research; check the terms before commercial use.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

# ViT-B/16 is the accuracy/speed sweet spot. Swap for ``dinov3-vits16-…`` (smaller,
# faster, hidden=384) or ``dinov3-vitl16-…`` (larger, more accurate) as needed.
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DINOv3Backbone(nn.Module):
    """A frozen DINOv3 backbone that returns dense patch features as a 2D grid.

    The backbone is always in eval mode with ``requires_grad=False``; its forward
    pass runs under ``torch.no_grad()`` so no activations are retained for it. Heads
    built on top start their gradient graph at the returned feature tensor.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_MODEL,
        image_size: int = 768,
        token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token = token

        self.processor = AutoImageProcessor.from_pretrained(model_id, token=self.token)
        self.model = AutoModel.from_pretrained(model_id, token=self.token).to(self.device)
        # Freeze: no parameter of the backbone ever receives gradients.
        self.model.requires_grad_(False)
        self.model.eval()

        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)
        # ViT variants prepend a CLS token and N register tokens before the patch
        # tokens; ConvNeXt variants emit a spatial map directly (handled in forward).
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))

        self.set_image_size(image_size)
        logger.info(
            "Loaded frozen DINOv3 backbone '%s' (patch=%d, hidden=%d) on %s",
            model_id, self.patch_size, self.hidden_size, self.device,
        )

    # -- configuration ------------------------------------------------------

    def set_image_size(self, image_size: int) -> None:
        """Force the processor to a fixed square size (must be a multiple of patch_size).

        A fixed square input gives a fixed ``(Hp, Wp)`` patch grid, which keeps the
        head's mask resolution predictable and lets training resize GT masks to match.
        """
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be a multiple of the patch size "
                f"({self.patch_size}) so the patch grid is exact."
            )
        self.image_size = image_size
        self.processor.do_resize = True
        self.processor.size = {"height": image_size, "width": image_size}
        # Disable any center-crop so the full (resized) image maps onto the grid.
        if hasattr(self.processor, "do_center_crop"):
            self.processor.do_center_crop = False
        if hasattr(self.processor, "crop_size"):
            self.processor.crop_size = {"height": image_size, "width": image_size}

    @property
    def grid_size(self) -> int:
        """Side length of the square patch grid for the current ``image_size``."""
        return self.image_size // self.patch_size

    # -- features -----------------------------------------------------------

    def preprocess(
        self,
        images: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> torch.Tensor:
        """Resize+normalize one image or a list of ``(H, W, 3)`` uint8 RGB arrays.

        Returns ``pixel_values`` of shape ``(B, 3, image_size, image_size)`` on the
        backbone's device.
        """
        proc = self.processor(images=images, return_tensors="pt")
        return proc["pixel_values"].to(self.device)

    @torch.no_grad()
    def forward(
        self, pixel_values: torch.Tensor, return_cls: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Extract dense patch features as ``(B, hidden_size, Hp, Wp)``.

        With ``return_cls=True`` also return the global ``CLS`` token ``(B, hidden_size)`` --
        a holistic descriptor of the whole image, used e.g. to re-identify a crop against a
        reference. For ConvNeXt-style backbones (no CLS token) the CLS is approximated by a
        global average pool of the spatial map.
        """
        out = self.model(pixel_values=pixel_values)
        last_hidden = out.last_hidden_state

        # ConvNeXt-style backbones already return a spatial map (B, C, H, W).
        if last_hidden.dim() == 4:
            if return_cls:
                return last_hidden, last_hidden.mean(dim=(2, 3))
            return last_hidden

        batch = pixel_values.shape[0]
        hp = pixel_values.shape[-2] // self.patch_size
        wp = pixel_values.shape[-1] // self.patch_size
        # Drop the CLS + register prefix tokens (computed by subtraction so it is
        # robust regardless of how many special tokens a given variant carries).
        num_prefix = last_hidden.shape[1] - hp * wp
        patches = last_hidden[:, num_prefix:, :]  # (B, Hp*Wp, C)
        grid = patches.transpose(1, 2).reshape(batch, self.hidden_size, hp, wp)
        if return_cls:
            cls = last_hidden[:, 0, :]            # the CLS token is the first prefix token
            return grid, cls
        return grid
