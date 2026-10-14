"""Choroidalyzer: choroid region, choroidal vessels and fovea on OCT B-scans.

Choroidalyzer (Engelmann et al., IOVS 2024, https://github.com/justinengelmann/Choroidalyzer)
is a small U-Net that reads one grayscale B-scan and writes three sigmoid maps. It
is a *semantic* model with a fixed label space, taken from the upstream files:
``Choroidalyzer.outputs = ['region', 'vessel', 'fovea']`` names the three output
channels of ``UNet(out_channels=3)``, and ``DEFAULT_THRESHOLDS = (0.5, 0.5, 0.1)``
binarises them in that order. See :data:`CLASSES`.

Everything after the network -- thresholds, one object per connected area, vessels
nested in the choroid, the fovea kept once -- is the generic semantic adapter in
:mod:`models.semantic`; this module only supplies the classes and the scores. Every
object names its class (``Contour.model_class``); the backend maps classes onto a
dataset's labels.

The upstream repository has no packaging (and no licence file), so its code is not
a dependency. :class:`_ChoroidUNet` is a re-implementation of the same network whose
module names match the released state dict, which is loaded strictly; the weights
are fetched from the upstream release on first use and checked against a pinned
SHA-256.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch import nn

from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo, ModelPerformance

from models.base import CapabilityModel
from models.registry import register_model
from models.semantic import SemanticClass, SemanticInstanceSegmentation

logger = logging.getLogger(__name__)

WEIGHTS_URL = (
    "https://github.com/justinengelmann/Choroidalyzer/releases/download/v1.0/"
    "choroidalyzer_model_weights.pth"
)
WEIGHTS_SHA256 = "4b9b0c0548e0524d0ad57700c7372cdf3853452c905d1ebf85bfc48748623697"

#: The fixed label space, in network channel order, with upstream's thresholds.
CLASSES: tuple[SemanticClass, ...] = (
    SemanticClass("region", "Choroid region", 0.5,
                  description="The vascular layer between the retina and the sclera."),
    SemanticClass("vessel", "Choroidal vessels", 0.5, part_of="region",
                  description="Vessel lumina inside the choroid."),
    # A soft location heatmap, not a mask: one object, confidence = its peak.
    SemanticClass("fovea", "Fovea", 0.1, single_instance=True,
                  description="The foveal pit at the centre of the macula."),
)

#: Depth of the U-Net: inputs must be a multiple of ``2 ** _DEPTH`` on both sides.
_DEPTH = 7
_CHANNELS = (8, 16, 32, 64, 64, 64, 64, 64)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        norm=nn.BatchNorm2d(out_channels),
        act=nn.ReLU(),
    ))


class _Up(nn.Module):
    """Halve the channels, upsample x2, concatenate the skip and convolve twice."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.conv1 = _conv_bn_relu(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = _conv_bn_relu(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.upsample(x), skip], dim=1)
        return self.conv2(self.conv1(x))


class _ChoroidUNet(nn.Module):
    """The Choroidalyzer network: 1 grayscale channel in, 3 logit maps out.

    Upstream: ``UNet(in_channels=1, out_channels=3, depth=7, channels='8_doublemax-64',
    up_type='conv_then_interpolate', extra_out_conv=True)``.
    """

    def __init__(self) -> None:
        super().__init__()
        c = _CHANNELS
        self.in_conv = _conv_bn_relu(1, c[0])
        self.down_blocks = nn.ModuleList(
            nn.Sequential(OrderedDict(
                pool=nn.MaxPool2d(2),
                conv1=_conv_bn_relu(c[d], c[d + 1]),
                conv2=_conv_bn_relu(c[d + 1], c[d + 1]),
            ))
            for d in range(_DEPTH)
        )
        self.up_blocks = nn.ModuleList(
            _Up(c[d + 1], c[d], c[d]) for d in reversed(range(_DEPTH))
        )
        self.out_conv = nn.Sequential(_conv_bn_relu(c[0], c[0]), nn.Conv2d(c[0], len(CLASSES), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = [x]
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        skips.pop()  # the bottleneck is not a skip
        for up in self.up_blocks:
            x = up(x, skips.pop())
        return self.out_conv(x)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@register_model
class Choroidalyzer(SemanticInstanceSegmentation, CapabilityModel):
    """Choroid region, choroidal vessels and fovea on OCT B-scans, as instances."""

    semantic_classes = CLASSES
    model_info = InstanceSegmentationModelInfo(
        registry_key="choroidalyzer",
        name="Choroidalyzer",
        description=(
            "Segments the choroid region, the choroidal vessels and the fovea on retinal OCT "
            "B-scans (Engelmann et al., IOVS 2024). A semantic model: each connected area it "
            "finds becomes one object, so vessels that touch arrive as one object to split."
        ),
        info_url="https://github.com/justinengelmann/Choroidalyzer",
        usage_tip="For single OCT B-scans (slices), not fundus or IR images.",
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "ophthalmology",
            "modality": "oct",
            "publisher": "Engelmann et al.",
            "architecture": "U-Net",
            "semantic_to_instance": "connected-components",
        },
        status="ready",
        trainable=False,
        architecture="U-Net",
        input_resolution=[768, 768],
        performance=ModelPerformance(num_parameters=788_283),
    )

    _unpicklable_attrs = ("_net",)

    def __init__(self) -> None:
        super().__init__()
        self._net: Optional[_ChoroidUNet] = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._net = None

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_weights(self) -> None:
        # The torch hub cache, where upstream's own loader keeps the file too.
        checkpoint = os.path.join(torch.hub.get_dir(), "checkpoints", "choroidalyzer_model_weights.pth")
        if not _file_has_sha256(checkpoint, WEIGHTS_SHA256):
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, checkpoint, progress=False)
        if not _file_has_sha256(checkpoint, WEIGHTS_SHA256):
            raise RuntimeError(f"Choroidalyzer weights at {checkpoint} do not match the pinned SHA-256.")
        net = _ChoroidUNet()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self._net = net.eval().to(self._device())

    @torch.inference_mode()
    def class_scores(self, image: np.ndarray) -> np.ndarray:
        """Run the network on an image and return ``(3, H, W)`` class probabilities.

        The network halves the resolution seven times, so the scan is padded with black
        (OCT background) to a multiple of 128 on both sides and the result cropped back.
        """
        self._ensure_weights()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        multiple = 2 ** _DEPTH
        pad_h, pad_w = -height % multiple, -width % multiple
        top, left = pad_h // 2, pad_w // 2
        padded = np.pad(gray, ((top, pad_h - top), (left, pad_w - left)))

        x = torch.from_numpy(padded).to(self._device(), torch.float32)
        x = (x / 255.0 - 0.5) / 0.5
        logits = self._net(x[None, None])[0]
        probs = logits.sigmoid()[:, top:top + height, left:left + width]
        return probs.float().cpu().numpy()


def _file_has_sha256(path: str, expected: str) -> bool:
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest() == expected
    except FileNotFoundError:
        return False
