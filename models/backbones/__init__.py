"""Shared, in-process vision backbones for IQUANA AI models.

These are *libraries*, not services: a model wrapper imports a backbone and owns
an instance of it in the same process. Sharing the code here (rather than running
a separate backbone service) keeps features on the same device as the task head,
avoids serializing multi-megabyte feature tensors over the network, and lets
training backprop into the head without a process boundary.

They live in this service rather than in ``iquana-toolbox`` because they are the
only reason a consumer would need ``torch``/``transformers``: the toolbox is also
installed by the backend, which does not run a model. Keeping the heavy weights
here lets the toolbox stay a schema/registry library.
"""

from models.backbones.dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Backbone

__all__ = ["DINOv3Backbone", "DEFAULT_DINOV3_MODEL"]
