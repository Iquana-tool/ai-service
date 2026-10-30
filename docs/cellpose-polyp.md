# Provisional Cellpose-SAM polyp inference

The experimental provider is registered as `cpsam_polyp_v3_b_seed17` for the
`instance-segmentation` task. It proposes editable polyp contours; the existing
preview workflow persists them only after user acceptance. The backend supplies
the selected dataset label as `request.label`, so the provider does not assume a
benchmark category ID.

## Runtime and weights

Install the pinned optional inference dependency in the AI service environment:

```bash
uv sync --extra cellpose-polyp
```

The lock selects `torch==2.14.0+cu130` on Linux. Deploy to a host with a CUDA
driver compatible with CUDA 13.0; the provider reports unavailable CUDA rather
than silently switching to CPU.

Configure `IQUANA_CELLPOSE_POLYP_CHECKPOINT` to the fine-tuned `cpsam_best`
artifact, and set `IQUANA_CELLPOSE_POLYP_CHECKPOINT_SHA256` to its expected
64-character hexadecimal SHA-256. The SHA setting defaults to the recorded
provisional checkpoint digest below; replacements must set both path and SHA.
There is no default path: until the variable points at an existing file (and the
extra is installed), the model registers as `not_ready` and is not offered. Restart
the service after setting it. The file has default SHA-256
`8d17682741428d0c71b93dc6112de9ff9d6595fc586fe797bf4566423b788eba`; a missing
or mismatched file fails with a clear error. The provider builds the pinned
Cellpose 4.2.1.1 CPSAM architecture directly and strictly loads the complete
fine-tuned state, so it never downloads or substitutes stock Cellpose weights
and needs no separate base checkpoint.

CUDA is explicit by default (`IQUANA_CELLPOSE_POLYP_DEVICE=cuda:0`). Set the
variable to `cuda:N` to select a device or to `cpu` for an intentional CPU run;
there is no automatic CPU fallback. The loaded model is cached in its worker,
and a worker serializes Cellpose inference to bound GPU memory use. Its cache
identity is the resolved checkpoint path, verified SHA-256, and selected
device. Restart workers after changing any of those settings; predictions
themselves are not cached.

The verified checkpoint was strictly loaded on CPU with the trained
`(1, 80, 80, 1024)` positional grid. A real inference comparison was not run:
the development host exposes CUDA driver version 12.6 (12060), which cannot
initialize the locked CUDA 13.0 runtime, and the toaster GPUs were assigned to
active training jobs during verification. No latency or GPU-output claim is
made from the CPU load check.

## Prediction behavior

Images are converted from BGR to RGB and kept at native resolution. Each image
side must be at least 640 pixels. The provider runs 640×640 outer tiles with
128-pixel overlap, in row-major order. Instance centroids belong to half-open
tile cores whose cuts are the integer midpoints of adjacent tile starts and
ends; this assigns an overlapping detection to one tile only.

The Cellpose call uses `batch_size=1`, `bsize=640`, `tile_overlap=0.1`, RGB
`channel_axis=2`, `resample=True`, normalization, `diameter=None`, cell-probability
threshold `0.0`, flow threshold `0.4`, `min_size=15`, `max_size_fraction=0.4`,
and `augment=False`. The checkpoint's trained 80×80 positional embeddings are
loaded after adapting only the fresh architecture's initial 32×32 grid.

Candidate confidence is the mean sigmoid of the raw cell-probability logits
inside each mask. The reference candidate floor is `0.001`; the provisional
application threshold defaults to `0.05` and can be changed independently.

When the request includes `parameters.parent_regions`, it must contain native
image-size COCO RLEs. The provider runs only tiles intersecting their union and
keeps a mask only when its rounded, clipped centroid falls inside that
undilated union (margin 0). An omitted or empty list means whole-image
inference. These ROIs must come from explicitly selected application outlines;
the provider does not use benchmark fragment annotations.

The response contains `contours` plus `prediction_metadata`. Metadata retains
each exact native-size mask as COCO RLE with its score and stable
`source_instance_id`; contour polygons derived from the same mask share that
ID. The shared contour serializer normalizes polygon x/y coordinates, fills
holes, and emits disconnected components as separate contour items. The raw
RLE preserves the original mask, including holes and disconnected regions.
