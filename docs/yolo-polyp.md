# YOLO C seed 1 polyp provider

Registry key: `yolo_polyp_v3_c_seed1`. Displayed beside Cellpose in Instance Segmentation; runs save ordinary editable Objects, with dataset label mapping for model class `polyp` and provenance `instance_segmentation`.

Install the optional `yolo-polyp` extra (Ultralytics 8.4.165). The package and pretrained weights use the Ultralytics AGPL-3.0/Enterprise licensing model. Runtime imports are lazy, so unrelated providers do not require this extra.

Set `IQUANA_YOLO_POLYP_CHECKPOINT` to the trained C seed 1 checkpoint, outside Git. Local default: `/home/shivamt/iquana-dev/model-artifacts/yolo-polyp/yolo_C_seed1.pt`. Expected SHA-256: `7e9ac2eec159b4d227d1b8c6553de224a982f420f96b738362ef04bbfad08edf`. Missing/mismatched files fail explicitly; stock weights are not substituted.

`IQUANA_YOLO_POLYP_DEVICE` defaults to `cuda:0`; set `cpu` for an explicit CPU run. Restart the worker after changing checkpoint/device settings. Each worker serializes YOLO inference and reuses its loaded network.

The adapter matches the recorded benchmark inference configuration: native 640-pixel BGR tiles with 128-pixel overlap, confidence candidate floor .001, max_det 1000, NMS IoU .7, retina masks with .5 decoding threshold, FP32 inference without augmentation, half-open centroid tile ownership, exact native-size RLE metadata, and normalized application contours. The application threshold defaults to the recorded validation operating point .25. Each image side must be at least 640 pixels.

Saved same-image ancestor outlines may bound inference through native-size COCO `parent_regions` RLEs (margin 0). No ROI means Whole image. Benchmark annotations and held-out images are not consulted at runtime. Cancellation, retry, duplicate suppression, ancestry validation, and normal editing reuse the existing polyp workflow.

The local checkpoint hash was verified and real weights loaded on CPU. Synthetic-image inference completed both on CPU and through the running HTTP service on the RTX 3090 (`cuda:0`). This is a runtime smoke test, not an accuracy or representative performance measurement. Existing local Torch 2.6/cu124 and tracked Torch/CUDA dependency versions were retained.
