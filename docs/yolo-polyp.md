# YOLO11m-seg polyp provider

## Deployment

The fine-tuned checkpoint is hosted at
[`iquana/iquana-yolo-polyp`](https://huggingface.co/iquana/iquana-yolo-polyp)
as `yolo_polyp_finetuned.pt`. Follow the [Hugging Face deployment setup](../README.md#polyp-model-checkpoints-from-hugging-face)
to download the pinned artifact and configure its local path. The setup requires
read access to the private repository; keep credentials and downloaded weights
outside Git.

Install the optional `yolo-polyp` extra (`ultralytics==8.4.165`). The provider
uses the service's PyTorch installation. The package and pretrained weights use
the Ultralytics AGPL-3.0/Enterprise licensing model. It defaults to `cuda:0`,
which requires an available CUDA GPU; set `IQUANA_YOLO_POLYP_DEVICE=cpu` for CPU
inference or `cuda:N` for another GPU. Set `IQUANA_YOLO_POLYP_CHECKPOINT` to the
checkpoint path. Until the extra and checkpoint are available, the model is
`not_ready` and is not offered. The provider verifies the checkpoint SHA-256
(`7e9ac2eec159b4d227d1b8c6553de224a982f420f96b738362ef04bbfad08edf`) and never
substitutes stock weights. Restart the service workers after changing its
checkpoint or device settings.

## Use in Segment labels

The stable registry key is `yolo_polyp_v3_c_seed1`. Configure the model's
`polyp` class mapping to a dataset label first, then choose that label and YOLO
in its settings under Segment labels. Predictions are saved directly as ordinary
editable Objects with the mapped label.

Use **Use hierarchy** to limit inference to saved parent outlines; turn it off
to run on the whole image. If matching objects already exist in the scope,
choose **Patch** to add predictions that do not overlap them, or **Override** to
replace them with the new results.

Inference uses native-resolution 640-pixel tiles with 128-pixel overlap. Images
smaller than 640 pixels on either side are padded without rescaling, and saved
contours are mapped back to the original image dimensions.
