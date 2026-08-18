"""DataLoaders for instance segmentation tasks using torchvision and COCO datasets.

Lives in this service (not in ``iquana-toolbox``) because it is the only consumer
of ``torch``/``torchvision`` outside a model: the toolbox is also installed by the
backend, which does not train anything.
"""

from typing import Optional, Callable, Dict, Any, List
import numpy as np
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader


class COCOInstanceSegmentationDataset(CocoDetection):
    """
    PyTorch Dataset for COCO-style instance segmentation.

    Each item is ``(image, target)`` where:

    * ``image`` is an ``np.ndarray`` of shape ``(H, W, 3)`` (uint8, RGB).
    * ``target`` is ``{"masks": list[np.ndarray (H, W) uint8], "labels": list[int]}``
      with one binary mask per instance and its COCO ``category_id`` (the dataset
      label id). Masks are rasterised independently via ``coco.annToMask`` and may
      overlap, so nested/contained instances are preserved.

    Mapping ``category_id`` to a contiguous class index is left to the consumer
    (the model knows which labels it is training on).
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms_pipeline: Optional[Callable] = None,
    ):
        """
        Initialize COCO Instance Segmentation Dataset.

        Args:
            root (str): Root directory containing the images
            annFile (str): Path to the COCO annotation JSON file
            transforms_pipeline (Optional[Callable]): Optional transforms to apply to images
        """
        super().__init__(root=root, annFile=annFile)
        self.transforms_pipeline = transforms_pipeline

    def __getitem__(self, idx: int) -> tuple:
        """Return ``(image_np, {"masks": [...], "labels": [...]})`` for one sample."""
        image, anns = super().__getitem__(idx)  # PIL image, list of COCO annotation dicts

        image_np = np.array(image.convert("RGB"))
        target = self._process_targets(anns)

        if self.transforms_pipeline is not None:
            image_np = self.transforms_pipeline(image_np)

        return image_np, target

    def _process_targets(self, anns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rasterise each annotation to its own binary mask and keep its category id."""
        masks: List[np.ndarray] = []
        labels: List[int] = []

        for ann in anns:
            if "category_id" not in ann or "segmentation" not in ann:
                continue
            # annToMask handles both polygon and RLE segmentations and rasterises
            # at the annotation's image size.
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if mask.sum() == 0:
                continue
            masks.append(mask)
            labels.append(int(ann["category_id"]))

        return {"masks": masks, "labels": labels}


def get_coco_instance_segmentation_dataset(
    image_folder: str,
    annotation_file: str,
    transforms_pipeline: Optional[Callable] = None,
    **kwargs
) -> COCOInstanceSegmentationDataset:
    """
    Create and return a COCO instance segmentation dataset.

    This is a convenience function that creates a COCOInstanceSegmentationDataset
    with COCO-style annotations.

    Args:
        image_folder (str): Path to folder containing images
        annotation_file (str): Path to COCO annotation JSON file
        transforms_pipeline (Optional[Callable]): Optional transforms to apply
        **kwargs: Additional arguments passed to the dataset

    Returns:
        COCOInstanceSegmentationDataset: PyTorch dataset for instance segmentation

    Example:
        >>> dataset = get_coco_instance_segmentation_dataset(
        ...     image_folder='/path/to/images',
        ...     annotation_file='/path/to/annotations.json'
        ... )
        >>> image, target = dataset[0]
    """
    return COCOInstanceSegmentationDataset(
        root=image_folder,
        annFile=annotation_file,
        transforms_pipeline=transforms_pipeline,
    )


def get_coco_instance_segmentation_dataloader(
    image_folder: str,
    annotation_file: str,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    transforms_pipeline: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """
    Create and return a DataLoader for COCO instance segmentation.

    Args:
        image_folder (str): Path to folder containing images
        annotation_file (str): Path to COCO annotation JSON file
        batch_size (int): Batch size for the dataloader. Default: 8
        num_workers (int): Number of worker processes. Default: 0
        shuffle (bool): Whether to shuffle the dataset. Default: True
        transforms_pipeline (Optional[Callable]): Optional transforms to apply
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        DataLoader: PyTorch DataLoader for instance segmentation

    Example:
        >>> dataloader = get_coco_instance_segmentation_dataloader(
        ...     image_folder='/path/to/images',
        ...     annotation_file='/path/to/annotations.json',
        ...     batch_size=16,
        ...     num_workers=4
        ... )
        >>> for images, targets in dataloader:
        ...     pass
    """
    dataset = get_coco_instance_segmentation_dataset(
        image_folder=image_folder,
        annotation_file=annotation_file,
        transforms_pipeline=transforms_pipeline,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs
    )

    return dataloader
