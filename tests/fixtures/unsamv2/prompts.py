"""Prompts for the UnSAMv2 parity test, in pixels of the 600 x 400 fixture images.

The images are SAM 2's notebook examples (facebookresearch/sam2, Apache-2.0),
downscaled. ``reference_unsamv2_plus.npz`` holds the masks the authors'
predictor returns for every image x prompt x granularity, in this order.
"""
IMAGES = ["truck.jpg", "groceries.jpg", "cars.jpg"]

GRANULARITIES = [0.1, 0.5, 1.0]

PROMPTS = {
    "truck.jpg": [
        {"points": [[167, 125]], "labels": [1]},
        {"box": [142, 200, 233, 292]},
        {"points": [[167, 125], [100, 220]], "labels": [1, 0]},
    ],
    "groceries.jpg": [
        {"points": [[300, 200]], "labels": [1]},
        {"box": [150, 100, 330, 300]},
        {"points": [[300, 200], [420, 120]], "labels": [1, 1]},
    ],
    "cars.jpg": [
        {"points": [[153, 200]], "labels": [1]},
        {"box": [20, 73, 460, 227]},
        {"points": [[153, 200]], "labels": [1], "box": [20, 73, 460, 227]},
    ],
}
