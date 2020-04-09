from typing import Tuple, List

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        detection_records: List[Tuple[int, int, int, int]],  # List[ detections ], previously computed detections
        detection_frame_deltas: List[int],  # List[ frame_delta ], for previously computed detections
) -> List[Tuple[int, int, int, int]]:
    # detections is a list
    # each value is the bounding box of a pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    detections = [(2, 4, 8, 10)]
    return detections
