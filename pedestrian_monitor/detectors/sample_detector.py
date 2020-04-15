from typing import Tuple, List, Any, Dict

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        features: np.ndarray,  # The features extracted from preprocessors
        feature_records: List[np.ndarray],  # List[ features ], the current is [0]
        feature_frame_deltas: List[int],  # List[ frame_delta ], for feature_records
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        image_records: List[np.ndarray],  # List[ images ], previously displayed images, the current is [0]
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        previous_detection_records: List[Tuple[int, int, int, int]],  # List[ detections ], previously computed results
        previous_detection_frame_deltas: List[int],  # List[ frame_delta ], for previously computed detections
        storage: Dict[str, Any],  # It will be handed over to the next detector, please mutate this object directly
) -> List[Tuple[int, int, int, int]]:
    # detections is a list
    # each value is the bounding box of a pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    detections = [(2, 4, 80, 100)]
    return detections
