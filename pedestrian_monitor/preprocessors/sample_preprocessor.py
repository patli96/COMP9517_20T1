from typing import Tuple, List, Dict, Any

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        previous_feature_records: List[np.ndarray],  # List[ images ], previously displayed images
        previous_feature_frame_deltas: List[int],  # List[ frame_delta ], for previously computed detections
        storage: Dict[str, Any],  # It will be handed over to the next preprocessor, please mutate this object directly
) -> np.ndarray:
    # features is a numpy array
    # It will be used in detectors for pedestrian detection.
    # To skip the whole preprocess, you may simply return a features
    features = image
    return features
