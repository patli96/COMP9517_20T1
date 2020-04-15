from typing import Dict, Tuple, List, Any

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        detections: List[Tuple[int, int, int, int]],  # List[ (y1, x1, y2, x2), ... ]
        detection_records: List[List[Tuple[int, int, int, int]]],  # List[ detections ], the current is [0]
        detection_frame_deltas: List[int],  # List[ detection_frame_delta ], the current is [0]
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        previous_pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],  # List[ pedestrians ], previous ones
        previous_pedestrian_frame_deltas: List[int],  # List[ pedestrian_frame_delta ], for previously computed ones
        previous_tracks: Dict[int, List[Tuple[int, int]]],  # Dict{ pedestrian_id: [(y, x), ...], ... }
        # The new tracks should be modified based on the previous_tracks
        # Both tracks and pedestrians shared the same frame_delta_records as they both came from trackers
        storage: Dict[str, Any],  # It will be handed over to the next detector, please mutate this object directly
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[int, List[Tuple[int, int]]]]:
    pedestrians = dict(zip(list(range(len(detections))), detections))
    return pedestrians, {}