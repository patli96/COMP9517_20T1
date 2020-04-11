from typing import Dict, Tuple, List

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        detections: List[Tuple[int, int, int, int]],  # List[ (y1, x1, y2, x2), ... ]
        detection_records: List[List[Tuple[int, int, int, int]]],  # List[ detections ], the current is [0]
        detection_frame_deltas: List[int],  # List[ detection_frame_delta ], the current is [0]
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        previous_pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],  # List[ pedestrians ], previously computed
        previous_pedestrian_frame_deltas: List[int],  # List[ pedestrian_frame_delta ], for previously computed pedestrians
        previous_tracks: Dict[int, List[Tuple[int, int]]],  # Dict{ pedestrian_id: [(y, x), ...], ... }
        # The new tracks should be modified based on the previous_tracks
        # Both tracks and pedestrians shared the same frame_delta_records as they both came from trackers
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[int, List[Tuple[int, int]]]]:
    # pedestrians is a dict
    # its indexes are pedestrian ids, which should be stable between frames
    # the pedestrian id does not need to start at 0, but it needs to be unique and stable
    # Example: 0: (2, 4, 6, 8) -> 0: (3, 6, 9, 11), the index 0 means the same person moves
    # its values are tuples that define the bounding box of this pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    pedestrians = {
        0: (2, 4, 6, 8),
    }
    # tracks is a dict
    # its indexes are pedestrian ids, be aware that only currently detected pedestrians can have their tracks.
    # which means set(list(tracks.keys())) == set(list(pedestrians.keys()))
    # its values are lists that contain tuple of points, their format is (y, x)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    tracks = {
        0: [(3, 3), (5, 5)]
    }
    return pedestrians, tracks
