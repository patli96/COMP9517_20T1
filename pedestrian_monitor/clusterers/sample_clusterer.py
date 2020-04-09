from typing import Dict, Tuple, List

import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        pedestrians: Dict[int, Tuple[int, int, int, int]],  # Dict{ pedestrian_id: (y1, x1, y2, x2), ... }
        pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],  # List[ pedestrians ], the current is [0]
        pedestrian_frame_deltas: List[int],  # List[ pedestrian_frame_delta ], the current is [0]
        tracks: Dict[int, List[Tuple[int, int]]],  # Dict{ pedestrian_id: [(y, x), ...], ... }
        # There is no track_records as each (y, x) is a record, and tracks have all pedestrians' tracks
        # Tracks of those who disappeared or moved out of the image will be removed, and they're useless
        # Both tracks and pedestrians shared the same frame_delta_records as they both came from trackers
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        group_records: List[Dict[int, List[int]]],  # List[ groups ], previously computed groups
        group_frame_deltas: List[int],  # List[ frame_delta ], for previously computed groups
) -> Dict[int, List[int]]:
    # groups is a dict
    # its indexes are group ids, which should be stable between frames
    # the group id does not need to start at 0, but it needs to be unique and stable
    # Example: 0: [0, 1] -> 0: [0, 1, 8] -> 0: [1, 8]
    # its values are lists that contain >=2 unique pedestrian id
    # Example: [1] is not a group and should not be returned
    groups = {
        0: [0, 1],
        1: [2, 6, 4],
        2: [5, 3],
    }
    return groups