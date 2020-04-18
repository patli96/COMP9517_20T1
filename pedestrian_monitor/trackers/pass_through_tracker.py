from typing import Dict, Tuple, List, Any
import math

import numpy as np

SHOW_TRACKS = True  # Change this to False if you wanna completely hide tracks
TELEPORT_MIN_DISTANCE = 20  # This assumes that one pedestrian can never move more than these pixels in one frame gap

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
    if type(detections) == dict:
        pedestrians = dict(detections)
    else:
        pedestrians = dict(zip(list(range(len(detections))), detections))
        pedestrians = dict((p_id, box) for (p_id, box) in pedestrians.items() if box is not None)
    if not SHOW_TRACKS:
        return pedestrians, {}

    tracks = previous_tracks
    for (p_id, box) in pedestrians.items():
        center = (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))
        tracks.setdefault(p_id, [])

        # Clear the tracker if the pedestrian "teleport"
        if len(tracks[p_id]) > 0 and math.hypot(
                tracks[p_id][0][1] - center[1], tracks[p_id][0][0] - center[0]
        ) > (TELEPORT_MIN_DISTANCE * frame_delta):
            tracks[p_id] = [center]
        else:
            tracks[p_id].insert(0, center)
            # Limit the max length to avoid extremely long tracks
            tracks[p_id] = tracks[p_id][:300]

    tracks = dict((p_id, points) for (p_id, points) in tracks.items() if pedestrians.get(p_id, None) is not None)

    return pedestrians, tracks
