from typing import Dict, Tuple, List, Any

import numpy as np
import math

def get_center(box: Tuple[int, int, int, int]):
    y = ((box[2] - box[0]) / 2) + box[0]
    x = ((box[3] - box[1]) / 2) + box[1]
    return (y, x)

def distance(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]):
    y1, x1 = get_center(b1)
    y2, x2 = get_center(b2)
    return math.sqrt((y1 - y2)**2 + (x1 - x2)**2)


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

    tolerance = 20
    counter = 0
    if 'pedestrian_counter' in storage:
        counter = storage['pedestrian_counter']

    pedestrians = {}
    tracks = previous_tracks
    remaining_detections = [i for i in range(len(detections))]
    if (len(previous_pedestrian_records) > 0):
        scores = {}
        for id, box2 in previous_pedestrian_records[0].items():
            for idx, box1 in enumerate(detections):

                dist = distance(box1, box2)
                if id not in scores or scores[id][0] > dist:
                    # if it is closer to another pedestrian -> continue
                    scores[id] = (dist, idx)

        # now deal with duplicates and people leaving / entering


        for id, (dist, idx) in scores.items():
            if dist < tolerance:
                pedestrians[id] = detections[idx]
                tracks[id].insert(0, get_center(detections[idx]))
                remaining_detections.remove(idx)

    for idx in remaining_detections:
        pedestrians[counter] = detections[idx]
        tracks[counter] = [get_center(detections[idx])]
        counter += 1

    storage['pedestrian_counter'] = counter

    return pedestrians, tracks
