from typing import Dict, Tuple, List, Any

import numpy as np
from scipy.spatial import cKDTree
import math

def get_center(box: Tuple[int, int, int, int]):
    y = ((box[2] - box[0]) / 2) + box[0]
    x = ((box[3] - box[1]) / 2) + box[1]
    return (y, x)

def distance(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]):
    y1, x1 = get_center(b1)
    y2, x2 = get_center(b2)
    return math.sqrt((y1 - y2)**2 + (x1 - x2)**2)

# from https://stackoverflow.com/questions/15363419/finding-nearest-items-across-two-lists-arrays-in-python/15366296
def nearest_neighbors_kd_tree(x, y, k) :
    x, y = map(np.asarray, (x, y))
    tree = cKDTree(y)
    scores, ordered_neighbors = tree.query(x, k)
    nearest_neighbor = np.empty((len(x),), dtype=np.intp)
    nearest_neighbor.fill(-1)
    used_y = set()
    for j, neigh_j in enumerate(ordered_neighbors) :
        for k in neigh_j :
            if k not in used_y :
                nearest_neighbor[j] = k
                used_y.add(k)
                break
    return scores, nearest_neighbor
# end from


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

    pedestrians = {}
    tracks = {}

    if 'pedestrian_counter' not in storage:
        storage['pedestrian_counter'] = 0

    det_points = np.array([get_center(d) for d in detection_records[0]])
    # print(det_points)
    if (len(detection_records) < 2):
        storage['pedestrian_counter'] = 0
        remaining_detections = np.lexsort((det_points[:,1], det_points[:,0]))
    else:
        prev_points = np.array([get_center(v) for k, v in previous_pedestrian_records[0].items()])
        # print(prev_points)
        # prev_points = np.array([get_center(d) for d in detection_records[1]])
        scores, ind = nearest_neighbors_kd_tree(prev_points, det_points, storage['pedestrian_counter'])
        relevant_keys = [k for k in previous_pedestrian_records[0].keys()]
        for leave_count in range(len(ind) - len(detections)):
            # print('someone left! detected', len(detections), 'but', len(ind), 'previously')
            worst1 = (0, -1)
            for id, s in enumerate(scores):
                if s[0] > worst1[0]:
                    worst1 = (s[0], id)
            prev_points2 = np.delete(prev_points, worst1[1], 0)
            scores2, ind2 = nearest_neighbors_kd_tree(prev_points2, det_points, storage['pedestrian_counter'])
            worst2 = (0, -1)
            for id, s in enumerate(scores2):
                if s[0] > worst2[0]:
                    worst2 = (s[0], id)
            # print('score 1:', worst1, 'score 2:', worst2)
            if (worst1 > worst2):
                # print('removing', worst1[1]+1, 'to achieve score', worst2[0])
                prev_points = prev_points2
                ind = ind2
                # print('before removal:', previous_pedestrian_records[0].keys())
                relevant_keys = [k for i, k in enumerate(relevant_keys) if i != worst1[1]]
                # print('remaining people:', relevant_keys)
            else:
                input('UNTESTED REMOVAL')
                # print('removing', worst2[1]+1, 'to achieve score', worst1[0])
                prev_points = prev_points[:-1]
                scores3, ind = nearest_neighbors_kd_tree(prev_points, det_points, storage['pedestrian_counter'])
                relevant_keys = relevant_keys[:-1]

        worst_best_score = np.max(scores[:, 0])
        if (worst_best_score > 40):
            print('bad score', worst_best_score)
            input()

        for i, prev_id in zip(ind, relevant_keys):
            if len(detections) <= i:
                input('Eeek! Should not occur.')
            # print(prev_id, '->', i)
            pedestrians[prev_id] = detections[i]
            if len(previous_tracks) < 1:
                tracks[prev_id] = [get_center(detections[i])]
            else:
                tracks[prev_id] = previous_tracks[prev_id] + [get_center(detections[i])]
        remaining_detections = [i for i in range(len(detections)) if i not in ind]

    # each person who enters the frame is treated as a new pedestrian

    # print('remaining:', remaining_detections, 'total', storage['pedestrian_counter'])

    for i in range(len(remaining_detections)):
        storage['pedestrian_counter'] += 1
        id = storage['pedestrian_counter']
        idx = remaining_detections[i]
        pedestrians[id] = detections[idx]
        tracks[id] = [get_center(detections[idx])]
        # print('adding detection', idx, 'as pedestrian', id)

    pedestrians = {k: pedestrians[k] for k in sorted(sorted(pedestrians.keys()))}

    # if (len(previous_pedestrian_records) > 0):
    #     print('previous')
    #     print({k: get_center(v) for k, v in previous_pedestrian_records[0].items()})
    #     print('detections')
    #     print({k+1: get_center(v) for k, v in enumerate(detections)})
    #     print('new')
    #     print({k: get_center(v) for k, v in pedestrians.items()})

    return pedestrians, tracks
