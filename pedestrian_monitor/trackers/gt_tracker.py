from typing import Tuple, List, Any, Dict
from pathlib import Path
import csv
import copy

import numpy as np


def _load_gt():
    gt_file_path = Path(__file__).resolve().parent.parent.parent / 'ground_truth' / 'gt.txt'
    gt_columns = ['frame', 'p_id', 'left', 'top', 'width', 'height', 'used', 'x', 'y', 'z']
    gt_pedestrians = {}
    gt_tracks = {}
    if gt_file_path.is_file():
        with open(gt_file_path, newline='') as csvfile:
            gt_reader = csv.reader(csvfile, delimiter=',')
            for row in gt_reader:
                row = dict(zip(gt_columns, row))
                row['frame'] = int(row['frame']) - 1
                row['p_id'] = int(row['p_id'])
                row['left'] = round(float(row['left']))
                row['top'] = round(float(row['top']))
                row['width'] = round(float(row['width']))
                row['height'] = round(float(row['height']))
                row['bottom'] = row['top'] + row['height']
                row['right'] = row['left'] + row['width']
                row['used'] = int(row['used']) == 1
                row['x'] = float(row['x'])
                row['y'] = float(row['y'])
                row['z'] = float(row['z'])
                gt_pedestrians.setdefault(row['frame'], {})
                gt_pedestrians[row['frame']][row['p_id']] = (row['top'], row['left'], row['bottom'], row['right'])
                gt_tracks.setdefault(row['frame'], copy.deepcopy(gt_tracks.get(row['frame'] - 1, {})))
                gt_tracks[row['frame']].setdefault(row['p_id'], [])
                gt_tracks[row['frame']][row['p_id']].insert(0,
                    (round(row['top'] + 0.5 * row['height']), round(row['left'] + 0.5 * row['width']))
                )
                # Limit the maximum length of tracks
                gt_tracks[row['frame']][row['p_id']] = gt_tracks[row['frame']][row['p_id']][:300]
    gt_tracks_copy = copy.deepcopy(gt_tracks)
    for image_index in gt_tracks.keys():
        for p_id in gt_tracks[image_index].keys():
            if gt_pedestrians.get(image_index, None) is None:
                del gt_tracks_copy[image_index][p_id]
                continue
            if gt_pedestrians[image_index].get(p_id, None) is None:
                del gt_tracks_copy[image_index][p_id]
                continue
    return gt_pedestrians, gt_tracks_copy


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
    # pedestrians is a dict
    # its indexes are pedestrian ids, which should be stable between frames
    # the pedestrian id does not need to start at 0, but it needs to be unique and stable
    # Example: 0: (2, 4, 6, 8) -> 0: (3, 6, 9, 11), the index 0 means the same person moves
    # its values are tuples that define the bounding box of this pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    if storage.get('gt_pedestrians', None) is None or storage.get('gt_tracks', None) is None:
        storage['gt_pedestrians'], storage['gt_tracks'] = _load_gt()
    pedestrians = storage['gt_pedestrians'].get(image_index, dict())
    # tracks is a dict
    # its indexes are pedestrian ids, be aware that only currently detected pedestrians can have their tracks.
    # which means set(list(tracks.keys())) == set(list(pedestrians.keys()))
    # its values are lists that contain tuple of points, their format is (y, x)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    tracks = storage['gt_tracks'].get(image_index, dict())
    return pedestrians, tracks
