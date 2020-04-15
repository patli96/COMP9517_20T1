from typing import Tuple, List, Any, Dict
from pathlib import Path
import csv

import numpy as np


def _load_gt():
    gt_file_path = Path(__file__).resolve().parent.parent.parent / 'ground_truth' / 'gt.txt'
    gt_columns = ['frame', 'p_id', 'left', 'top', 'width', 'height', 'used', 'x', 'y', 'z']
    gt_detections = {}
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
                gt_detections.setdefault(row['frame'], [])
                gt_detections[row['frame']].append((row['top'], row['left'], row['bottom'], row['right']))
    return gt_detections


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
    if storage.get('gt_detections', None) is None:
        storage['gt_detections'] = _load_gt()
    detections = storage['gt_detections'].get(image_index, list())
    return detections
