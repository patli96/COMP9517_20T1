import socket
from typing import Tuple, List, Any, Dict
import json
from pathlib import Path

from ..console_arguments import get_args

import numpy as np
import http.client

MIN_CONFIDENCE = 0.5

CSP_SERVER_PORT = 8097
TIMEOUT_SECONDS = 600

EVALUATION = True
EVALUATION_FOLDER = 'CSP_' + str(MIN_CONFIDENCE)

if get_args().preprocessor == 'background_subtraction':
    EVALUATION_FOLDER += '_no_background'

GROUND_TRUTH_OUTPUT = True

EVALUATION_PATH = (Path(__file__).parent.parent.parent
                   / 'evaluation' / 'detection' / 'detection_results' / EVALUATION_FOLDER
                   ).resolve().absolute()

GROUND_TRUTH_PATH = (Path(__file__).parent.parent.parent / 'ground_truth').resolve().absolute()

# Mkdir if not exist
EVALUATION_PATH.mkdir(parents=True, exist_ok=True)


def _get_evaluation_file_path(image_index: int):
    return str(EVALUATION_PATH / (str(image_index + 1).zfill(6) + '.txt'))


def _get_gt_file_path():
    return str(GROUND_TRUTH_PATH / (EVALUATION_FOLDER + '.txt'))


def _to_evaluation_format(detections: np.ndarray):
    return '\n'.join(
        list(map(
            lambda x: 'person {conf} {x1} {y1} {x2} {y2}'.format(
                y1=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.float32(x[0]))
                ]][0],
                x1=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.float32(x[1]))
                ]][0],
                y2=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.float32(x[2]))
                ]][0],
                x2=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.float32(x[3]))
                ]][0],
                conf=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.float32(x[4]))
                ]][0],
            ),
            detections.tolist(),
        ))
    ) + '\n'


def _to_gt_format(image_index: int, detections: np.ndarray):
    return '\n'.join(
        list(map(
            lambda x: '{frame},0,{x1},{y1},{w},{h},1,0.0,0.0,0'.format(
                frame=str(image_index + 1),
                y1=str(int(round(x[0]))),
                x1=str(int(round(x[1]))),
                w=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.abs(np.float32(x[3]) - np.float32(x[1])))
                ]][0],
                h=[y + '0' if y.endswith('.') else y for y in [
                    np.format_float_positional(np.abs(np.float32(x[2]) - np.float32(x[0])))
                ]][0],
            ),
            detections.tolist(),
        ))
    ) + '\n'


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
    if storage.get('not_first_run', None) is None:
        import os
        import glob

        with open(_get_gt_file_path(), 'w') as f:
            f.write('')

        files = glob.glob(str(EVALUATION_PATH / '*'))
        for f in files:
            os.remove(f)
        storage['not_first_run'] = True
    # detections is a list
    # each value is the bounding box of a pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    try:
        if storage.get('csp_server_running', None) is None:
            print('Connecting to the CSP server, please launch it using: \n'
                  ' -iw ' + str(features.shape[1]) +
                  ' -ih ' + str(features.shape[0]) +
                  ' --port ' + str(CSP_SERVER_PORT) +
                  ' --weight YOUR_WEIGHT_FILE_PATH'
                  )
            storage['csp_server_running'] = http.client.HTTPConnection(
                'localhost', CSP_SERVER_PORT, timeout=1
            )
            while True:
                try:
                    storage['csp_server_running'].connect()
                    storage['csp_server_running'].request('GET', '/ready')
                    if bytes.decode(storage['csp_server_running'].getresponse().read(), 'utf-8').strip() != 'true':
                        print('Wrong status of CSP server.')
                        continue
                    storage['csp_server_running'].close()
                    storage['csp_server_running'] = http.client.HTTPConnection(
                        'localhost', CSP_SERVER_PORT, timeout=TIMEOUT_SECONDS
                    )
                    storage['csp_server_running'].connect()
                    print('CSP server connected.')
                    break
                except (socket.timeout, OSError):
                    if storage.get('csp_server_running', None) is not None:
                        storage['csp_server_running'].close()
                    storage['csp_server_running'] = http.client.HTTPConnection(
                        'localhost', CSP_SERVER_PORT, timeout=1
                    )
                    continue

        client = storage['csp_server_running']
        client.request('POST', '/process', json.dumps({'img': features.tolist(), 'score': MIN_CONFIDENCE}))

        # Format: [y1, x1, y2, x2, confidence]
        raw_detections = np.array(json.loads(client.getresponse().read()), float)

        if EVALUATION:
            with open(_get_evaluation_file_path(image_index), 'w') as f:
                f.write(_to_evaluation_format(raw_detections))

        if GROUND_TRUTH_OUTPUT:
            with open(_get_gt_file_path(), 'a') as f:
                f.write(_to_gt_format(image_index, raw_detections))

        if len(raw_detections) <= 0:  # If nothing detected
            return []

        detections = list(map(tuple, np.round(raw_detections[:, [0, 1, 2, 3]]).astype(int)))
        return detections
    except (socket.timeout, OSError):
        if storage.get('csp_server_running', None) is not None:
            storage['csp_server_running'].close()
        storage['csp_server_running'] = None
        return []
