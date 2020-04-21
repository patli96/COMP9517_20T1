import socket
from typing import Tuple, List, Any, Dict
import json

import numpy as np
import http.client

CSP_SERVER_PORT = 8098
TIMEOUT_SECONDS = 3

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
    try:
        if storage.get('csp_server_running', None) is None:
            print('Connecting to the CSP server, please launch it using: \n'
                  ' -iw ' + str(image.shape[1]) +
                  ' -ih ' + str(image.shape[0]) +
                  ' --port ' + str(CSP_SERVER_PORT) +
                  ' --weight YOUR_WEIGHT_FILE_PATH'
                  )
            storage['csp_server_running'] = http.client.HTTPConnection(
                'localhost', CSP_SERVER_PORT, timeout=TIMEOUT_SECONDS
            )
            while True:
                try:
                    storage['csp_server_running'].connect()
                    storage['csp_server_running'].request('GET', '/ready')
                    if bytes.decode(storage['csp_server_running'].getresponse().read(), 'utf-8').strip() != 'true':
                        print('Wrong status of CSP server.')
                        continue
                    print('CSP server connected.')
                    break
                except socket.timeout:
                    continue

        client = storage['csp_server_running']
        client.request('POST', '/process', json.dumps(image.tolist()))
        raw_detections = np.array(json.loads(client.getresponse().read()), float)
        # Format: [y1, x1, y2, x2, confidence]
        # TODO: evaluations
        detections = list(map(tuple, np.round(raw_detections[:, [0, 1, 2, 3]]).astype(int)))
        return detections
    except socket.timeout:
        if storage.get('csp_server_running', None) is not None:
            storage['csp_server_running'].close()
        storage['csp_server_running'] = None
        return []
