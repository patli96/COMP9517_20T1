from typing import Tuple, List, Any, Dict
import os
import numpy as np
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
net = cv2.dnn.readNet(dir_path + "/yolov3.weights",
                      dir_path + "/yolov3.cfg")

# Get only the the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def nms(  # This function applies non-maximum suppresssion to remove duplicated bounding boxes
        rects, threshold):
    rects = np.asarray(rects)
    if len(rects) == 0:
        return []
    filtered_rects = []
    y1 = rects[:, 0]
    x1 = rects[:, 1]
    y2 = rects[:, 2]
    x2 = rects[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort all rects by the bottom right points
    rect_idxs = np.argsort(y2)
    while len(rect_idxs) > 0:
        last = len(rect_idxs) - 1
        i = rect_idxs[last]
        filtered_rects.append(i)
        suppress_rects = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # get current index
            j = rect_idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            width = max(0, xx2 - xx1 + 1)
            heigth = max(0, yy2 - yy1 + 1)
            # compute overlap between the computed bounding box and the bounding box in the area list
            overlap = float(width * heigth) / area[j]
            # if there overlap is greater than the threshold, add current bounding box to the suppression array
            if overlap > threshold:
                suppress_rects.append(pos)
        # delete all indexes from the index list that are in the suppression list
        rect_idxs = np.delete(rect_idxs, suppress_rects)
    # return only the filtered bounding boxes
    return rects[filtered_rects]


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        features: np.ndarray,  # The features extracted from preprocessors
        # List[ features ], the current is [0]
        feature_records: List[np.ndarray],
        # List[ frame_delta ], for feature_records
        feature_frame_deltas: List[int],
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        # List[ images ], previously displayed images, the current is [0]
        image_records: List[np.ndarray],
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        # List[ detections ], previously computed results
        previous_detection_records: List[Tuple[int, int, int, int]],
        # List[ frame_delta ], for previously computed detections
        previous_detection_frame_deltas: List[int],
        # It will be handed over to the next detector, please mutate this object directly
        storage: Dict[str, Any],
) -> List[Tuple[int, int, int, int]]:
    # detections is a list
    # each value is the bounding box of a pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order

    height, width, channels = features.shape
    # TODO, abstract the resize factor, right now it is 0.5
    blob = cv2.dnn.blobFromImage(
<<<<<<< HEAD
        features, 1/255, (416, 416), (0, 0, 0), True, crop=False)
=======
        features, 1/255, (round(768/2), round(576/2)), (0, 0, 0), True, crop=False)
>>>>>>> origin/alex_experimentation
    net.setInput(blob)
    detections = net.forward(output_layers)

    rects = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # id 0 is the class_id for person object
            if confidence > 0.3 and class_id == 0:
                # Object detected
                x_center = int(detection[0] * width)
                y_center = int(detection[1] * height)
                rect_width = int(detection[2] * width)
                rect_height = int(detection[3] * height)
                # Rectangle coordinates
                x1 = int(x_center - rect_width / 2)
                y1 = int(y_center - rect_height / 2)
                x2 = x1 + rect_width
                y2 = y1 + rect_height
                rects.append((y1, x1, y2, x2, confidence))
    filtered_rects = nms(rects, 0.5)
    return filtered_rects
