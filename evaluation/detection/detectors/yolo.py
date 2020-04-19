import numpy as np
import cv2
from datetime import datetime

net = cv2.dnn.readNet("../../pedestrian_monitor/detectors/yolov3.weights", "../../pedestrian_monitor/detectors/yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(raw, conf_threshold=0.3, img_background=[]):
    start = datetime.now()
    img = cv2.imread(raw)

    # Background removal
    # sub = np.subtract(img, img_background)
    # sub = (sub - sub.min()) / (sub.max() - sub.min()) * 255
    # sub = np.float32(sub.reshape(img.shape))
    # img = sub

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    class_ids = []
    confidences = []
    rects = []

    # loop over the detections
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = x1 + w
                y2 = y1 + h
                rects.append([x1, y1, x2, y2, confidence])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    pick = nms(rects, 0.5)
    return pick

def nms(rects, threshold):
    rects = np.asarray(rects)
    if len(rects) == 0:
        return []
    pick = []
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # print('overlap', overlap)
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > threshold:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return rects[pick]
