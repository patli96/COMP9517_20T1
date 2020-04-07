import numpy as np
import cv2


def hog_detect(raw):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    img = cv2.imread(raw)
    rects, weights = hog.detectMultiScale(img, winStride=(4, 4),
                                          padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img
