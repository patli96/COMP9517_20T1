
import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier('./models/haarcascade_fullbody.xml')


def haarcascade(raw):
    frame = cv2.imread(raw)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in bodies:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    return frame
