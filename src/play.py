import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.cv2_haarcascade import haarcascade
from models.cv2_hog import hog_detect

SEQUENCES_BASE_PATH = '../../sequence'

sorted_sequences = sorted(os.listdir(
    SEQUENCES_BASE_PATH), key=lambda x: int(os.path.splitext(x)[0]))

img = None
for seq in sorted_sequences:
    imgPath = SEQUENCES_BASE_PATH + '/' + seq
    # frame = haarcascade(imgPath)
    frame = hog_detect(imgPath)

    if img is None:
        img = plt.imshow(frame)
    else:
        img.set_data(frame)
    plt.pause(.15)
    plt.draw()
