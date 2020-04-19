import sys
import os
import cv2
import numpy as np
from detectors.yolo import detect

SEQUENCES_BASE_PATH = '../../sequence'

sorted_sequences = sorted(os.listdir(
    SEQUENCES_BASE_PATH), key=lambda x: int(os.path.splitext(x)[0]))

# Prepare backgound
images = []
for seq in sorted_sequences:
    imgPath = SEQUENCES_BASE_PATH + '/' + seq
    image = cv2.imread(imgPath)
    images.append(image)

imgs_array = np.array([img[0] for img in images])
imgs_modes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=imgs_array)

def generate_bounding_boxes_file(folder_name, conf_thres, image_mod):
    for seq in sorted_sequences:
        imgPath = SEQUENCES_BASE_PATH + '/' + seq
        # image = cv2.imread(imgPath)
        rects = detect(imgPath, conf_thres, image_mod)
        f = open('./detection_results/' + folder_name +
                 '/' + seq.split('.')[0] + '.txt', 'a+')
        for j in range(len(rects)):
            x1, y1, x2, y2, conf = rects[j]
            line_template = "person %s %s %s %s %s" if j == len(
                rects) - 1 else "person %s %s %s %s %s\n"
            line = line_template % (conf, x1, y1, x2, y2)
            print(line)
            f.write(line)

generate_bounding_boxes_file('yolo', 0.3, imgs_modes)
