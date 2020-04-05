import os
import matplotlib.pyplot as plt

FILE_DIRS = os.path.dirname(os.path.realpath('__file__'))
SEQUENCES_BASE_PATH = '../../../sequence'

sorted_sequences = sorted(os.listdir(
    SEQUENCES_BASE_PATH), key=lambda x: int(os.path.splitext(x)[0]))

img = None
for seq in sorted_sequences:
    imgPath = SEQUENCES_BASE_PATH + '/' + seq
    im = plt.imread(imgPath)
    if img is None:
        img = plt.imshow(im)
    else:
        img.set_data(im)
    plt.pause(.1)
    plt.draw()
