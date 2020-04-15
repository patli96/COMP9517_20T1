import glob
import os
import time
from pathlib import Path

import cv2 as cv
from natsort import natsorted

opencv_supported_formats = (
    '.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm',
    '.ppm', '.pxm', '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic'
)


def get_image_paths(path_str):
    path = Path(os.path.expandvars(path_str)).expanduser().absolute()
    image_paths = []
    if '*' in str(path) or '?' in str(path):
        for entry_str in glob.iglob(str(path)):
            if os.path.islink(entry_str) and os.path.exists(entry_str):
                entry_str = str(Path(entry_str).resolve().absolute())
            if os.path.isdir(entry_str):
                sub_dir = str(Path(entry_str).resolve().absolute())
                with os.scandir(sub_dir) as folder:
                    for sub_entry in folder:
                        if sub_entry.is_file()\
                                and os.path.splitext(sub_entry.name)[1].lower() in opencv_supported_formats:
                            image_paths.append(
                                str(Path(sub_entry.path + '/' + sub_entry.name).resolve().absolute())
                            )
            if os.path.isfile(entry_str):
                image_paths.append(str(Path(entry_str).resolve().absolute()))
    elif os.path.isfile(path):
        image_paths.append(str(path))
    elif os.path.isdir(path):
        with os.scandir(path) as folder:
            for sub_entry in folder:
                if sub_entry.is_file()\
                        and os.path.splitext(sub_entry.name)[1].lower() in opencv_supported_formats:
                    image_paths.append(str(Path(sub_entry.path + '/' + sub_entry.name).resolve().absolute()))
    return natsorted(image_paths)


class ImageFileIterator:
    def __init__(self, path_str, listen=False):
        self.path_str = path_str
        self.listen = listen
        self.image_paths = get_image_paths(path_str)
        self.index = 0
        self.incremental_index = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.image_paths):
            if self.listen:
                while True:
                    self.image_paths = set(get_image_paths(self.path_str)) - set(self.image_paths)
                    if len(self.image_paths) > 0:
                        self.index = 0
                        self.image_paths = list(self.image_paths)
                        break
                    time.sleep(0.2)
            else:
                raise StopIteration
        image = cv.imread(self.image_paths[self.index])
        self.index += 1
        self.incremental_index += 1
        return image, self.image_paths[self.index - 1], self.incremental_index