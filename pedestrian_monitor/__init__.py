"""COMP9517 20T1 Project - Pedestrian Detecting, Tracking and Clustering"""
import copy
import sys
import time
from typing import List, Optional

import cv2 as cv
import numpy as np
from PyQt5 import QtCore

from .clusterers.sample_clusterer import compute as clusterer_compute
from .console_arguments import get_args
from .detectors.sample_detector import compute as detector_compute
from .draw_utils import get_status_text, apply_selection
from .draw_utils import resize_image, mark_selection, mark_pedestrians, mark_tracks, mark_groups, merge_overlay
from .file_handlers import get_image_paths, ImageFileIterator
from .mouse_handlers import mouse_callback_factory
from .trackers.sample_tracker import compute as tracker_compute

__version__ = '0.1.0'


def main():
    args = get_args()

    if args.path_video == '':
        iterator = ImageFileIterator(args.path, False)
    else:
        # TODO: VideoCapture iterator unfinished
        iterator = ImageFileIterator(args.path, False)

    print('')
    print('How to control:')
    print('[LeftMouseButton] Pause/Resume')
    paused = False

    def toggle_pause(x, y):
        nonlocal paused
        paused = not paused

    print('[RightMouseButton] Draw bounding box')
    selection_top_left = (0, 0)
    selection_bottom_right = (0, 0)

    def init_selection(x, y):
        nonlocal selection_top_left
        selection_top_left = (y, x)

    def drag_selection(x, y):
        nonlocal selection_bottom_right
        selection_bottom_right = (y, x)

    def cancel_selection(x, y):
        nonlocal selection_top_left, selection_bottom_right
        selection_top_left = (0, 0)
        selection_bottom_right = (0, 0)

    print('[MiddleMouseButton] Toggle all overlays')
    display_overlay = True

    def show_overlay(x, y):
        nonlocal display_overlay
        display_overlay = True

    def hide_overlay(x, y):
        nonlocal display_overlay
        display_overlay = False

    # Fix High DPI Scaling Issues, these should be called before creating any windows
    # These are called because OpenCV uses Qt, and doesn't export any Qt settings
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    window_name = 'Pedestrian Monitor'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback_factory(
        on_lmb_clicked=toggle_pause,
        on_rmb_clicked=cancel_selection,
        on_rmb_pressed=init_selection,
        on_rmb_dragging=drag_selection,
        on_mmb_pressed=hide_overlay,
        on_mmb_released=show_overlay,
    ))

    width = int(args.width)
    height = int(args.height)
    preserve_width = args.width == -1
    preserve_height = args.height == -1
    if args.fps <= 0:
        print('Invalid FPS: FPS has to be positive.', file=sys.stderr)
        sys.exit(1)
    frame_duration = 1 / float(args.fps)
    frame_skipping = args.frame_skipping
    first_frame = True
    image = np.zeros((576, 768, 3), np.uint8)
    image_path = ''

    max_records = 20  # Max number of records

    image_records = []

    detection_records = []
    detection_frame_deltas = []

    pedestrian_records = []
    pedestrian_frame_deltas = []

    tracks = {}

    group_records = []
    group_frame_deltas = []

    def store_record(record, frame_delta: int, records: List, frame_deltas: Optional[List]):
        records.insert(0, record)
        if len(records) > max_records:
            records = records[:max_records]
        if frame_deltas is not None:
            frame_deltas.insert(0, frame_delta)
            if len(frame_deltas) > max_records:
                frame_deltas = frame_deltas[:max_records]
        return records, frame_deltas

    if frame_skipping:
        # TODO: Multi-processing non-blocking player with frame dropping
        # TODO: init pools and managers
        pass
    while cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) > 0:
        frame_start_time = time.perf_counter()
        if not paused:
            image, image_path = next(iterator, (image, image_path))
        if first_frame:
            first_frame = False
            if width <= 0:
                width = image.shape[1]
            if height <= 0:
                height = image.shape[0]
            # Pause at first frame
            paused = True
        image = resize_image(image, width, height, preserve_width, preserve_height)
        image_records, _ = store_record(image.copy(), 1, image_records, None)

        overlay = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        overlay_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        frame_delta = 1  # TODO: it will be > 1 or 0 if frame skipping is enabled.

        detections = detector_compute(
            image=image,
            frame_delta=frame_delta,
            image_records=image_records,
            detection_records=detection_records,
            detection_frame_deltas=detection_frame_deltas
        )
        detection_records, detection_frame_deltas = store_record(
            copy.deepcopy(detections), frame_delta, detection_records, detection_frame_deltas
        )

        pedestrians, tracks = tracker_compute(
            detections=detections,
            detection_records=detection_records,
            detection_frame_deltas=detection_frame_deltas,
            image=image,
            image_records=image_records,
            frame_delta=frame_delta,
            pedestrian_records=pedestrian_records,
            pedestrian_frame_deltas=pedestrian_frame_deltas,
            previous_tracks=tracks,
        )
        pedestrian_records, pedestrian_frame_deltas = store_record(
            copy.deepcopy(pedestrians), frame_delta, pedestrian_records, pedestrian_frame_deltas
        )

        groups = clusterer_compute(
            pedestrians=pedestrians,
            pedestrian_records=pedestrian_records,
            pedestrian_frame_deltas=pedestrian_frame_deltas,
            tracks=tracks,
            image=image,
            image_records=image_records,
            frame_delta=frame_delta,
            group_records=group_records,
            group_frame_deltas=group_frame_deltas,
        )
        group_records, group_frame_deltas = store_record(
            copy.deepcopy(groups), frame_delta, group_records, group_frame_deltas
        )

        p_outside, p_inside, p_entering, p_leaving = apply_selection(
            pedestrians, tracks, selection_top_left, selection_bottom_right
        )
        overlay, overlay_mask = mark_selection(overlay, overlay_mask, selection_top_left, selection_bottom_right)
        overlay, overlay_mask = mark_pedestrians(overlay, overlay_mask, p_outside, p_inside, p_entering, p_leaving)
        overlay, overlay_mask = mark_tracks(overlay, overlay_mask, tracks)
        overlay, overlay_mask = mark_groups(overlay, overlay_mask, groups, pedestrians)
        image = merge_overlay(image, overlay, overlay_mask)

        cv.imshow(window_name, image)
        cv.waitKey(1)
        frame_end_time = time.perf_counter()
        if frame_end_time - frame_start_time > 0:
            time.sleep(max(0.0, frame_duration - frame_end_time + frame_start_time))
    cv.destroyAllWindows()
