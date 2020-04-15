"""COMP9517 20T1 Project - Pedestrian Detecting, Tracking and Clustering"""
import copy
import sys
import time
import importlib
from typing import List, Optional

import cv2 as cv
import numpy as np
from PyQt5 import QtCore

from .console_arguments import get_args
from .draw_utils import get_status_text, apply_selection
from .draw_utils import resize_image, mark_selection, mark_pedestrians, mark_tracks, mark_groups, merge_overlay
from .file_handlers import get_image_paths, ImageFileIterator
from .mouse_handlers import mouse_callback_factory

__version__ = '0.2.0'


def main():
    args = get_args()

    from .preprocessors.sample_preprocessor import compute as preprocessor_compute
    from .detectors.sample_detector import compute as detector_compute
    from .trackers.sample_tracker import compute as tracker_compute
    from .clusterers.sample_clusterer import compute as clusterer_compute

    if args.preprocessor != 'sample_preprocessor':
        preprocessor = importlib.import_module('.preprocessors.' + args.preprocessor, package=__name__)
        preprocessor_compute = preprocessor.compute
    if args.detector != 'sample_detector':
        detector = importlib.import_module('.detectors.' + args.detector, package=__name__)
        detector_compute = detector.compute
    if args.tracker != 'sample_tracker':
        tracker = importlib.import_module('.trackers.' + args.tracker, package=__name__)
        tracker_compute = tracker.compute
    if args.clusterer != 'sample_clusterer':
        clusterer = importlib.import_module('.clusterers.' + args.clusterer, package=__name__)
        clusterer_compute = clusterer.compute

    if args.path_video == '':
        iterator = ImageFileIterator(args.path, False)
    else:
        # TODO: VideoCapture iterator unfinished
        iterator = ImageFileIterator(args.path, False)

    width = int(args.width)
    height = int(args.height)
    preserve_width = args.width == -1
    preserve_height = args.height == -1

    print('')
    print('How to control:')
    print('[LeftMouseButton] Pause/Resume')
    paused = False

    def toggle_pause(x, y):
        nonlocal paused
        paused = not paused

    print('[RightMouseButton] Draw bounding box')
    selection_point1 = (0, 0)
    selection_point2 = (0, 0)
    selection_top_left = (0, 0)
    selection_bottom_right = (0, 0)
    finish_selection = True

    def init_selection(x, y):
        nonlocal selection_top_left, selection_bottom_right, finish_selection
        nonlocal selection_point1, selection_point2
        selection_point1 = (y, x)
        selection_top_left = selection_point1
        if finish_selection:
            selection_point2 = selection_point1
            selection_bottom_right = selection_top_left
            finish_selection = False

    def drag_selection(x, y):
        nonlocal selection_top_left, selection_bottom_right
        nonlocal selection_point1, selection_point2
        selection_point2 = (max(2, min(height - 3, y)), max(2, min(width - 3, x)))
        selection_top_left = (
            min(selection_point1[0], selection_point2[0]),
            min(selection_point1[1], selection_point2[1])
        )
        selection_bottom_right = (
            max(selection_point1[0], selection_point2[0]),
            max(selection_point1[1], selection_point2[1])
        )

    def cancel_selection(x, y):
        nonlocal selection_top_left, selection_bottom_right, finish_selection
        nonlocal selection_point1, selection_point2
        selection_top_left = (0, 0)
        selection_bottom_right = (0, 0)
        selection_point1 = (0, 0)
        selection_point2 = (0, 0)
        finish_selection = True

    def done_selection(x, y):
        nonlocal selection_top_left, selection_bottom_right, finish_selection
        nonlocal selection_point1, selection_point2
        selection_top_left = (
            min(selection_point1[0], selection_point2[0]),
            min(selection_point1[1], selection_point2[1])
        )
        selection_bottom_right = (
            max(selection_point1[0], selection_point2[0]),
            max(selection_point1[1], selection_point2[1])
        )
        finish_selection = True

    print('[MiddleMouseButton] Switch between None, Task 1, Task 2 and Task 3')
    overlay_mode = 1

    def switch_overlay(x, y):
        nonlocal overlay_mode
        overlay_mode = (overlay_mode + 1) % 4

    # Fix High DPI Scaling Issues, these should be called before creating any windows
    # These are called because OpenCV uses Qt, and doesn't export any Qt settings
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    window_name = 'Pedestrian Monitor'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
    cv.setMouseCallback(window_name, mouse_callback_factory(
        on_lmb_clicked=toggle_pause,
        on_rmb_clicked=cancel_selection,
        on_rmb_pressed=init_selection,
        on_rmb_dragging=drag_selection,
        on_rmb_released=done_selection,
        on_mmb_clicked=switch_overlay,
    ))

    if args.fps <= 0:
        print('Invalid FPS: FPS has to be positive.', file=sys.stderr)
        sys.exit(1)
    frame_duration = 1 / float(args.fps)
    frame_skipping = args.frame_skipping
    first_frame = True
    image = np.zeros((576, 768, 3), np.uint8)
    image_path = ''
    image_index = 0

    max_records = 20  # Max number of records

    image_records = []

    feature_records = []
    feature_frame_deltas = []
    preprocessor_storage = {}

    detection_records = []
    detection_frame_deltas = []
    detector_storage = {}

    pedestrian_records = []
    pedestrian_frame_deltas = []
    tracks = {}
    tracker_storage = {}

    group_records = []
    group_frame_deltas = []
    clusterer_storage = {}

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
            image, image_path, image_index = next(iterator, (image, image_path, image_index))
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

        features = preprocessor_compute(
            image=image,
            image_index=image_index,
            frame_delta=frame_delta,
            image_records=image_records,
            previous_feature_records=feature_records,
            previous_feature_frame_deltas=feature_frame_deltas,
            storage=preprocessor_storage,
        )
        feature_records, feature_frame_deltas = store_record(
            copy.deepcopy(features), frame_delta, feature_records, feature_frame_deltas
        )

        if not args.no_detector:
            detections = detector_compute(
                features=features,
                feature_records=feature_records,
                feature_frame_deltas=feature_frame_deltas,
                image=image,
                image_index=image_index,
                image_records=image_records,
                frame_delta=frame_delta,
                previous_detection_records=detection_records,
                previous_detection_frame_deltas=detection_frame_deltas,
                storage=detector_storage,
            )
        else:
            detections = {}
        detection_records, detection_frame_deltas = store_record(
            copy.deepcopy(detections), frame_delta, detection_records, detection_frame_deltas
        )

        if not args.no_tracker:
            pedestrians, tracks = tracker_compute(
                detections=detections,
                detection_records=detection_records,
                detection_frame_deltas=detection_frame_deltas,
                image=image,
                image_index=image_index,
                image_records=image_records,
                frame_delta=frame_delta,
                previous_pedestrian_records=pedestrian_records,
                previous_pedestrian_frame_deltas=pedestrian_frame_deltas,
                previous_tracks=tracks,
                storage=tracker_storage,
            )
        else:
            pedestrians = {}
            tracks = {}
        pedestrian_records, pedestrian_frame_deltas = store_record(
            copy.deepcopy(pedestrians), frame_delta, pedestrian_records, pedestrian_frame_deltas
        )

        if not args.no_clusterer:
            groups, entering_members, leaving_members = clusterer_compute(
                pedestrians=pedestrians,
                pedestrian_records=pedestrian_records,
                pedestrian_frame_deltas=pedestrian_frame_deltas,
                tracks=tracks,
                image=image,
                image_index=image_index,
                image_records=image_records,
                frame_delta=frame_delta,
                previous_group_records=group_records,
                previous_group_frame_deltas=group_frame_deltas,
                storage=clusterer_storage,
            )
        else:
            groups = {}
            entering_members = {}
            leaving_members = {}
        group_records, group_frame_deltas = store_record(
            copy.deepcopy(groups), frame_delta, group_records, group_frame_deltas
        )

        if overlay_mode == 2:
            p_outside, p_inside, p_entering, p_leaving = apply_selection(
                pedestrians, tracks, selection_top_left, selection_bottom_right
            )
        else:
            p_outside = pedestrians
            p_inside = {}
            p_entering = {}
            p_leaving = {}
        overlay, overlay_mask = mark_selection(overlay, overlay_mask, selection_top_left, selection_bottom_right)
        overlay, overlay_mask = mark_pedestrians(overlay, overlay_mask, p_outside, p_inside, p_entering, p_leaving)
        overlay, overlay_mask = mark_tracks(overlay, overlay_mask, tracks)

        if overlay_mode == 3:
            overlay, overlay_mask = mark_groups(
                overlay=overlay,
                mask=overlay_mask,
                groups=groups,
                entering_members=entering_members,
                leaving_members=leaving_members,
                groups_record=group_records,
                pedestrians=pedestrians,
                paused=paused or image_index <= 0,
            )

        image_merged = merge_overlay(image, overlay, overlay_mask, overlay_mode)

        cv.imshow(window_name, image_merged)
        keyboard_key = cv.waitKey(1) & 0xFF
        if keyboard_key == 27:  # [Esc]
            break
        elif keyboard_key == 32:  # [Space]
            toggle_pause(0, 0)
        elif keyboard_key == 48:  # [0]
            overlay_mode = 0
        elif keyboard_key == 49:  # [1]
            overlay_mode = 1
        elif keyboard_key == 50:  # [2]
            overlay_mode = 2
        elif keyboard_key == 51:  # [3]
            overlay_mode = 3
        frame_end_time = time.perf_counter()
        if frame_end_time - frame_start_time > 0:
            time.sleep(max(0.0, frame_duration - frame_end_time + frame_start_time))
    cv.destroyAllWindows()
