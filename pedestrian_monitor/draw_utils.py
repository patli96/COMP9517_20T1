from typing import List, Dict, Tuple
import math

import numpy as np
import cv2 as cv

# Colors (B, G, R)
# Selection
color_selection = (255, 255, 255)  # White
color_outside = (90, 240, 140)  # Green
color_inside = (130, 105, 255)  # Red
color_entering = (90, 220, 240)  # Yellow
color_leaving = (70, 160, 255)  # Orange
color_group = (255, 170, 150)  # Blue
color_track = (255, 150, 230)  # Purple
color_entering_group = (230, 200, 110)  # Light Blue
color_leaving_group = (185, 230, 90)  # Cyan


def _draw_text(text, pos, color, overlay, mask):
    label_pos = (max(18, min(overlay.shape[0] - 3, pos[0])), max(0, min(overlay.shape[1] - 30, pos[1])))
    overlay = cv.putText(
        overlay,
        text,
        (label_pos[1], label_pos[0] - 3),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=1.0,
        color=color,
        thickness=1,
        lineType=cv.LINE_AA,
        bottomLeftOrigin=False,
    )
    mask = cv.putText(
        mask,
        text,
        (label_pos[1], label_pos[0] - 3),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=1.0,
        color=255,
        thickness=1,
        lineType=cv.LINE_AA,
        bottomLeftOrigin=False,
    )
    return overlay, mask


def _draw_box(text, box, color, overlay, mask):
    overlay, mask = _draw_text(text, (box[0], box[1]), color, overlay, mask)
    overlay = cv.rectangle(
        overlay,
        (box[1], box[0]),
        (box[3], box[2]),
        color=color,
        thickness=1,
        lineType=cv.LINE_AA,
        shift=0,
    )
    mask = cv.rectangle(
        mask,
        (box[1], box[0]),
        (box[3], box[2]),
        color=255,
        thickness=1,
        lineType=cv.LINE_AA,
        shift=0,
    )
    return overlay, mask


def loading():
    loading_image = np.zeros((64, 200), np.uint8)
    loading_image = cv.putText(
        loading_image,
        'Loading, please wait...',
        (5, loading_image.shape[0] - 5 - 21),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=1.0,
        color=255,
        thickness=1,
        lineType=cv.LINE_AA,
        bottomLeftOrigin=False,
    )
    return loading_image


def mark_selection(
        overlay: np.ndarray,
        mask: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        overlay_mode:int,
        total_inside: int,
        total_leaving: int,
        total_entering: int,
):
    if top_left == bottom_right:
        return overlay, mask
    if overlay_mode == 2:
        title = 'Selection (inside: '\
                + str(total_inside)\
                + ', leaving: '\
                + str(total_leaving)\
                + ', entering: '\
                + str(total_entering)\
                + ')'
    else:
        title = 'Selection'
    overlay, mask = _draw_box(
        title,
        (top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
        color_selection,
        overlay,
        mask
    )
    return overlay, mask


def mark_pedestrians(
        overlay: np.ndarray,
        mask: np.ndarray,
        p_outside: Dict[int, Tuple[int, int, int, int]],
        p_inside: Dict[int, Tuple[int, int, int, int]],
        p_entering: Dict[int, Tuple[int, int, int, int]],
        p_leaving: Dict[int, Tuple[int, int, int, int]],
):
    def draw_batch(pedestrians, color):
        nonlocal overlay, mask
        boxes = []
        for (p_id, box) in pedestrians.items():
            overlay, mask = _draw_text('P_' + str(p_id), (box[0], box[1]), color, overlay, mask)
            boxes.append(((box[1], box[0]), (box[3], box[0]), (box[3], box[2]), (box[1], box[2])))
        boxes = np.array(boxes)
        overlay = cv.polylines(
            overlay,
            boxes,
            True,
            color=color,
            thickness=1,
            lineType=cv.LINE_AA,
            shift=0,
        )
        mask = cv.polylines(
            mask,
            boxes,
            True,
            color=255,
            thickness=1,
            lineType=cv.LINE_AA,
            shift=0,
        )
    draw_batch(p_outside, color_outside)
    draw_batch(p_entering, color_entering)
    draw_batch(p_leaving, color_leaving)
    draw_batch(p_inside, color_inside)
    return overlay, mask


def mark_tracks(
        overlay: np.ndarray,
        mask: np.ndarray,
        tracks: Dict[int, List[Tuple[int, int]]],
        pedestrians: Dict[int, Tuple[int, int, int, int]],
):
    for (p_id, points) in tracks.items():
        if pedestrians.get(p_id, None) is None:
            continue
        if len(points) > 1:
            points = np.fliplr(np.array(points, dtype=np.int32))
            overlay = cv.polylines(
                overlay,
                [points],
                False,
                color=color_track,
                thickness=1,
                lineType=cv.LINE_AA,
                shift=0,
            )
            mask = cv.polylines(
                mask,
                [points],
                False,
                color=255,
                thickness=1,
                lineType=cv.LINE_AA,
                shift=0,
            )
    return overlay, mask


def mark_groups(
        overlay: np.ndarray,
        mask: np.ndarray,
        groups: Dict[int, List[int]],
        entering_members: Dict[int, List[int]],
        leaving_members: Dict[int, List[int]],
        groups_record: List[Dict[int, List[int]]],
        pedestrians: Dict[int, Tuple[int, int, int, int]],
        paused: bool,
):
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    prev_groups = groups_record[1] if len(groups_record) >= 2 else {}
    prev_group_ids = set(list(prev_groups.keys()))
    group_ids = set(list(groups.keys()))

    if not paused:
        # Print New & Dismissed Groups
        new_group_ids = list(group_ids - prev_group_ids)
        dismissed_group_ids = list(prev_group_ids - group_ids)
        new_group_prefix = 'New Group IDs:'
        new_group_prefix_empty = ' ' * len(new_group_prefix)
        dismissed_group_prefix = 'Dismissed Group IDs:'
        dismissed_group_prefix_empty = ' ' * len(dismissed_group_prefix)
        if len(group_ids.union(prev_group_ids)) > 0:
            number_length = len(str(max(999999, max(group_ids.union(prev_group_ids)))))
        else:
            number_length = 6
        max_numbers_per_line = (80 - max(len(dismissed_group_prefix), len(new_group_prefix))) \
            // (number_length + 2)
        if len(new_group_ids) > 0:
            new_group_id_chunks = list(chunks(new_group_ids, max_numbers_per_line))
            print(
                new_group_prefix
                + ', '.join(list(str(x).rjust(number_length) for x in new_group_id_chunks[0]))
            )
            for i in range(1, len(new_group_id_chunks)):
                print(
                    new_group_prefix_empty
                    + ', '.join(list(str(x).rjust(number_length) for x in new_group_id_chunks[i]))
                )
        if len(dismissed_group_ids) > 0:
            dismissed_group_id_chunks = list(chunks(dismissed_group_ids, max_numbers_per_line))
            print(
                dismissed_group_prefix
                + ', '.join(list(str(x).rjust(number_length) for x in dismissed_group_id_chunks[0]))
            )
            for i in range(1, len(dismissed_group_id_chunks)):
                print(
                    dismissed_group_prefix_empty
                    + ', '.join(list(str(x).rjust(number_length) for x in dismissed_group_id_chunks[i]))
                )

    # Draw leaving & entering pedestrians
    for p_ids in entering_members.values():
        for p_id in p_ids:
            box = pedestrians.get(p_id, None)
            if box is not None:
                overlay, mask = _draw_box('P_' + str(p_id), box, color_entering_group, overlay, mask)
    for p_ids in leaving_members.values():
        for p_id in p_ids:
            box = pedestrians.get(p_id, None)
            if box is not None:
                box = list(box)
                # Offset a bit so that entering and leaving will not overlap
                box[0] = max(2, box[0] - 2)
                box[1] = max(2, box[1] - 2)
                box[2] = min(overlay.shape[0], box[2] + 2)
                box[3] = min(overlay.shape[1], box[3] + 2)
                overlay, mask = _draw_box('P_' + str(p_id), box, color_leaving_group, overlay, mask)

    # Draw Group boxes
    for (g_id, p_ids) in groups.items():
        group_box = [np.inf, np.inf, -np.inf, -np.inf]
        is_valid = False
        for p_id in p_ids:
            box = pedestrians.get(p_id, None)
            if box is not None:
                is_valid = True
                group_box[0] = min(group_box[0], box[0])
                group_box[1] = min(group_box[1], box[1])
                group_box[2] = max(group_box[2], box[2])
                group_box[3] = max(group_box[3], box[3])
        if is_valid:
            # Offset a bit so that the bounding box of groups will not overlap pedestrian boxes
            group_box[0] = max(2, group_box[0] - 4)
            group_box[1] = max(2, group_box[1] - 4)
            group_box[2] = min(overlay.shape[0], group_box[2] + 4)
            group_box[3] = min(overlay.shape[1], group_box[3] + 4)
            overlay, mask = _draw_box('Group_' + str(g_id), group_box, color_group, overlay, mask)

    return overlay, mask


def get_status_text(
        total: int = 0,
        total_inside: int = 0,
        total_entering: int = 0,
        total_leaving: int = 0,
        total_groups: int = 0,
        overlay_mode: int = 0,
):
    return 'Total: %d    Inside: %d (%d entering, %d leaving)    Groups: %d' \
           % (int(total), int(total_inside), int(total_entering), int(total_leaving), int(total_groups))


def append_image_status_text(image:np.ndarray, status_text:str):
    status_image_shape = list(image.shape)
    status_image_shape[0] = 20
    status_image_shape = tuple(status_image_shape)
    status_image = np.full(status_image_shape, 255, dtype=np.uint8)
    cv.putText(
        status_image,
        status_text,
        (2, status_image_shape[0] - 5),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=1.0,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv.LINE_AA,
        bottomLeftOrigin=False,
    )
    image_appended = np.concatenate((image, status_image), axis=0)
    return image_appended


def resize_image(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        preserve_width: bool,
        preserve_height: bool,
):
    return image


def merge_overlay(image: np.ndarray, image_index: int, overlay_image: np.ndarray, overlay_mask: np.ndarray, overlay_mode: int):
    image_combined = image.copy()

    if overlay_mode > 0:
        overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
        overlay_mask = overlay_mask.astype(np.float32) / 255
        overlay_image = overlay_image.astype(np.float32)
        image_combined = image_combined.astype(np.float32)
        overlay_image *= overlay_mask
        image_combined *= 1 - overlay_mask
        image_combined += overlay_image
        image_combined = image_combined.astype(np.uint8)

    if overlay_mode == 0:
        image_combined = append_image_status_text(
            image_combined,
            '[Frame: ' + str(image_index) + '][OverlayMode 0] All overlays are hidden.',
        )
    elif overlay_mode == 1:
        image_combined = append_image_status_text(
            image_combined,
            '[Frame: ' + str(image_index) + '][OverlayMode 1] Show overlays for Task 1.',
        )
    elif overlay_mode == 2:
        image_combined = append_image_status_text(
            image_combined,
            '[Frame: ' + str(image_index) + '][OverlayMode 2] Show overlays for Task 2.',
        )
    elif overlay_mode == 3:
        image_combined = append_image_status_text(
            image_combined,
            '[Frame: ' + str(image_index) + '][OverlayMode 3] Show overlays for Task 3.',
        )
    return image_combined


T2_MAX_BOUNDARY = 45
T2_MIN_BOUNDARY = 20
T2_BOUNDARY_RATE = 0.15
T2_MAX_POINTS_CONSIDERED = 10
T2_POINT_WEIGHT_REDUCTION_RATE = 0.7
T2_TELEPORT_MIN_DISTANCE = 20


def apply_selection(
        pedestrians: Dict[int, Tuple[int, int, int, int]],
        pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],
        pedestrian_frame_deltas: List[int],
        tracks: Dict[int, List[Tuple[int, int]]],
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        image_height: int,
        image_width: int,
):
    if top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1]:
        # We don't handle any zero-size selections
        return pedestrians, dict(), dict(), dict()

    # Get base properties
    top, left = top_left
    bottom, right = bottom_right
    height = bottom - top
    width = right - left

    # Boundaries
    # The outer boundary is height_or_width * BOUNDARY_RATE, but limited in [MIN_BOUNDARY, MAX_BOUNDARY]
    y_outer = max(T2_MIN_BOUNDARY, min(T2_MAX_BOUNDARY, round(height * T2_BOUNDARY_RATE)))
    x_outer = max(T2_MIN_BOUNDARY, min(T2_MAX_BOUNDARY, round(width * T2_BOUNDARY_RATE)))
    # The inner should be no more than
    y_inner = min(int((bottom - top) / 2), y_outer)
    x_inner = min(int((right - left) / 2), x_outer)

    # Inner & Outer bounding box
    top_inner = min(bottom, top + y_inner)
    top_outer = max(0, top - y_outer)
    bottom_inner = max(top, bottom - y_inner)
    bottom_outer = min(image_height, bottom + y_outer)
    left_inner = min(right, left + x_inner)
    left_outer = max(0, left - x_outer)
    right_inner = max(left, right - x_inner)
    right_outer = min(image_width, right + x_outer)

    # Init
    p_outside = {}
    p_inside = {}
    p_entering = {}
    p_leaving = {}

    for (p_id, p_box) in pedestrians.items():
        p_points = tracks.get(p_id, None)
        if p_points is not None and len(p_points) > 0:
            p_center = p_points[0]
            p_points = []
        else:
            p_center = None
        if p_center is None:
            p_center = (round((p_box[0] + p_box[2]) / 2), round((p_box[1] + p_box[3]) / 2))

        # Check inside inner / outside outer / on boundary
        if bottom_inner > p_center[0] > top_inner and right_inner > p_center[1] > left_inner:
            p_inside[p_id] = p_box
            continue
        elif p_center[0] > bottom_outer\
                or p_center[0] < top_outer\
                or p_center[1] > right_outer\
                or p_center[1] < left_outer:
            p_outside[p_id] = p_box
            continue

        # Check if is inside
        is_inside = False
        if bottom >= p_center[0] >= top and right >= p_center[1] >= left:
            is_inside = True

        # Get the mean walking direction
        total_weights = 1.0
        point_weight = 1.0
        direction = (0.0, 0.0)
        for i in range(1, T2_MAX_POINTS_CONSIDERED + 1):
            if len(p_points) > i:
                p_ref_center = p_points[i]
            elif len(pedestrian_records) > i:
                p_ref_box = pedestrian_records[i].get(p_id, None)
                if p_ref_box is None:
                    break
                p_ref_center = (round((p_ref_box[0] + p_ref_box[2]) / 2), round((p_ref_box[1] + p_ref_box[3]) / 2))
            else:
                break
            if len(pedestrian_frame_deltas) > i:
                p_ref_frame_delta = pedestrian_frame_deltas[i]
            else:
                p_ref_frame_delta = 1
            p_vec = (float(p_center[0] - p_ref_center[0]), float(p_center[1] - p_ref_center[1]))
            if math.hypot(p_vec[0], p_vec[1]) > T2_TELEPORT_MIN_DISTANCE * p_ref_frame_delta:
                # If the pedestrian "teleport"
                break
            if p_vec != (0.0, 0.0):
                point_weight *= T2_POINT_WEIGHT_REDUCTION_RATE
                total_weights += point_weight
                norm = np.linalg.norm(p_vec, ord=2)
                p_vec = (p_vec[0] / norm * point_weight, p_vec[1] / norm * point_weight)
                direction = (direction[0] + p_vec[0], direction[1] + p_vec[1])
        direction = (direction[0] / total_weights, direction[1] / total_weights)

        # If is moving
        if not math.isclose(math.hypot(direction[0], direction[1]), 0.0, rel_tol=0.1, abs_tol=0.1):
            # Normalize direction
            norm = np.linalg.norm(direction, ord=2)
            direction = (direction[0] / norm, direction[1] / norm)

            # If inside, leaving: leaving
            # If inside, not leaving: inside
            # If outside, leaving: outside
            # If outside, not leaving, intersects: entering
            # If outside, not leaving, not intersects: outside

            # Check if the direction is leaving the inner box
            is_leaving = False
            if p_center[0] < top_inner and direction[0] < 0:
                is_leaving = True
            elif p_center[0] > bottom_inner and direction[0] > 0:
                is_leaving = True
            elif p_center[1] < left_inner and direction[1] < 0:
                is_leaving = True
            elif p_center[1] > right_inner and direction[1] > 0:
                is_leaving = True

            if is_inside and is_leaving:
                p_leaving[p_id] = p_box
                continue
            elif is_inside and not is_leaving:
                p_inside[p_id] = p_box
                continue
            elif not is_inside and is_leaving:
                p_outside[p_id] = p_box
                continue

            # Check the line intersects the exact box
            intersects = False
            # Formula of the line: Ax + By + C = 0, x is [0], y is [1]
            # Line: (x1, y1) to (x2, y2)
            x1 = p_center[0]
            y1 = p_center[1]
            x2 = x1 + direction[0]
            y2 = y1 + direction[1]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2

            total_positive = 0
            for box_point in [(top, left), (top, right), (bottom, left), (bottom, right)]:
                d = a * box_point[0] + b * box_point[1] + c
                if math.isclose(d, 0.0, rel_tol=0.0001, abs_tol=0.0001):
                    intersects = True
                    break
                elif d > -0.0001:
                    total_positive += 1
            if 4 > total_positive > 0:
                intersects = True

            if not is_inside and not is_leaving and intersects:
                p_entering[p_id] = p_box
                continue
            elif not is_inside and not is_leaving and not intersects:
                p_outside[p_id] = p_box
        else:
            if is_inside:
                p_inside[p_id] = p_box
            else:
                p_outside[p_id] = p_box

    return p_outside, p_inside, p_entering, p_leaving
