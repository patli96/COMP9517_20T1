from typing import List, Dict, Tuple

import numpy as np
import cv2 as cv

# Colors (B, G, R)
# Selection
color_selection = (255, 255, 255)  # White
color_outside = (90, 240, 140)  # Green
color_inside = (120, 120, 255)  # Red
color_entering = (90, 210, 240)  # Yellow
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
):
    if top_left == bottom_right:
        return overlay, mask
    overlay, mask = _draw_box(
        'Selection',
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
    for (p_id, box) in p_outside.items():
        overlay, mask = _draw_box('P_' + str(p_id), box, color_outside, overlay, mask)
    for (p_id, box) in p_inside.items():
        overlay, mask = _draw_box('P_' + str(p_id), box, color_inside, overlay, mask)
    for (p_id, box) in p_entering.items():
        overlay, mask = _draw_box('P_' + str(p_id), box, color_entering, overlay, mask)
    for (p_id, box) in p_leaving.items():
        overlay, mask = _draw_box('P_' + str(p_id), box, color_leaving, overlay, mask)
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
            for point_i in range(1, len(points)):
                overlay = cv.line(
                    overlay,
                    pt1=(points[point_i - 1][1], points[point_i - 1][0]),
                    pt2=(points[point_i][1], points[point_i][0]),
                    color=color_track,
                    thickness=1,
                    lineType=cv.LINE_AA,
                    shift=0,
                )
                mask = cv.line(
                    mask,
                    pt1=(points[point_i - 1][1], points[point_i - 1][0]),
                    pt2=(points[point_i][1], points[point_i][0]),
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
    image_appended = image.copy()
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
    image_appended = np.concatenate((image_appended, status_image), axis=0)
    return image_appended


def resize_image(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        preserve_width: bool,
        preserve_height: bool,
):
    return image


def merge_overlay(image: np.ndarray, overlay_image: np.ndarray, overlay_mask: np.ndarray, overlay_mode: int):
    image_combined = image.copy()

    if overlay_mode > 0:
        overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
        overlay_mask = overlay_mask.astype(float) / 255
        overlay_image = overlay_image.astype(float)
        image_combined = image_combined.astype(float)
        overlay_image *= overlay_mask
        image_combined *= 1 - overlay_mask
        image_combined += overlay_image
        image_combined = np.rint(image_combined).astype(np.uint8)

    if overlay_mode == 0:
        image_combined = append_image_status_text(
            image_combined,
            '[OverlayMode 0] All Overlays are hidden.',
        )
    if overlay_mode == 1:
        image_combined = append_image_status_text(
            image_combined,
            '[OverlayMode 1] Show overlays for Task 1.',
        )
    if overlay_mode == 2:
        image_combined = append_image_status_text(
            image_combined,
            '[OverlayMode 2] Show overlays for Task 2. (Unfinished)',
        )
    if overlay_mode == 3:
        image_combined = append_image_status_text(
            image_combined,
            '[OverlayMode 3] Show overlays for Task 3.',
        )
    return image_combined


def apply_selection(
        pedestrians: Dict[int, Tuple[int, int, int, int]],
        tracks: Dict[int, List[Tuple[int, int]]],
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
):
    p_outside = pedestrians
    p_inside = pedestrians
    p_entering = pedestrians
    p_leaving = pedestrians
    return p_outside, p_inside, p_entering, p_leaving
