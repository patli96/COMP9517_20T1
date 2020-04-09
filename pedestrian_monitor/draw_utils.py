from typing import List, Dict, Tuple

import numpy as np

# Colors (B, G, R)
# Selection
color_selection = (255, 255, 255)  # White
color_outside = (90, 255, 150)  # Green
color_inside = (130, 130, 255)  # Red
color_entering = (100, 220, 255)  # Yellow
color_leaving = (60, 130, 255)  # Orange
color_group = (255, 190, 50)  # Blue
color_track = (180, 80, 160)  # Purple


def mark_selection(
        overlay: np.ndarray,
        mask: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
):
    return overlay, mask


def mark_pedestrians(
        overlay: np.ndarray,
        mask: np.ndarray,
        p_outside: Dict[int, Tuple[int, int, int, int]],
        p_inside: Dict[int, Tuple[int, int, int, int]],
        p_entering: Dict[int, Tuple[int, int, int, int]],
        p_leaving: Dict[int, Tuple[int, int, int, int]],
):
    return overlay, mask


def mark_tracks(
        overlay: np.ndarray,
        mask: np.ndarray,
        tracks: Dict[int, List[Tuple[int, int]]],
):
    return overlay, mask


def mark_groups(
        overlay: np.ndarray,
        mask: np.ndarray,
        groups: Dict[int, List[int]],
        pedestrians: Dict[int, Tuple[int, int, int, int]],
):
    return overlay, mask


def get_status_text(
        total: int = 0,
        total_inside: int = 0,
        total_entering: int = 0,
        total_leaving: int = 0,
        total_groups: int = 0
):
    return 'Total: %d    Inside: %d (%d entering, %d leaving)    Groups: %d' \
           % (int(total), int(total_inside), int(total_entering), int(total_leaving), int(total_groups))


def resize_image(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        preserve_width: bool,
        preserve_height: bool,
):
    return image


def merge_overlay(image: np.ndarray, overlay_image: np.ndarray, overlay_mask: np.ndarray):
    return image


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
