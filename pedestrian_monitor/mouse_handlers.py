import time

import cv2 as cv


def mouse_callback_factory(
        on_lmb_pressed=None,
        on_lmb_released=None,
        on_lmb_clicked=None,
        on_lmb_dragging=None,
        on_rmb_pressed=None,
        on_rmb_released=None,
        on_rmb_clicked=None,
        on_rmb_dragging=None,
        on_mmb_pressed=None,
        on_mmb_released=None,
        on_mmb_clicked=None,
        on_mmb_dragging=None,
        min_dragging_time=0.06,
):
    lmb_pressed = False
    lmb_dragging = False
    lmb_pressed_time = 0
    rmb_pressed = False
    rmb_dragging = False
    rmb_pressed_time = 0
    mmb_pressed = False
    mmb_dragging = False
    mmb_pressed_time = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal lmb_pressed, rmb_pressed, mmb_pressed
        nonlocal lmb_dragging, rmb_dragging, mmb_dragging
        nonlocal lmb_pressed_time, rmb_pressed_time, mmb_pressed_time
        if event == cv.EVENT_LBUTTONDOWN:
            lmb_pressed = True
            lmb_pressed_time = time.perf_counter()
            if callable(on_lmb_pressed):
                on_lmb_pressed(x, y)
        elif event == cv.EVENT_LBUTTONUP:
            lmb_pressed = False
            if time.perf_counter() - lmb_pressed_time <= min_dragging_time or not lmb_dragging:
                if callable(on_lmb_clicked):
                    on_lmb_clicked(x, y)
            lmb_dragging = False
            if callable(on_lmb_released):
                on_lmb_released(x, y)
        elif event == cv.EVENT_RBUTTONDOWN:
            rmb_pressed = True
            rmb_pressed_time = time.perf_counter()
            if callable(on_rmb_pressed):
                on_rmb_pressed(x, y)
        elif event == cv.EVENT_RBUTTONUP:
            rmb_pressed = False
            if time.perf_counter() - rmb_pressed_time <= min_dragging_time or not rmb_dragging:
                if callable(on_rmb_clicked):
                    on_rmb_clicked(x, y)
            rmb_dragging = False
            if callable(on_rmb_released):
                on_rmb_released(x, y)
        elif event == cv.EVENT_MBUTTONDOWN:
            mmb_pressed = True
            mmb_pressed_time = time.perf_counter()
            if callable(on_mmb_pressed):
                on_mmb_pressed(x, y)
        elif event == cv.EVENT_MBUTTONUP:
            mmb_pressed = False
            if time.perf_counter() - mmb_pressed_time <= min_dragging_time or not mmb_dragging:
                if callable(on_mmb_clicked):
                    on_mmb_clicked(x, y)
            mmb_dragging = False
            if callable(on_mmb_released):
                on_mmb_released(x, y)
        elif event == cv.EVENT_MOUSEMOVE:
            current_time = time.perf_counter()
            if lmb_pressed and callable(on_lmb_dragging) and current_time - lmb_pressed_time > min_dragging_time:
                lmb_dragging = True
                on_lmb_dragging(x, y)
            if rmb_pressed and callable(on_rmb_dragging) and current_time - rmb_pressed_time > min_dragging_time:
                rmb_dragging = True
                on_rmb_dragging(x, y)
            if mmb_pressed and callable(on_mmb_dragging) and current_time - mmb_pressed_time > min_dragging_time:
                mmb_dragging = True
                on_mmb_dragging(x, y)

    return mouse_callback
