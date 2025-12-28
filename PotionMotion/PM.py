import os, sys
import cv2
import win32gui as pywgui
import numpy as np
import time

import threading
import multiprocessing
from multiprocessing.sharedctypes import Synchronized
import ctypes

import Helpers.HelperControl as helper_control
import Helpers.HelperKeyboard as helper_keyboard
import Helpers.HelperScreen as helper_screen
import Helpers.HelperPotionMotion as helper_pm

DEBUG = False

def potionMotion():
    currPid = os.getpid()
    #game_window_fn = 'gameWindow'
    #mousemovement_window_fn = 'mouseMovement'
    #manage_game_fn = 'manageGame'
    #stored as [color_map index (row_0), ... (row 0), color_map index (row 0), color_map index (row 1),]

    wizardHwnd = helper_screen.getWizardWindow()

    with helper_control.ControlStruct() as keybd_ctrl:
        keyb_paused_children = multiprocessing.Value(ctypes.c_bool, False)
        keyb_thread = threading.Thread(name='keyboard', target=helper_control.ControlStructWrapper,
                                       args=[keybd_ctrl, helper_control.keyb_control_struct_manager,
                                        [keybd_ctrl, keyb_paused_children]])
        keyb_thread.start()

        with helper_control.ControlStruct(parent=keybd_ctrl) as manager_control:
            retries = 0
            time_between_checks = 3
            manager_confidence = .8
            with (helper_screen.ScreenCapture(pywgui.GetDesktopWindow()) as manager_capture,
                  helper_screen.ScreenCapture(wizardHwnd) as game_capture):
                banner_template = helper_screen.openImage('PotionMotion/PM_Banner.png')
                continue_template = helper_screen.openImage('PotionMotion/PM_Continue.png')
                play_template = helper_screen.openImage('PotionMotion/PM_Play.png')
                manager_thread = threading.Thread(name='manager', target=helper_control.ControlStructWrapper,
                                                args=[keybd_ctrl, helper_pm.manage_game, 
                                                      [manager_control, manager_capture, 
                                                       banner_template, continue_template, play_template, 
                                                       retries, time_between_checks, manager_confidence]])
                manager_thread.start()

                with (helper_control.ControlStruct(parent=manager_control) as screenshot_control,
                      helper_control.ControlStruct(parent=manager_control) as mouse_control,
                      multiprocessing.Manager() as manager):
                    maxSize = 5
                    moves = multiprocessing.Queue(maxSize)
                    #queued_moves = {}
                    queued_moves = manager.dict()
                    #safe_screenshot = [True]
                    safe_screenshot:Synchronized[bool] = multiprocessing.Value(ctypes.c_bool, True)
                    curr_suffix:Synchronized[int] = multiprocessing.Value(ctypes.c_int, 0)
                    curr_retries:Synchronized[int] = multiprocessing.Value(ctypes.c_int, 0)
                    screenshot_proc = multiprocessing.Process(name='screenshot', target=helper_control.ControlStructWrapper, 
                                                            args=[screenshot_control, helper_pm.findMoves, 
                                                                  [moves, queued_moves, safe_screenshot, 
                                                                   curr_suffix, screenshot_control, game_capture, 
                                                                   curr_retries, 'PotionMotion/noSoln/']])
                    screenshot_proc.start()

                    mouse_thread = threading.Thread(name='mouse', target=helper_control.ControlStructWrapper, 
                                                    args=[mouse_control, helper_pm.consumeMoves, 
                                                          [game_capture, moves, queued_moves, safe_screenshot]])
                    mouse_thread.start()

                    screenshot_proc.join()
                    mouse_thread.join()
                manager_thread.join()
        keyb_thread.join()

def fps_test(width:int, height:int):
    #cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    prev = time.time()
    fps_ema = 0.0
    alpha = 0.1  # smoothing factor

    with helper_screen.ScreenCapture(pywgui.GetDesktopWindow(), "_.bmp") as sc:
        while True:
            # IMPORTANT: SaveBitmapFile writes BMP
            #bitmap_return = sc.saveBitmap(height=800, width=800)
            #img = cv2.cvtColor(np.array(bitmap_return), cv2.COLOR_RGB2BGR)
            bitmap_return = None
            bitmap_return = sc.saveBitmap(height=height, width=width)
            img = cv2.cvtColor(np.array(bitmap_return), cv2.COLOR_RGB2BGR)

            if img is None:
                # If read races the write, brief retry helps
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue

            cv2.imshow("Preview", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            now = time.time()
            dt = now - prev
            prev = now

            if dt > 0:
                fps_inst = 1.0 / dt
                fps_ema = fps_inst if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps_inst
                print(f"{fps_ema:.1f} FPS")

    cv2.destroyAllWindows()

def fps_test_controlled(width:int, height:int, sc:helper_screen.ScreenCapture, windowName:str):
    prev = time.time()
    fps_ema = 0.0
    alpha = 0.1  # smoothing factor

    bitmap_return = None
    bitmap_return = sc.saveBitmap(height=height, width=width)
    img = cv2.cvtColor(np.array(bitmap_return), cv2.COLOR_RGB2BGR)
    cv2.imshow(windowName, img)

    _ = cv2.waitKey(1)

    now = time.time()
    dt = now - prev
    prev = now

    if dt > 0:
        fps_inst = 1.0 / dt
        fps_ema = fps_inst if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps_inst
        print(f"{fps_ema:.1f} FPS")

def main():
    _=helper_keyboard.key_pressed('p')
    potionMotion()


# This function is special, needs information about all other ControlStructs so it has to exist outside of the helper
def test_kbd(control_struct:helper_control.ControlStruct):

    while True:
        #print(iter)
        if (helper_keyboard.key_pressed('q')):
            control_struct.stop()
            break

        if (helper_keyboard.key_pressed('p')):
            if (control_struct.paused.value):
                control_struct.unpause(unpause_all_offspring=True)
            else:
                control_struct.pause()

def worker1(arg):
    print(arg, ' | Worker 1')
    time.sleep(1)

if __name__ == '__main__':
    """
    sanity_check_fn = 'noSoln/NoSoln0.bmp'
    height, width = setupGameWindowHwnd(sanity_check_fn)
    rows_ = 6
    cols_ = 7
    sq_height = int(height / rows_)
    sq_width = int(width / cols_)
    saveGameWindow()
    processImage(img, )
    """

    """
    _=helper_keyboard.key_pressed('p')
    with helper_control.ControlStruct() as outer:
        with helper_control.ControlStruct(parent=outer) as inner:
            paused_children = multiprocessing.Value(ctypes.c_bool, False)
            outer_thread = multiprocessing.Process(name='kbd', target=helper_control.ControlStructWrapper, args=[outer, helper_control.keyb_control_struct_manager, [outer, paused_children]])
            outer_thread.start()

            with helper_screen.ScreenCapture(pywgui.GetDesktopWindow()) as sc:
                cv2_window_name = 'Preview'
                inner_thread = multiprocessing.Process(name='worker', target=helper_control.ControlStructWrapper, args=[inner, fps_test_controlled, [1920, 1080, sc, cv2_window_name]])
                inner_thread.start()
                inner_thread.join()
                cv2.destroyAllWindows()
            outer_thread.join()
    """

    """
    with helper_screen.ScreenCapture(helper_screen.getWizardWindow()) as screen_capture:
        adjHeight, adjWidth, _, screen_offset, topLeft = screen_capture.setupGameWindowPM()
        y_offset, x_offset = topLeft
        #game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=screen_sync, set_focus=set_focus)
        game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=[True], set_focus=True, save_to_file=True)
        if (game_window is str):
            game_window = helper_screen.openImage(game_window)
    """

    main()
