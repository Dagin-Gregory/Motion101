import pyautogui as pyat
import random
import time
from multiprocessing.sharedctypes import Synchronized

SIMULATE_PLAYER = False

def dragMouse(start_y, start_x, end_y, end_x, safe_screenshot:Synchronized, duration=.001):
    mouseButton = 'left'
    # A mouse drag is moving the mouse while holding down the left mouse button(LMB)
    with safe_screenshot.get_lock():
        safe_screenshot.value = False
        pyat.moveTo(start_x, start_y, duration, _pause=False)
        pyat.mouseDown(button=mouseButton)
        pyat.moveTo(end_x, end_y, duration, _pause=False)
        if (SIMULATE_PLAYER):
            random_sleep = random.random()*.1
            time.sleep(random_sleep)
        time.sleep(duration)
        pyat.mouseUp(button=mouseButton)
        safe_screenshot.value = True

def leftclick(y, x, duration=.1):
    mouseButton = 'left'
    pyat.moveTo(x, y, _pause=False)
    pyat.click(x, y, button=mouseButton)
    pyat.mouseDown(x, y, button=mouseButton)
    time.sleep(duration)
    pyat.mouseUp(button=mouseButton)
