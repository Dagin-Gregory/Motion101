import time
import pyautogui
#import keyboard
from pynput import mouse

def get_xy():
    pos = pyautogui.position()
    x_abs = pos.x
    y_abs = pos.y
    return (y_abs, x_abs)

def mouse_click(x, y, button, pressed):
    # Assume it's a left mouse click
    if (pressed):
        return get_xy()

    else:
        return (-1,-1)

def record_actions(file_name):
    listener = mouse.Listener(on_click=mouse_click)
    listener.start()
    while(True):
        print(listener)
        print("a")
        #if (pyautogui.keyboard.)
def main():
    record_actions("")

if __name__ == '__main__':
    main()