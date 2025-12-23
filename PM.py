from PIL import Image
from math import *
import shutil
import random
import queue
import time
import os
import threading
import signal
import pyautogui as pyat
import win32gui as pyw
import win32ui as pywui
import win32api as pywapi
import win32process as pywproc
import win32con as pywcon
import numpy as np
import cv2
import numpy.typing as npt
import matplotlib.pyplot as plt
import multiprocessing
import enum

DEBUG = False
SIMULATE_PLAYER = False
SYNC_SCREEN = True


def openImage(img_name:str):
    image_ptr = Image.open(img_name)
    return image_ptr

def delta(color_a, color_b) :
    delta_E = sqrt( (color_a[0] - color_b[0]) ** 2 +
                    (color_a[1] - color_b[1]) ** 2 +
                    (color_a[2] - color_b[2]) ** 2)
    return delta_E

def lin_xy(x, y, shift=0, shift_y=0, rows_=6, cols_=7):
    linear_idx = ((y-shift_y)%rows_)*cols_ + ((x-shift)%cols_)
    return linear_idx

def xy(x, y, xy_map, shift=0, shift_y=0) -> int:
    linear_idx = lin_xy(x, y, shift, shift_y)
    value = xy_map[linear_idx]

    return value

def roll_row(xy_map, y, shift, cols_=7) -> None:
    new_row = []
    for x in range(cols_):
        new_row.append(xy(x, y, xy_map, shift))
    for x in range(cols_):
        xy_map[lin_xy(x, y)] = new_row[x]

def roll_col(xy_map, x, shift, rows_=6) -> None:
    new_col = []
    for y in range(rows_):
        new_col.append(xy(x, y, xy_map, shift_y=shift))
    for y in range(rows_):
        xy_map[lin_xy(x, y)] = new_col[y]

def process_square(image:Image.Image, x_off, y_off, sq_height, sq_width):
    r_avg = 0
    g_avg = 0
    b_avg = 0
    total_pixels = 0
    for square_w in range(0, sq_width):
        for square_l in range(0, sq_height):
            curr_pixel = image.getpixel((square_w + x_off, square_l + y_off))

            r_avg += curr_pixel[0]
            g_avg += curr_pixel[1]
            b_avg += curr_pixel[2]
            total_pixels += 1
    if (total_pixels == 0):
        return [0, 0, 0]
    r_avg /= total_pixels
    g_avg /= total_pixels
    b_avg /= total_pixels
    return [r_avg, g_avg, b_avg]

def processImage(image:Image.Image, sq_height, sq_width, rows_=6, cols_=7, similar_thresh=10) :
    color_map = []
    xy_map = []
    for curr_row in range(0, rows_) : 
        for curr_col in range(0, cols_) :
            y_offset = curr_row * sq_height
            x_offset = curr_col * sq_width
            avg_colors = process_square(image, x_offset, y_offset, sq_height, sq_width)
            if (len(color_map) == 0) :
                color_map.append(avg_colors)
                xy_map.append(0)

            else :
                found_group = False
                for i in range(0, len(color_map)) :
                    delta_e = delta(color_map[i], avg_colors)
                    #colors are similar enough to be the same
                    if (delta_e <= similar_thresh) :
                        xy_map.append(i)
                        found_group = True
                        break
                if (not found_group) :
                    xy_map.append(len(color_map))
                    color_map.append(avg_colors)
    return (xy_map, color_map)

def bfs(x_start, y_start, search_color, xy_map, rows_=6, cols_=7):
    found = 1
    neighbors = valid_colors(x_start, y_start, xy_map, search_color)
    q = queue.Queue()
    for neighbor in neighbors:
        q.put(neighbor)
    
    explored = {}
    for i in range(rows_*cols_):
        explored[i] = False
    explored[lin_xy(x_start, y_start)] = True

    while(not q.empty()):
        current_neighbor = q.get()
        y,x = current_neighbor
        linear_neighbor = lin_xy(x, y)
        if (explored[linear_neighbor] == True):
            continue
        explored[linear_neighbor] = True
        y_neighbor, x_neighbor = current_neighbor

        neighbors = valid_colors(x_neighbor, y_neighbor, xy_map, search_color)
        found += 1
        for neighbor in neighbors:
            q.put(neighbor)
    return found

def valid_colors(x, y, xy_map, search_color, rows_=6, cols_=7):
    #Assumes shift is being applied to the current row, y
    #UP, RIGHT, DOWN, LEFT
    neighbors = [(y-1, x),
                 (y, x+1),
                 (y+1, x),
                 (y, x-1)]
    for i in range(len(neighbors)):
        coord = neighbors[i]
        # Must shift in order to count it as a neighbor
        if (coord[0] < 0 or coord[0] >= rows_ or
            coord[1] < 0 or coord[1] >= cols_):
            neighbors[i] = -1

    same_neighbors = []
    for neighbor in neighbors:
        if(neighbor != -1):
            n_y, n_x = neighbor
            neighbor_color = xy(n_x, n_y, xy_map)
            if (neighbor_color == search_color):
                same_neighbors.append((n_y, n_x))
    return same_neighbors

def find_horizontal_moves(xy_map, rows_=6, cols_=7):
    horizontal_moves = []
    x = 0
    for y in range(rows_):
        score = 0
        for shift in range(1, cols_):
            roll_row(xy_map, y, 1)
            for x in range(cols_):
                search_color = xy(x, y, xy_map, 0)
                score = bfs(x, y, search_color, xy_map)
                if (score >= 3):
                    horizontal_moves.append(((y, 0, shift),0,score))
                    break
        roll_row(xy_map, y, 1)
    return horizontal_moves

def find_best_horizontal_move(xy_map, rows_=6, cols_=7):
    x = 0
    best_pair = (None, -1)
    for y in range(rows_):
        for shift in range(1, cols_):
            score = 0
            roll_row(xy_map, y, 1)
            for x in range(cols_):
                search_color = xy(x, y, xy_map, 0)
                potential_score = bfs(x, y, search_color, xy_map)
                if (potential_score >= 3):
                    score += potential_score
            _,best_score = best_pair
            if (score > best_score and score >= 3):
                best_pair = (((y, 0, shift),0),score)
        roll_row(xy_map, y, 1)
    best_move,best_score = best_pair
    return best_move

def find_vertical_moves(xy_map, rows_=6, cols_=7):
    vertical_moves = []
    y = 0
    for x in range(cols_):
        for shift in range(1, rows_):
            roll_col(xy_map, x, 1)
            for y in range(rows_):
                search_color = xy(x, y, xy_map, 0)
                score = bfs(x, y, search_color, xy_map)
                if (score >= 3):
                    vertical_moves.append(((0, x, shift),1,score))
                    break
        roll_col(xy_map, x, 1)
    return vertical_moves

def find_best_vertical_move(xy_map, rows_=6, cols_=7):
    y = 0
    best_pair = (None, -1)
    for x in range(cols_):
        for shift in range(1, rows_):
            score = 0
            roll_col(xy_map, x, 1)
            for y in range(rows_):
                search_color = xy(x, y, xy_map, 0)
                potential_score = bfs(x, y, search_color, xy_map)
                if (potential_score >= 3):
                    score += potential_score
                _,best_score = best_pair
            if (score > best_score and score >= 3):
                best_pair = (((0, x, shift),1),score)
        roll_col(xy_map, x, 1)
    best_move,best_score = best_pair
    return best_move

def xy_to_pixelcoord(x, y, sq_height, sq_width):
    # Image is divided into a grid of squares
    x_pixel = sq_width*x + sq_width/2
    y_pixel = sq_height*y + sq_height/2
    return (y_pixel, x_pixel)

def leftclick(y, x, duration=.1):
    mouseButton = 'left'
    pyat.moveTo(x, y, _pause=False)
    pyat.click(x, y, button=mouseButton)
    pyat.mouseDown(x, y, button=mouseButton)
    time.sleep(duration)
    pyat.mouseUp(button=mouseButton)

# duration is time in seconds
def dragMouse(start_y, start_x, end_y, end_x, safe_screenshot:list[bool]=[True], duration=.001):
    mouseButton = 'left'
    # A mouse drag is moving the mouse while holding down the left mouse button(LMB)
    safe_screenshot[0] = False
    pyat.moveTo(start_x, start_y, duration, _pause=False)
    pyat.mouseDown(button=mouseButton)
    pyat.moveTo(end_x, end_y, duration, _pause=False)
    if (SIMULATE_PLAYER):
        random_sleep = random.random()*.1
        time.sleep(random_sleep)
    time.sleep(duration)
    pyat.mouseUp(button=mouseButton)
    safe_screenshot[0] = True

def grayscale(arr:npt.ArrayLike) -> npt.ArrayLike:
    grayscale_array = np.mean(arr, axis=-1)
    return grayscale_array

#y,x
def plotX(src:Image.Image, point:tuple[int,int]):
    # y,x
    r,c = point

    plt.figure(figsize=(6, 6))

    shape = np.shape(src)
    # Display image
    if len(shape) == 2:
        plt.imshow(src, cmap='gray')
    else:
        plt.imshow(src)

    # Overlay red X
    plt.scatter(
        c, r,            # x = col, y = row
        marker='x',
        s=65,
        c='red',
        linewidths=1
    )

    plt.title("Template Match Location")
    plt.axis('off')
    plt.show()

# Returns top left pixel of template in src
def templateMatch(src:Image.Image, template:Image.Image, img_name = "templateMatch", confidence_interval=.9, save_img=False):
    src_gray = grayscale(src)
    template_gray = grayscale(template)

    src_f = src_gray.astype(np.float32)
    tpl_f = template_gray.astype(np.float32)

    out = cv2.matchTemplate(src_f, tpl_f, cv2.TM_CCOEFF_NORMED)
    topLeft = np.unravel_index(np.argmax(out), out.shape)
    confidence = out[topLeft]
    confident_match = True
    if (confidence < confidence_interval):
        confident_match = False
    
    if save_img:
        vis = out.copy()
        vis -= vis.min()
        if vis.max() > 0:
            vis /= vis.max()
        vis = (vis * 255).astype(np.uint8)
        Image.fromarray(vis, mode="L").save(img_name + ".png")
        plotX(src, topLeft)

    return (topLeft, confidence, confident_match)

def execute_move(move, sq_height, sq_width, x_offset, y_offset, safe_screenshot:list[bool]=[True], duration=.001, rows_=6, cols_=7):
    # move: ((y,x,shift),direction)
    # direction: 0 (horizontal)
    if (move is None):
        return
    move_info, direction = move
    y_move, x_move, shift = move_info
    y_pixel_start,x_pixel_start = xy_to_pixelcoord(x_move, y_move, sq_height, sq_width)
    # Horizontal move
    if (direction == 0):
        x_shifted = (x_move+shift)%cols_
        _,x_pixel_end = xy_to_pixelcoord(x_shifted, y_move, sq_height, sq_width)
        dragMouse(y_pixel_start+y_offset, x_pixel_start+x_offset,
                  y_pixel_start+y_offset, x_pixel_end+x_offset,
                  safe_screenshot=safe_screenshot, duration=duration)
    # Vertical move
    if (direction == 1):
        y_shifted = (y_move+shift)%rows_
        y_pixel_end,_ = xy_to_pixelcoord(x_move, y_shifted, sq_height, sq_width)
        dragMouse(y_pixel_start+y_offset, x_pixel_start+x_offset,
                  y_pixel_end+y_offset,   x_pixel_start+x_offset,
                  safe_screenshot=safe_screenshot, duration=duration)

"""     
def DEBUG_show_colors(color_map, xy_map):
    debug_name = "DEBUG_colors.png"
    shutil.copyfile(game_window_fn, debug_name)
    d_im = Image.open(debug_name)
    d_width = d_im.width
    d_height = d_im.height

    d_sq_width = (int) (d_width / cols_)
    d_sq_height = (int) (d_height / rows_)

    for x in range(0, cols_) :
        for y in range(0, rows_) : 
            
            rgb = color_map[xy(x, y, xy_map)]
            for sq_x in range(0, d_sq_width) :
                for sq_y in range(0, d_sq_height) :
                    d_im.putpixel((x * d_sq_width + sq_x, y * d_sq_height + sq_y),
                                  ((int) (rgb[0]),
                                   (int) (rgb[1]),
                                   (int) (rgb[2])))
    d_im.save(debug_name)

def DEBUG_show_xy_mappings(xy_map) :
    for i in range(0, cols_ * rows_) :
        if (i % cols_ == 0 and i != 0) :
            print("\n")
        print(xy_map[i])
"""

def relevantWindow(windowText:str, wordToFind:str=''):
    windowText = windowText.lower()
    wordToFind = wordToFind.lower()
    return wordToFind in windowText

def addToWindowList(hwnd:int, arr:list[int]):
    arr.append(hwnd)

def getWindowPos(hwnd:int):
    set_focus(hwnd)
    _,_,_,_,location = pyw.GetWindowPlacement(hwnd)
    x_start,y_start,_,_ = location
    x_start = max(0,x_start)
    y_start = max(0,y_start)
    return (y_start,x_start)

def windowCallback(hwnd:int, windowList:list[int]):
    windowText = pyw.GetWindowText(hwnd)
    if (relevantWindow(windowText, 'Wizard101')):
        addToWindowList(hwnd, windowList)

def save_bitmap(hwnd:int, filename:str, width=-1, height=-1, src_x_offset=0, src_y_offset=0, dest_x_offset=0, dest_y_offset=0, setFocus=True) -> None:
    try:
        width = max(0, width)
        height = max(0, height)
        if (setFocus):
            set_focus(hwnd)
        if (width <= 0 and height <= 0):
            _,_,width,height = pyw.GetClientRect(hwnd)
        wDC = pyw.GetWindowDC(hwnd)
        dcObj=pywui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        dataBitMap = pywui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((src_x_offset,src_y_offset), (width, height) , dcObj, (dest_x_offset,dest_y_offset), pywcon.SRCCOPY)
        dataBitMap.SaveBitmapFile(cDC, filename)
    except:
        _=0

def set_focus(hwnd:int) -> None:
    try:
        hwnd_tid,_ = pywproc.GetWindowThreadProcessId(hwnd)
        current_proc_id = pywapi.GetCurrentThreadId()
        pywproc.AttachThreadInput(current_proc_id, hwnd_tid, True)
        pyw.SetForegroundWindow(hwnd)
    except:
        _=0

def key_to_VK(key_char:str):
    return pywapi.VkKeyScan(key_char)

def key_pressed(key:str):
    key_char = key_to_VK(key)
    if (pywapi.GetAsyncKeyState(key_char) != 0):
        return True
    return False

def saveGameWindow(hwnd:int, fn:str, height:int, width:int, x_offset:int, y_offset:int, safe=list[bool], setFocus=True) -> None:
    try:
        while (not safe[0]):
            _ = 0
        save_bitmap(hwnd,fn, width, height, dest_x_offset=x_offset, dest_y_offset=y_offset, setFocus=setFocus)
    except:
        _ = 0
        #print("Couldn't save bitmap.")

def placeInQueue(queue:multiprocessing.Queue, item, queued_moves = dict[int,bool], drop_item=True, unique=False, timeout=0) -> None:
    item_hash = hash(item)
    item_exists = queued_moves.get(item_hash)
    if (item_exists is None):
        item_exists = False
    if (drop_item):
        try:
            if (unique):
                if (not item_exists):
                    queue.put(item, block=True, timeout=timeout)
                    queued_moves[item_hash] = True
            else:
                queue.put(item, block=True, timeout=timeout)
        except:
            return
    else:
        if (unique):
            if (not item_exists):
                queue.put(item, block=True)
                queued_moves[item_hash] = True
        else:
            queue.put(item, block=True)

def getScreenOffset(hwnd:int):
    y_screenOffset, x_screenOffset = getWindowPos(hwnd)
    return (y_screenOffset, x_screenOffset)

def setupGameWindowHwnd(hwnd:int, fn:str):
    set_focus(hwnd)
    y_screenOffset, x_screenOffset = getScreenOffset(hwnd)
    _,_,windowWidth, windowHeight = pyw.GetClientRect(hwnd)

    save_bitmap(hwnd, fn, windowWidth, windowHeight)
    src = openImage(fn)
    templateTL = openImage('topLeft.png')
    templateBR = openImage('botRight.png')
    topLeft,_,_ = templateMatch(src, templateTL, save_img=False)
    botRight,_,_ = templateMatch(src, templateBR, save_img=False)
    
    # Resolution: 1024x768
    widthOffset  = 19
    heightOffset = 19
    old_y,old_x = topLeft
    topLeft = (old_y+heightOffset, old_x+widthOffset)
    adjWidth  = botRight[1]-topLeft[1]
    adjHeight = botRight[0]-topLeft[0]
    y_gameOffset = y_screenOffset+topLeft[0]
    x_gameOffset = x_screenOffset+topLeft[1]
    return (adjHeight, adjWidth, (y_gameOffset, x_gameOffset), (y_screenOffset, x_screenOffset), topLeft)

# Pass in the filename of an image rather than a hwnd, useful for testing
def setupGameWindowFn(src_fn:str):
    src = openImage(src_fn)
    templateTL = openImage('topLeft.png')
    templateBR = openImage('botRight.png')
    topLeft,_,_ = templateMatch(src, templateTL, save_img=False)
    botRight,_,_ = templateMatch(src, templateBR, save_img=False)
    
    # Resolution: 1024x768
    widthOffset  = 19
    heightOffset = 19
    old_y,old_x = topLeft
    topLeft = (old_y+heightOffset, old_x+widthOffset)
    adjWidth  = botRight[1]-topLeft[1]
    adjHeight = botRight[0]-topLeft[0]
    return (adjHeight, adjWidth)

class SyncRef:
    sync_var = False
    def __init__(self):
        self.sync_var = False

    def getVar(self) -> bool:
        return self.sync_var

    def setVar(self, var:bool):
        self.sync_var = var

class ControlStruct:
    started = False
    obj = None
    name = ''
    target = None
    is_thread = True
    args = []
    sync_variable = None
    def __init__(self, name:str, target:callable, args:list=[], is_thread:bool=True, sync_variable:SyncRef=None):
        self.name = name
        self.target = target
        self.args = args
        self.is_thread = is_thread
        #if (not self.is_thread):
        #    self.obj = multiprocessing.Process(name=self.name, target=self.target, args=self.args)
        #self.obj = threading.Thread(name=self.name, target=self.target, args=self.args)
        self.sync_variable = sync_variable
        self.started = False

    def setArgs(self, args:list):
        self.args = args

    def getSyncRef(self) -> SyncRef:
        return self.sync_variable

    def restart(self):
        if (self.started == True):
            #if (not self.is_thread):
            #    self.obj = multiprocessing.Process(name=self.name, target=self.target, args=self.args)
            #self.obj = threading.Thread(name=self.name, target=self.target, args=self.args)
            self.started = False

    def start(self):
        if (not self.started):
            if (not self.is_thread):
                self.obj = multiprocessing.Process(name=self.name, target=self.target, args=self.args)
            self.obj = threading.Thread(name=self.name, target=self.target, args=self.args)
            self.obj.start()
            self.started = True

    def join(self):
        if (self.started):
            self.obj.join()

def findMoves(hwnd:int, fn:str, moves:multiprocessing.Queue, queued_moves:dict[int, bool], safe_to_screenshot:list[bool], curr_suffix:list[int], stop_event:list[bool], setFocus=True, rows_=6, cols_=7):
    unsafe_sync = [True]
    screen_sync = safe_to_screenshot

    if (not SYNC_SCREEN):
        screen_sync = unsafe_sync
    adjHeight, adjWidth, _, screen_offset, topLeft = setupGameWindowHwnd(hwnd, fn)
    y_offset, x_offset = topLeft
    saveGameWindow(hwnd, fn, adjHeight, adjWidth, x_offset, y_offset, safe=screen_sync, setFocus=setFocus)
    im_ = openImage(fn)
    sq_height = int(adjHeight / rows_)
    sq_width = int(adjWidth / cols_)
    max_retries = 20
    curr_retries = 0
    stop_bool = len(stop_event)-1

    while (curr_retries < max_retries and not stop_event[stop_bool]):
        xy_map = []
        xy_map,_ = processImage(im_, sq_height, sq_width)
        best_vertical = find_best_vertical_move(xy_map)
        if (best_vertical is not None):
            placeInQueue(moves, best_vertical, queued_moves=queued_moves, unique=True)
        if (moves.empty()):
            best_horizontal = find_best_horizontal_move(xy_map)
            if (best_horizontal is not None):
                placeInQueue(moves, best_horizontal, queued_moves=queued_moves, unique=True)

        if (moves.empty()):
            curr_retries += 1
            print(curr_retries)
        else:
            curr_retries = 0
        saveGameWindow(hwnd, fn, adjHeight, adjWidth, x_offset, y_offset, safe=screen_sync, setFocus=setFocus)
        im_ = openImage(fn)
    
    if (curr_retries >= max_retries):
        time.sleep(1)
        saveGameWindow(hwnd, fn, adjHeight, adjWidth, x_offset, y_offset, safe=unsafe_sync, setFocus=setFocus)
        print("No solutions found.")
        no_soln_fn = 'noSoln/NoSoln'+str(curr_suffix[0])+'.bmp'
        #save_bitmap(pyw.GetDesktopWindow(), no_soln_fn)
        save_bitmap(hwnd, no_soln_fn)
        curr_suffix[0] += 1
        stop_event[stop_bool] = True

def consumeMoves(hwnd:int, fn:str, moveArr:multiprocessing.Queue, queued_moves:dict[int,bool], safe_to_screenshot:list[bool], stop_event:list[bool], rows_=6, cols_=7):
    adjHeight, adjWidth, game_offset, screen_offset, topLeft = setupGameWindowHwnd(hwnd, fn)
    y_game_offset, x_game_offset = game_offset
    sq_height = int(adjHeight / rows_)
    sq_width = int(adjWidth / cols_)
    stop_bool = len(stop_event)-1
    while (not stop_event[stop_bool]):
        if (not moveArr.empty()):
            move = moveArr.get(block=False)
            execute_move(move, sq_height, sq_width, x_game_offset, y_game_offset, safe_screenshot=safe_to_screenshot, duration=.01)
            queued_moves[hash(move)] = False

#def stopThreads(stop_event:list[bool]):
def stopThreads(sync_refs:list[SyncRef], sync_ref:SyncRef):
    """
    while (not stop_event[0]):
        if (key_pressed('q')):
            for i in range(len(stop_event)):
                stop_event[i] = True
    """
    while (not sync_ref.getVar()):
        if (key_pressed('q')):
            for ref in sync_refs:
                ref.setVar(True)
            sync_ref.setVar(True)

def continuous_find_then_click(src_hwnd, fn, template:Image.Image, y_offset=0, x_offset=0, height=-1, width=-1,
                                                                   src_y_offset=0, src_x_offset=0, dest_y_offset=0, dest_x_offset=0, retries=5,
                                                                   confidence=.9, plot_instead_click=False, stop_event:list[bool]=[False], strict_matches_only:bool=False):
    # Want to click on the center of the template, template_match returns top left corner of match
    y_offset_template = int(template.height/2)
    x_offset_template = int(template.width/2)
    y_offset += y_offset_template
    x_offset += x_offset_template
    stop_idx = len(stop_event)-1
    
    if (retries < 0):
        retries = 9999

    while (retries >= 0):
        retries -= 1
        save_bitmap(src_hwnd, fn, height=height, width=width)
        src = openImage(fn)
        position,_,confident_match = templateMatch(src, template)
        if (stop_event[stop_idx]):
            return
        if ((confident_match) or not strict_matches_only):
            y, x = position
            y += y_offset
            x += x_offset
            if (plot_instead_click):
                save_bitmap(pyw.GetDesktopWindow(), fn, height=1080, width=1920)
                src = openImage(fn)
                plotX(src, (y,x))
            else:
                leftclick(y, x)
            return

def manage_game(hwnd:int, stop_event:list[bool], threads:list[ControlStruct], retries:int=1, time_between_checks:int=3, confidence:int=.9):
    fullscreen_fn = 'fullGameScreen.bmp'
    stop_bool = len(stop_event)-1
    while (not stop_event[1]):
        save_bitmap(hwnd, fullscreen_fn)
        src = openImage(fullscreen_fn)
        banner_template = openImage('PM_Banner.png')
        _,_,confident_match = templateMatch(src, banner_template)
        # Game is over, banner not found
        if (not confident_match):
            print("Game over, banner not found")
            stop_event[stop_bool] = True
            continue_template = openImage('PM_Continue.png')
            play_template = openImage('PM_Play.png')
            for thread in threads:
                thread.join()
            stop_event[stop_bool] = False

            y_offset_screen, x_offset_screen = getScreenOffset(hwnd)
            continuous_find_then_click(hwnd, fullscreen_fn, continue_template, retries=retries,
                                       y_offset=y_offset_screen, x_offset=x_offset_screen,
                                       plot_instead_click=False, confidence=confidence, stop_event=stop_event)
            
            continuous_find_then_click(hwnd, fullscreen_fn, play_template, retries=retries,
                                       y_offset=y_offset_screen, x_offset=x_offset_screen,
                                       plot_instead_click=False, confidence=confidence, stop_event=stop_event)
            for thread in threads:
                #thread.restart()
                thread.start()
        else:
            for thread in threads:
                thread.start()
        time.sleep(time_between_checks)
    
    for thread in threads:
        thread.join()

def getWizardWindow() -> int:
    windowList = []
    pyw.EnumWindows(windowCallback, windowList)
    #if (len(windowList) <= 0 or len(windowList) > 1):
    if (len(windowList) <= 0):
        raise Exception('Problem finding Wizard101 window')
    # Assume only 1 window open for now
    wizardHwnd:int = windowList[0]
    return wizardHwnd

def main():
    currPid = os.getpid()
    game_window_fn = 'gameWindow.bmp'
    #stored as [color_map index (row_0), ... (row 0), color_map index (row 0), color_map index (row 1),]

    wizardHwnd = getWizardWindow()

    maxSize = 8
    moves = multiprocessing.Queue(maxSize)
    queued_moves = {}
    safe_screenshot = [True]
    stop_event = [False, False, False]
    curr_suffix = [0]
    retries = 0
    time_between_checks = 3
    manager_confidence = .8
    proc_screenshot_args = [wizardHwnd, game_window_fn, moves, queued_moves, safe_screenshot, curr_suffix, stop_event]
    thr_mouse_args = [wizardHwnd, game_window_fn, moves, queued_moves, safe_screenshot, stop_event]

    #proc_screenshot = ControlStruct(name='Screen Shot', target=findMoves, args=proc_screenshot_args, is_thread=False)
    #thread_mouse_movement = ControlStruct('Mouse', target=consumeMoves, args=thr_mouse_args)
    #thread_stop_event = ControlStruct('Stop Thread', target=stopThreads, args=[stop_event])

    """
    proc_screenshot = ControlStruct(name='Screen Shot', target=findMoves, is_thread=False, sync_variable=[stop_event[0]])
    proc_screenshot.setArgs([wizardHwnd, game_window_fn, moves, queued_moves, safe_screenshot, curr_suffix, proc_screenshot.getSyncRef()])
    thread_mouse_movement = ControlStruct('Mouse', target=consumeMoves)
    thread_mouse_movement.setArgs([wizardHwnd, game_window_fn, moves, queued_moves, safe_screenshot, thread_mouse_movement.getSyncRef()])
    thread_stop_event = ControlStruct('Stop Thread', target=stopThreads)
    thread_stop_event.setArgs([thread_stop_event.getSyncRef()])

    helper_structs = [proc_screenshot,
                      thread_mouse_movement
                      ]
    thr_manager_args = [wizardHwnd, stop_event, helper_structs, retries, time_between_checks, manager_confidence]
    
    thread_manager = threading.Thread(name='Game Manager', target=manage_game, args=thr_manager_args)

    thread_stop_event.start()
    thread_manager.start()

    thread_stop_event.join()
    thread_manager.join()
    """
    #thread_stop_event = ControlStruct('Stop Thread', target=stopThreads, sync_variable=SyncRef())
    #thread_stop_event.setArgs([[], thread_stop_event.getSyncRef()])

    #thread_stop_event.start()
    #thread_stop_event.join()

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

    main()