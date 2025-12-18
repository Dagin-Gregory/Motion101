from PIL import Image
from math import *
import shutil
import queue
import time
import pyautogui as pyat
import win32gui as pyw
import win32ui as pywui
import win32api as pywapi
import win32process as pywproc

DEBUG = True

def open_image(img_name):
    image_ptr = Image.open(img_name)
    return image_ptr

def delta(color_a, color_b) :
    delta_E = sqrt( (color_a[0] - color_b[0]) ** 2 +
                    (color_a[1] - color_b[1]) ** 2 +
                    (color_a[2] - color_b[2]) ** 2)
    return delta_E

def xy(x, y, xy_map, shift=0) :
    idx = y*cols_ + ((x+shift)%cols_) 
    value = xy_map[idx]

    return value

def process_square(image, x_off, y_off, sq_height, sq_width):
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

    r_avg /= total_pixels
    g_avg /= total_pixels
    b_avg /= total_pixels
    return [r_avg, g_avg, b_avg]

def process_image(image, sq_height, sq_width) :
    color_map = []
    xy_map = []
    for curr_row in range(0, rows_) : 
        for curr_col in range(0, cols_) :
            x_offset = curr_col * sq_width
            y_offset = curr_row * sq_height
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

def bfs(x_start, y_start, xy_map, shift):
    found = 1
    thresh = 3
    search_color = xy(x_start, y_start, xy_map, shift)
    neighbors = valid_colors(x_start, y_start, xy_map, shift, search_color)
    q = queue.Queue()
    for neighbor in neighbors:
        q.put(neighbor)
    
    while(not q.empty()):
        current_neighbor = q.get()
        y_neighbor = current_neighbor[0]
        x_neighbor = current_neighbor[1]

        shift_neighbor = 0
        if (y_neighbor==y_start):
            shift_neighbor = shift
        neighbors = valid_colors(x_neighbor, y_neighbor, xy_map, shift_neighbor, search_color)
        found += len(neighbors)
        if (found >= thresh):
            return True
        for neighbor in neighbors:
            q.put(neighbor)
    return False

def valid_colors(x, y, xy_map, shift, search_color):
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
        """
        if (coord[0] < 0):
            neighbors[i] = (rows_-1, coord[1])
        if (coord[0] >= rows_):
            neighbors[i] = (0, coord[1])

        if (coord[1] < 0):
            neighbors[i] = (coord[0], cols_-1)
        if (coord[1] >= cols_):
            neighbors[i] = (coord[0], 0)
        """

    same_neighbors = []
    for neighbor in neighbors:
        if(neighbor != -1):
            neighbor_color = xy(neighbor[1], neighbor[0], xy_map, shift)
            if (neighbor_color == search_color):
                same_neighbors.append((neighbor[0], neighbor[1]))
    return same_neighbors

def find_horizontal_moves(xy_map):
    horizontal_moves = []
    x = 0
    for y in range(0, rows_):
        for shift in range(1, int(cols_/2)):
            if (bfs(x, y, xy_map, shift)):
                horizontal_moves.append((y,x,shift))
    return horizontal_moves

#def find_vertical_moves():

def xy_to_pixelcoord(x, y, sq_height, sq_width):
    # Image is divided into a grid of squares
    x_pixel = sq_width*x + sq_width/2
    y_pixel = sq_height*y + sq_height/2
    return (y_pixel, x_pixel)
    
# duration is time in seconds
def dragMouse(start_y, start_x, end_y, end_x, duration=.5):
    mouseButton = 'left'
    # A mouse drag is moving the mouse while holding down the left mouse button(LMB)
    pyat.moveTo(start_x, start_y, duration)
    pyat.mouseDown(button=mouseButton)
    pyat.moveTo(end_x, end_y, duration)
    pyat.mouseUp(button=mouseButton)

def execute_move(move, sq_height, sq_width, x_offset, y_offset):


    # move: ((y,x,shift),direction)
    # direction: 0 (horizontal)
    move_info, direction = move
    y_move, x_move, shift = move_info
    y_pixel_start,x_pixel_start = xy_to_pixelcoord(x_move, y_move, sq_height, sq_width)
    # Horizontal move
    if (direction == 0):
        x_shifted = (x_move+shift)%cols_
        x_pixel_end = xy_to_pixelcoord(x_shifted, y_move, sq_height, sq_width)
        dragMouse(y_pixel_start+y_offset, x_pixel_start+x_offset,
                  y_pixel_start+y_offset, x_pixel_end+x_offset,
                  duration=.25)
        

def DEBUG_show_colors(color_map, xy_map) :
    debug_name = "DEBUG_colors.png"
    shutil.copyfile(file_name, debug_name)
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

def relevantWindow(windowText, wordToFind):
    windowText = windowText.lower()
    wordToFind = wordToFind.lower()
    return wordToFind in windowText

def addToWindowList(hwnd, arr:list[int]):
    arr.append(hwnd)

def getWindowPos(hwnd):
    pyw.SetFocus(hwnd)
    _,_,_,_,location = pyw.GetWindowPlacement(hwnd)
    x_start,y_start,_,_ = location
    x_start = max(0,x_start)
    y_start = max(0,y_start)
    return (y_start,x_start)

def callback(hwnd, _):
    windowText = pyw.GetWindowText(hwnd)
    keyword = "Wizard101"
    if (relevantWindow(windowText, keyword)):
        addToWindowList(hwnd, windowList)

def main():
    color_map = []
    #stored as [color_map index (row_0), ... (row 0), color_map index (row 0), color_map index (row 1),]
    xy_map = []

    start = time.process_time()

    im_ = open_image(file_name)
    sq_width = int(im_.width / cols_)
    sq_height = int(im_.height / rows_)

    xy_map,color_map = process_image(im_, sq_height, sq_width)

    end = time.process_time()
    if (DEBUG):
        print( "Time to process image: ", (end - start) * 1000, "ms")
        print ("Types of bottles :", len(color_map))
        print ("Number of xy mappings :", len(xy_map))

    start = time.time()

    horizontal_moves = find_horizontal_moves(xy_map)

    end = time.time()
    if (DEBUG):
        print("Time to find horizontal moves: ", (end - start), "s")
        print("Horizontal moves: ", len(horizontal_moves))
        #print(horizontal_moves)

    #if (DEBUG):
    #    DEBUG_show_colors(color_map, xy_map)
    #    DEBUG_show_xy_mappings(xy_map)

if __name__ == '__main__':
    #file_name = "Test_img.png"
    file_name = "Test_6.png"

    rows_ = 6
    cols_ = 7
    similar_thresh = 10
    windowList = []
    x_screen_offset = 0
    y_screen_offset = 0
    width = 0
    height = 0
    current_proc_id = pywapi.GetCurrentThreadId()

    pyw.EnumWindows(callback, None)
    if (len(windowList) <= 0 or len(windowList) > 1):
        raise Exception("Problem finding Wizard101 window")
    # Assume only 1 window open for now
    wizardHwnd = windowList[0]
    wizardThreadId,_ = pywproc.GetWindowThreadProcessId(wizardHwnd)
    pywproc.AttachThreadInput(current_proc_id, wizardThreadId, True)
    
    y_screen_offset,x_screen_offset = getWindowPos(wizardHwnd)
    pyw.SetFocus(wizardHwnd)
    _,_,width,height = pyw.GetClientRect(wizardHwnd)
    bmp = pywui.CreateBitmap()
    windc = pyw.GetWindowDC(wizardHwnd)
    #pyw.DC
    #cbmp = pyw.CreateCompatibleBitmap(bmp)
    
    #pyw.CreateBitmap()
    #main()