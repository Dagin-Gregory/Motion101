import queue
from math import sqrt
from PIL import Image
import multiprocessing
import time
from multiprocessing.sharedctypes import Synchronized

import Helpers.HelperMouse as helper_mouse
import Helpers.HelperControl as helper_control
import Helpers.HelperScreen as helper_screen

SYNC_SCREEN = True
HORIZONTAL_MOVE = 0
VERTICAL_MOVE = 1

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
    # -1 is an invalid color and cannot be searched on
    if (search_color == -1):
        return []
    # Assumes shift is being applied to the current row, y
    # UP, RIGHT, DOWN, LEFT
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
                    horizontal_moves.append(((y, 0, shift), HORIZONTAL_MOVE, score))
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
                    vertical_moves.append(((0, x, shift), VERTICAL_MOVE, score))
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
    y_pixel = sq_height*y + sq_height/2
    x_pixel = sq_width*x + sq_width/2
    return (y_pixel, x_pixel)

# duration is time in seconds
def execute_move(move, sq_height, sq_width, x_offset, y_offset, safe_screenshot:Synchronized, duration=.001, rows_=6, cols_=7):
    # move: ((y,x,shift),direction)
    # direction: 0 (horizontal)
    if (move is None):
        return
    move_info, direction = move
    y_move, x_move, shift = move_info
    y_pixel_start,x_pixel_start = xy_to_pixelcoord(x_move, y_move, sq_height, sq_width)
    # Horizontal move
    if (direction == HORIZONTAL_MOVE):
        x_shifted = (x_move+shift)%cols_
        _,x_pixel_end = xy_to_pixelcoord(x_shifted, y_move, sq_height, sq_width)
        helper_mouse.dragMouse(y_pixel_start+y_offset, x_pixel_start+x_offset,
                               y_pixel_start+y_offset, x_pixel_end+x_offset,
                               safe_screenshot=safe_screenshot, duration=duration)
    # Vertical move
    if (direction == VERTICAL_MOVE):
        y_shifted = (y_move+shift)%rows_
        y_pixel_end,_ = xy_to_pixelcoord(x_move, y_shifted, sq_height, sq_width)
        helper_mouse.dragMouse(y_pixel_start+y_offset, x_pixel_start+x_offset,
                               y_pixel_end+y_offset,   x_pixel_start+x_offset,
                               safe_screenshot=safe_screenshot, duration=duration)

def DEBUG_show_colors(d_im:Image.Image, color_map, xy_map, cols_=7, rows_=6) :
    debug_name = "DEBUG_colors.png"
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

def findMoves(moves:multiprocessing.Queue, queued_moves:dict[int, bool], safe_to_screenshot:Synchronized, curr_suffix:Synchronized,
              control_struct:helper_control.ControlStruct, screen_capture:helper_screen.ScreenCapture, curr_retries:Synchronized, no_soln_path:str,
              prev_xy_map_sync:Synchronized, wait_until_settled=True, set_focus=True, rows_=6, cols_=7):
    
    adjHeight, adjWidth, _, screen_offset, topLeft = screen_capture.setupGameWindowPM()
    y_offset, x_offset = topLeft
    game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=safe_to_screenshot, set_focus=set_focus)
    #game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=screen_sync, set_focus=set_focus, save_to_file=True)
    if (game_window is str):
        game_window = helper_screen.openImage(game_window)
    if (game_window is None):
        return
    game_window:Image.Image

    sq_height = int(adjHeight / rows_)
    sq_width = int(adjWidth / cols_)
    max_retries = 20

    if (curr_retries.value < max_retries):
        xy_map = []
        xy_map,_ = processImage(game_window, sq_height, sq_width)

        # Only find moves when the entire board is settled
        """
        if (wait_until_settled):
            with prev_xy_map_sync.get_lock():
                prev_xy_map = prev_xy_map_sync.get_obj()
                for i in range(len(xy_map)):
                    if (xy_map[i] != prev_xy_map[i]):
                        for i in range(len(xy_map)):
                            prev_xy_map[i] = xy_map[i]
                        return
        """
        # Find matches only using settled squares
        if (wait_until_settled):
            with prev_xy_map_sync.get_lock():
                prev_xy_map = prev_xy_map_sync.get_obj()
                for i in range(len(xy_map)):
                    prev_val = xy_map[i]
                    if (xy_map[i] != prev_xy_map[i]):
                        xy_map[i] = -1
                    prev_xy_map[i] = prev_val
        
        # Finding best vertical moves
        """
        best_vertical = find_best_vertical_move(xy_map)
        if (best_vertical is not None):
            helper_control.placeInQueue(moves, best_vertical, queued_moves=queued_moves, unique=True, drop_item=False)

        best_horizontal = find_best_horizontal_move(xy_map)
        if (best_horizontal is not None):
            helper_control.placeInQueue(moves, best_horizontal, queued_moves=queued_moves, unique=True, drop_item=False)
        """

        # rows 0-5
        horizontal_moves = find_horizontal_moves(xy_map)
        # cols 0-6
        vertical_moves = find_vertical_moves(xy_map)
        # IDEA: Rather than finding the highest value move, we want to clear the bottom bottles as often as possible
        # Less 'stale' sections of the board should lead to longer games thus higher scores
        # do this by taking the first 'k' moves for the last col/row
        k = 1
        for it in range(min(k, len(vertical_moves))):
            idx = len(vertical_moves)-it-1
            move, dir, score = vertical_moves[idx]
            move_dir = (move, dir)
            helper_control.placeInQueue(moves, move_dir, queued_moves=queued_moves, unique=True, drop_item=False)
        for it in range(min(k, len(horizontal_moves))):
            idx = len(horizontal_moves)-it-1
            move, dir, score = horizontal_moves[idx]
            move_dir = (move, dir)
            helper_control.placeInQueue(moves, move_dir, queued_moves=queued_moves, unique=True, drop_item=False)
        

        if (moves.empty()):
            with curr_retries.get_lock():
                curr_retries.value += 1
        else:
            with curr_retries.get_lock():
                curr_retries.value = 0
        game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=safe_to_screenshot, set_focus=set_focus)
    
    else:
        time.sleep(1)
        game_window = screen_capture.saveGameWindow(adjHeight, adjWidth, x_offset, y_offset, safe=safe_to_screenshot, set_focus=set_focus)
        xy_map,color_map = processImage(game_window, sq_height, sq_width)
        DEBUG_show_colors(game_window, color_map, xy_map)
        print("No solutions found.")
        no_soln_fn = no_soln_path+'NoSoln'+str(curr_suffix.value)+'.bmp'
        bitmap_return = screen_capture.saveBitmap(fn=no_soln_fn, save_to_file=True)

        with curr_suffix.get_lock():
            curr_suffix.value += 1

        with curr_retries.get_lock():
            curr_retries.value = 0
            
        control_struct.pause()

def consumeMoves(screen_capture:helper_screen.ScreenCapture, moveArr:multiprocessing.Queue, queued_moves:dict[int,bool], safe_to_screenshot:Synchronized,
                 rows_=6, cols_=7):
    adjHeight, adjWidth, game_offset, screen_offset, topLeft = screen_capture.setupGameWindowPM()
    #while (adjHeight < 0 or adjWidth < 0):
    #    adjHeight, adjWidth, game_offset, screen_offset, topLeft = screen_capture.setupGameWindowPM()
    if (adjHeight <= 0 or adjWidth <= 0):
        return

    #_ = setupGameWindowPM(hwnd, fn)
    #print(_)
    #adjHeight, adjWidth, game_offset, screen_offset, topLeft = _

    y_game_offset, x_game_offset = game_offset
    sq_height = int(adjHeight / rows_)
    sq_width = int(adjWidth / cols_)
    if (not moveArr.empty()):
        move = moveArr.get(block=False)
        execute_move(move, sq_height, sq_width, x_game_offset, y_game_offset, safe_screenshot=safe_to_screenshot, duration=.01)
        queued_moves[hash(move)] = False

def continuous_find_then_click(screen_capture:helper_screen.ScreenCapture, template:Image.Image, y_offset=0, x_offset=0, height=-1, width=-1, retries=5,
                               confidence=.9, plot_instead_click=False, strict_matches_only:bool=False):
    # Want to click on the center of the template, template_match returns top left corner of match
    y_offset_template = int(template.height/2)
    x_offset_template = int(template.width/2)
    y_offset += y_offset_template
    x_offset += x_offset_template

    if (retries < 0):
        retries = 9999

    while (retries >= 0):
        retries -= 1
        src = screen_capture.saveBitmap(height=height, width=width)
        position,_,confident_match = helper_screen.templateMatch(src, template, confidence_interval=confidence)
        if ((confident_match) or not strict_matches_only):
            y, x = position
            y += y_offset
            x += x_offset
            #y += y_offset_template
            #x += x_offset_template
            if (plot_instead_click):
                src = screen_capture.saveBitmap(height=height, width=width)
                helper_screen.plotX(src, (y,x))
            else:
                helper_mouse.leftclick(y, x)
            return

def manage_game(control_struct:helper_control.ControlStruct, screen_capture:helper_screen.ScreenCapture, 
                banner_template:Image.Image, continue_template:Image.Image, play_template:Image.Image,
                retries:int=10, time_between_checks:int=3, confidence:int=.8):
    if (banner_template is None or continue_template is None or play_template is None):
        raise Exception('A passed template image is none')
    
    bitmap_return = screen_capture.saveBitmap()
    window_screenshot = bitmap_return
    if (window_screenshot is str):
        window_screenshot = helper_screen.openImage(window_screenshot)
    if (window_screenshot is None):
        return
    _,_,confident_match = helper_screen.templateMatch(window_screenshot, banner_template)
    # Game is over, banner not found
    if (not confident_match):
        with control_struct.sync_variable.get_lock():
            if (control_struct.sync_variable.value != control_struct.RUNNING):
                return
        print("Game over, banner not found")

        control_struct.pauseChildren()
        y_offset_screen, x_offset_screen = helper_screen.getScreenOffset(screen_capture.hwnd)
        continuous_find_then_click(screen_capture, continue_template, retries=retries,
                                    y_offset=y_offset_screen, x_offset=x_offset_screen,
                                    plot_instead_click=False, confidence=confidence)
        
        continuous_find_then_click(screen_capture, play_template, retries=retries,
                                    y_offset=y_offset_screen, x_offset=x_offset_screen,
                                    plot_instead_click=False, confidence=confidence)
        
        control_struct.unpause(unpause_all_offspring=True)
    time.sleep(time_between_checks)

