from PIL import Image
from math import sqrt
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import Queue

import Helpers.HelperMouse as helper_mouse

# TODO: Get real values for row and col
ROWS = 15
COLS = 7

INVALID = -1

LEFT_MOVE = 0
RIGHT_MOVE = 1

def delta(color_a, color_b) -> float:
    delta_E = sqrt( (color_a[0] - color_b[0]) ** 2 +
                    (color_a[1] - color_b[1]) ** 2 +
                    (color_a[2] - color_b[2]) ** 2)
    return delta_E

def process_square(image:Image.Image, x_off, y_off) -> tuple[int, int, int]:
    sq_height = image.height/ROWS
    sq_width = image.width/COLS
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

# TODO: Make sure black/background squares are marked as -1
def processImage(image:Image.Image, similar_thresh=10) -> tuple[list[int], list[int]]:
    sq_height = image.height/ROWS
    sq_width = image.width/COLS
    color_map = []
    xy_map = []
    for curr_row in range(0, ROWS) : 
        for curr_col in range(0, COLS) :
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
    return (xy_map.copy(), color_map.copy())

def _lin_xy(x, y, shift=0, shift_y=0) -> int:
    linear_idx = ((y-shift_y)%ROWS)*COLS + ((x-shift)%COLS)
    return linear_idx

def xy(x, y, xy_map, shift=0, shift_y=0) -> int:
    linear_idx = _lin_xy(x, y, shift, shift_y)
    value = xy_map[linear_idx]

    return value

def xy_to_pixelcoord(image:Image.Image, x, y):
    sq_height = image.height/ROWS
    sq_width = image.width/COLS
    # Image is divided into a grid of squares
    y_pixel = sq_height*y + sq_height/2
    x_pixel = sq_width*x + sq_width/2
    return (y_pixel, x_pixel)

def execute_move(move, image:Image.Image, x_offset, y_offset, safe_screenshot:Synchronized, duration=.001):
    # move: ((y,x),direction)
    if (move is None):
        return
    move_info, direction = move
    y_move, x_move = move_info
    # Horizontal move
    if (direction == LEFT_MOVE):
        y_pixel,x_pixel = xy_to_pixelcoord(image, x_move-1, y_move)
        helper_mouse.leftclick(y_pixel+y_offset, x_pixel+x_offset)
    # Vertical move
    if (direction == RIGHT_MOVE):
        y_pixel,x_pixel = xy_to_pixelcoord(image, x_move+1, y_move)
        helper_mouse.leftclick(y_pixel+y_offset, x_pixel+x_offset)

def _swapBlocks(xy_map:list[int], x:int, y:int, swap_dir:int) -> tuple[list[int], tuple[int, int]]:
    xy_return = xy_map.copy()
    x_move = 0
    if (swap_dir == LEFT_MOVE):
        x_move = -1
    if (swap_dir == RIGHT_MOVE):
        x_move = 1
    tmp = xy_return[_lin_xy(x+x_move, y)]
    xy_return[_lin_xy(x+x_move, y)] = xy_return[_lin_xy(x, y)]
    xy_return[_lin_xy(x, y)] = tmp
    # Make sure both columns fully drop
    curr_col_drop = 0
    moved_col_drop = 0
    for check_row in range(y-1, -1, -1):
        if (xy(x, check_row, xy_return) != -1):
            break
        curr_col_drop += 1
    for check_row in range(y-1, -1, -1):
        if (xy(x+x_move, check_row, xy_return) != -1):
            break
        moved_col_drop += 1

    new_y = y-moved_col_drop
    new_x = x+x_move

    for i in range(y, ROWS):
        xy_return[_lin_xy(x, i-curr_col_drop)] = xy_return[_lin_xy(x, i)]
        xy_return[_lin_xy(x+x_move, i-moved_col_drop)] = xy_return[_lin_xy(x+x_move, i)]

    return (xy_return.copy(), (new_y, new_x))

# Can only cancel in cardinal directions
def _blocksCancelled(xy_map:list[int], x:int, y:int) -> int:
    horizontal_cancel = 0
    vertical_cancel = 0
    curr_color = xy(x, y, xy_map)
    for check_x in range(x-1,-1,-1):
        if (xy(check_x, y, xy_map) != curr_color):
            break
        horizontal_cancel += 1
    for check_x in range(x+1, COLS):
        if (xy(check_x, y, xy_map) != curr_color):
            break
        horizontal_cancel += 1
    
    for check_y in range(y-1,-1,-1):
        if (xy(x, check_y, xy_map) != curr_color):
            break
        vertical_cancel += 1
    for check_y in range(y+1, ROWS):
        if (xy(x, check_y, xy_map) != curr_color):
            break
        vertical_cancel += 1

    total_cancelled = 0
    if (horizontal_cancel >= 3):
        total_cancelled += horizontal_cancel
    if (vertical_cancel >= 3):
        total_cancelled += vertical_cancel
    return total_cancelled

# TODO
def _searchMoveSpace(xy_map:list[int], x:int, y:int, max_moves:int, find_all_moves=False) -> list[tuple[list[tuple[int, int]], int]]:
    # (board:list[int], move_list:list[tuple[int, int]], MOVE_DIR)
    # we always intend to check the last move in move_list
    valid_point_sequences:list[tuple[list[tuple[int, int]], int]] = []
    moves_to_check = Queue()
    explored_boards:dict[int, bool] = {}
    moves_to_check.put((xy_map, [(y,x)], LEFT_MOVE))
    moves_to_check.put((xy_map, [(y,x)], RIGHT_MOVE))
    while(not moves_to_check.empty()):
        board, move_list, move_dir = moves_to_check.get(block=False)
        board_hash = hash(board)
        is_explored = explored_boards.get(board_hash)
        if (is_explored == None):
            continue
        if (is_explored):
            continue
        moves_used = len(move_list)
        if (moves_used > max_moves):
            continue
        board:list[int]
        move_list:list[tuple[int, int]]
        y, x = move_list[len(move_list)-1]
        new_board, new_position = _swapBlocks(board, x, y, move_dir)
        new_y, new_x = new_position
        score = _blocksCancelled(new_board, new_x, new_y)
        if (score >= 3):
            valid_point_sequences.append((move_list, score))
            if (not find_all_moves):
                break
        explored_boards[hash(board)] = True
        move_list.append(new_position)
        moves_to_check.put((new_board, move_list, LEFT_MOVE))
        moves_to_check.put((new_board, move_list, RIGHT_MOVE))

    return valid_point_sequences

def _findMoveNaive(xy_map:list[int], max_moves:int=10):
    # List of sequences of moves that yield points, each idx of potential moves
    # is a list of moves that yield points when a specific block is moved
    potential_moves:list[list[tuple[list[tuple[int, int]], int]]] = []
    # The background color is black, which is preprocessed to be -1 to indicate it should be ignored
    for y in range(ROWS-1, -1, -1):
        for x in range(COLS):
            # first instance of a block that should be solved for
            if (xy(x, y, xy_map) != INVALID):
                valid_moves = _searchMoveSpace(xy_map)
                if (len(valid_moves) > 0):
                    potential_moves.append(valid_moves)
