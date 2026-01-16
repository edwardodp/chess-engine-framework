import numpy as np
from numba import njit

# gets piece at specific square
@njit
def get_piece(board_data, file, rank):
    shifts = ((rank - 1) * 8 + (ord(file) - ord('a')))
    mask = 1 << shifts
    for i in range(len(board_data)):
        val = int(board_data[i])
        if val & mask:
            if i == 0: return 1
            elif i == 1 or i == 2: return 3
            elif i == 3: return 5
            elif i == 4: return 9
            elif i == 5: return 200
            elif i == 6: return -1
            elif i == 7 or i == 8: return -3
            elif i == 9: return -5
            elif i == 10: return -9
            else: return -200
    return 0 # empty

# checks if a white or black piece is at a square
@njit
def check_square(board_occupancy_data, file, rank, colour):
    shifts = ((rank - 1) * 8 + (ord(file) - ord('a')))
    mask = 1 << shifts
    if board_occupancy_data[colour] & mask: return True
    else: return False
