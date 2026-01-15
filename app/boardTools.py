import numpy as np
from numba import njit

@njit
def get_piece(board_pieces_data, rank, file):
    shifts = 63 - ((file - 1) * 9 + (ord(rank) - ord('a')))
    mask = 1 << shifts
    for i in range(len(board_pieces_data)):
        if board_pieces_data[i] & mask:
            if i == 0:
                return 1
            else if i == 1 or i == 2:
                return 3
            else if i == 3:
                return 5
            else if i == 4:
                return 9
            else if i == 5:
                return 200
            else if i == 6:
                return -1
            else if i == 7 or i == 8:
                return -3
            else if i == 9:
                return -5
            else if i == 10:
                return -9
            else:
                return -200

@njit
def check_square(board_occupancy_data, rank, file, colour):
    shifts = 63 - ((file - 1) * 9 + (ord(rank) - ord('a')))
    mask = 1 << shifts
    if board_occupancy_data[colour] & mask:
        return True
    else:
        return False

# function to get outposts

