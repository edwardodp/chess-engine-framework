import numpy as np
from numba import njit

# func to get
@njit
def get_piece(board_data, rank, file):
    shifts = 63 - ((file - 1) * 9 + (ord(rank) - ord('a')))
    mask = 1 << shifts
    for i in range(len(board_data)):
        if board_data[i] & mask:
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
    

# func to get what piece is at a square (given chess coords e.g. a1 given as 2 parameters a, 1)

# func to get value of a piece at a square (given chess coords)

# func to get what peices are on row (ranks)

# func to get what pieces are on column (files)

# func to check if king in check

# get all squares that a specific peice type is on e.g get all squares that white pawns are on

# function to get outposts

# function to get fianchetto bishops

# function to get mobility of white (number of legal moves)

# function to get mobility of black (number of legal moves)

# function to return move count
