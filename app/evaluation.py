import numpy as np
from numba import njit
import boardTools as bt

# pieces
PAWN_WHITE = 1
PAWN_BLACK = -1
KNIGHT_WHITE = 3
KNIGHT_BLACK = -3
BISHOP_WHITE = 3
BISHOP_BLACK = -3
ROOK_WHITE = 5
ROOK_BLACK = -5
QUEEN_WHITE = 9
QUEEN_BLACK = -9
KING_WHITE = 200
KING_BLACK = -200

@njit
def evaluation_function(board_pieces_data, board_occupancy_data, move_count):
    evaluation = 0
    # Your code goes here!

    return evaluation
