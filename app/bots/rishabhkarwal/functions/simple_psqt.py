"""Simple piece-square table implementation"""

import board_tools as bt
from numba import njit, int64, int32, uint32, uint64
import numpy as np

# piece values
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 100, 320, 330, 500, 900, 20000

# piece-square tables
# NOTE: defined from white's perspective

# pawn: encourage pushing to centre and promotion
_p = (
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
)

# knight: strong in centre, weak in corners
_n = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
)

# bishop: avoid corners, control diagonals
_b = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
)

# rook: loves 7th rank and centre files
_r = (
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
)

# queen: slight preference for centre, keep mobile
_q = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,   0,  5,  5,  5,  5,  0, -5,
     0,   0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
)

# king: hide in castle during middlegame (safety)
_k = (
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
)

def flip_and_flatten(table_tuple):
    """
    Takes a tuple defined visually (Rank 8 top, Rank 1 bottom)
    and flips it to match Little-Endian Bitboard memory (Rank 1 at start).
    """
    return np.array(table_tuple, dtype=np.int32).reshape(8, 8)[::-1].flatten()

PSQTs = np.zeros((6, 64), dtype=np.int32)
PSQTs[0] = flip_and_flatten(_p) + PAWN
PSQTs[1] = flip_and_flatten(_n) + KNIGHT
PSQTs[2] = flip_and_flatten(_b) + BISHOP
PSQTs[3] = flip_and_flatten(_r) + ROOK
PSQTs[4] = flip_and_flatten(_q) + QUEEN
PSQTs[5] = flip_and_flatten(_k) + KING

DE_BRUIJN_MAGIC = uint64(0x03f79d71b4cb0a89)
DE_BRUIJN_LOOKUP = np.array([
    0,   1, 48,  2, 57, 49, 28, 3,
    61, 58, 50, 42, 38, 29, 17, 4,
    62, 55, 59, 36, 53, 51, 43, 22,
    45, 39, 33, 30, 24, 18, 12, 5,
    63, 47, 56, 27, 60, 41, 37, 16,
    54, 35, 52, 21, 44, 32, 23, 11,
    46, 26, 40, 15, 34, 20, 31, 10,
    25, 14, 19,  9, 13,  8,  7,  6
], dtype=np.int32)

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    score = 0
    occupancy = uint64(board_occupancy[2])

    # local references of globals
    tables = PSQTs
    lookup = DE_BRUIJN_LOOKUP
    magic = DE_BRUIJN_MAGIC

    while occupancy:
        lsb = occupancy & (~occupancy + uint64(1))
        sq = lookup[((lsb * magic) >> uint64(58))] # de bruijn scan
        
        piece = bt.get_piece(board_pieces, sq)
        # white
        if piece <= 5: score += tables[piece, sq]
        # black
        elif piece <= 11: score -= tables[piece - 6, sq ^ 56]

        occupancy ^= lsb

    return -score if side_to_move == 1 else score