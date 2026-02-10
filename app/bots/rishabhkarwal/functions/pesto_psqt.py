"""Piece-square table implementation using PeSTO values"""

import board_tools as bt
from numba import njit, int64, int32, uint32, uint64
import numpy as np

# piece values
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 82, 337, 365, 477, 1025, 20000

# piece-square tables
# NOTE: defined from white's perspective

_p = (
    0,   0,   0,   0,   0,   0,   0,   0,
   98, 134,  61,  95,  68, 126,  34, -11,
   -6,   7,  26,  31,  65,  56,  25, -20,
  -14,  13,   6,  21,  23,  12,  17, -23,
  -27,  -2,  -5,  12,  17,   6,  10, -25,
  -26,  -4,  -4, -10,   3,   3,  33, -12,
  -35,  -1, -20, -23, -15,  24,  38, -22,
    0,   0,   0,   0,   0,   0,   0,   0,
)

_n = (
 -167, -89, -34, -49,  61, -97, -15, -107,
  -73, -41,  72,  36,  23,  62,   7,  -17,
  -47,  60,  37,  65,  84, 129,  73,   44,
   -9,  17,  19,  53,  37,  69,  18,   22,
  -13,   4,  16,  13,  28,  19,  21,   -8,
  -23,  -9,  12,  10,  19,  17,  25,  -16,
  -29, -53, -12,  -3,  -1,  18, -14,  -19,
 -105, -21, -58, -33, -17, -28, -19, -23,
)

_b = (
  -29,   4, -82, -37, -25, -42,   7,  -8,
  -26,  16, -18, -13,  30,  59,  18, -47,
  -16,  37,  43,  40,  35,  50,  37,  -2,
   -4,   5,  19,  50,  37,  37,   7,  -2,
   -6,  13,  13,  26,  34,  12,  10,   4,
    0,  15,  15,  15,  14,  27,  18,  10,
    4,  15,  16,   0,   7,  21,  33,   1,
  -33,  -3, -14, -21, -13, -12, -39, -21,
)

_r = (
   32,  42,  32,  51,  63,   9,  31,  43,
   27,  32,  58,  62,  80,  67,  26,  44,
   -5,  19,  26,  36,  17,  45,  61,  16,
  -24, -11,   7,  26,  24,  35,  -8, -20,
  -36, -26, -12,  -1,   9,  -7,   6, -23,
  -45, -25, -16, -17,   3,   0,  -5, -33,
  -44, -16, -20,  -9,  -1,  11,  -6, -71,
  -19, -13,   1,  17,  16,   7, -37, -26,
)

_q = (
  -28,   0,  29,  12,  59,  44,  43,  45,
  -24, -39,  -5,   1, -16,  57,  28,  54,
  -13, -17,   7,   8,  29,  56,  47,  57,
  -27, -27, -16, -16,  -1,  17,  -2,   1,
   -9, -26,  -9, -10,  -2,  -4,   3,  -3,
  -14,   2, -11,  -2,  -5,   2,  14,   5,
  -35,  -8,  11,   2,   8,  15,  -3,   1,
   -1, -18,  -9,  10, -15, -25, -31, -50,
)

_k = (
  -65,  23,  16, -15, -56, -34,   2,  13,
   29,  -1, -20,  -7,  -8,  -4, -38, -29,
   -9,  24,   2, -16, -20,   6,  22, -22,
  -17, -20, -12, -27, -30, -25, -14, -36,
  -49,  -1, -27, -39, -46, -44, -33, -51,
  -14,   3, -14, -13, -45, -28,  -4, -41,
  -27, -27, -16, -17, -25, -18,  -3, -19,
  -74, -35, -18, -18, -11,  15,   4, -17,
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