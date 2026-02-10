import board_tools as bt
from numba import njit, int64, int32, uint32

"""
Piece Value Constants
"""
# White Piece Values
PAWN_WHITE   = 100
KNIGHT_WHITE = 320
BISHOP_WHITE = 330
ROOK_WHITE   = 500
QUEEN_WHITE  = 900
KING_WHITE   = 20000

PAWN_BLACK   = -100
KNIGHT_BLACK = -320
BISHOP_BLACK = -330
ROOK_BLACK   = -500
QUEEN_BLACK  = -900
KING_BLACK   = -20000

WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    
    score = 0

    # White Material
    score += bt.count_bits(board_pieces[0]) * PAWN_WHITE
    score += bt.count_bits(board_pieces[1]) * KNIGHT_WHITE
    score += bt.count_bits(board_pieces[2]) * BISHOP_WHITE
    score += bt.count_bits(board_pieces[3]) * ROOK_WHITE
    score += bt.count_bits(board_pieces[4]) * QUEEN_WHITE
    score += bt.count_bits(board_pieces[5]) * KING_WHITE

    # Black Material
    score += bt.count_bits(board_pieces[6]) * PAWN_BLACK
    score += bt.count_bits(board_pieces[7]) * KNIGHT_BLACK
    score += bt.count_bits(board_pieces[8]) * BISHOP_BLACK
    score += bt.count_bits(board_pieces[9]) * ROOK_BLACK
    score += bt.count_bits(board_pieces[10]) * QUEEN_BLACK
    score += bt.count_bits(board_pieces[11]) * KING_BLACK

    if side_to_move == BLACK_TO_MOVE:
        return -score
    
    return score
