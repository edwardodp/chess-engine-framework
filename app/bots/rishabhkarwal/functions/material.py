""" Simple material count of pieces on board """

import board_tools as bt
from numba import njit, int64, int32, uint32

# piece values
PAWN   = 100
KNIGHT = 320
BISHOP = 330
ROOK   = 500
QUEEN  = 900
KING   = 20000

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    score = 0

    for sq in range(64):
        piece = bt.get_piece(board_pieces, sq)
        
        # skip empty squares
        if piece == 0: continue

        if piece == bt.WHITE_PAWN:     score += PAWN
        elif piece == bt.WHITE_KNIGHT: score += KNIGHT
        elif piece == bt.WHITE_BISHOP: score += BISHOP
        elif piece == bt.WHITE_ROOK:   score += ROOK
        elif piece == bt.WHITE_QUEEN:  score += QUEEN
        elif piece == bt.WHITE_KING:   score += KING

        elif piece == bt.BLACK_PAWN:   score -= PAWN
        elif piece == bt.BLACK_KNIGHT: score -= KNIGHT
        elif piece == bt.BLACK_BISHOP: score -= BISHOP
        elif piece == bt.BLACK_ROOK:   score -= ROOK
        elif piece == bt.BLACK_QUEEN:  score -= QUEEN
        elif piece == bt.BLACK_KING:   score -= KING

    if side_to_move == 1: return -score # black
    return score