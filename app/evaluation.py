import board_tools as bt
from numba import njit, int64, int32, uint32


"""Piece ID Constants"""
# Black pieces will be the negative of these (e.g., -PAWN)
EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6

"""Piece Value Constants"""
PAWN_VAL   = 100
KNIGHT_VAL = 320
BISHOP_VAL = 330
ROOK_VAL   = 500
QUEEN_VAL  = 900
KING_VAL   = 20000

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, move_count):
    score = 0
    
    for sq in range(64):
        piece = bt.get_piece(board_pieces, sq)
        
        if piece == EMPTY:
            continue
            
        if piece > 0:
            if piece == PAWN:
                score += PAWN_VAL
            elif piece == KNIGHT:
                score += KNIGHT_VAL
            elif piece == BISHOP:
                score += BISHOP_VAL
            elif piece == ROOK:
                score += ROOK_VAL
            elif piece == QUEEN:
                score += QUEEN_VAL
            elif piece == KING:
                score += KING_VAL
        
        else:
            if piece == -PAWN:
                score -= PAWN_VAL
            elif piece == -KNIGHT:
                score -= KNIGHT_VAL
            elif piece == -BISHOP:
                score -= BISHOP_VAL
            elif piece == -ROOK:
                score -= ROOK_VAL
            elif piece == -QUEEN:
                score -= QUEEN_VAL
            elif piece == -KING:
                score -= KING_VAL

    return score
