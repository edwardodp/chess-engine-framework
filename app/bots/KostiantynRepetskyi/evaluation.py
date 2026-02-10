import board_tools as bt
import numpy as np
from numba import njit, int64, int32, uint32

CENTER_CONTROL = np.array([
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,5,5,5,5,0,0,
    0,0,5,10,10,5,0,0,
    0,0,5,10,10,5,0,0,
    0,0,5,5,5,5,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
], dtype=np.int32)

KING_CENTER_CONTROL = np.array([
    0,0,0,0,0,0,0,0,
    0,5,5,5,5,5,5,0,
    0,5,10,10,10,10,5,0,
    0,5,10,20,20,10,5,0,
    0,5,10,20,20,10,5,0,
    0,5,10,10,10,10,5,0,
    0,5,5,5,5,5,5,0,
    0,0,0,0,0,0,0,0,
], dtype=np.int32)


"""
Piece Value Constants
"""
# White Piece Values
PAWN_WHITE   = 100
KNIGHT_WHITE = 300
BISHOP_WHITE = 360
ROOK_WHITE   = 510
QUEEN_WHITE  = 875
KING_WHITE   = 20000

# Black Piece Values (Negative)
PAWN_BLACK   = -100
KNIGHT_BLACK = -300
BISHOP_BLACK = -360
ROOK_BLACK   = -510
QUEEN_BLACK  = -875
KING_BLACK   = -20000

"""
Turn Constants
"""
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1


@njit
def _handle_white_pawn(sq, white_pawn_count, white_pawn_min_rank, white_score, endgame_score_adjustment):
    f = (sq % 8) + 1
    r = sq // 8
    white_pawn_count[f] += 1

    if r < white_pawn_min_rank[f]:
        white_pawn_min_rank[f] = r

    white_score += CENTER_CONTROL[sq]
    white_score += PAWN_WHITE

    endgame_score_adjustment -= CENTER_CONTROL[sq]
    endgame_score_adjustment += r * 2

    return white_score, endgame_score_adjustment


@njit
def _knight_defends(board_pieces, sq, defended_pieces, colour):
    f = sq % 8
    r = sq // 8
   
    for i, j in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        new_f = f + i
        new_r = r + j
        if 0 <= new_f <= 7 and 0 <= new_r <= 7:
            idx = new_r * 8 + new_f
            piece = bt.get_piece(board_pieces, idx)
            if piece != 0 and piece * colour > 0:
                defended_pieces[idx] = 1


@njit
def _king_defends(board_pieces, sq, defended_pieces, colour):
    f = sq % 8
    r = sq // 8
    for i, j in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)):
        new_f = f + i
        new_r = r + j
        if 0 <= new_f <= 7 and 0 <= new_r <= 7:
            new_idx = new_r * 8 + new_f
            piece = bt.get_piece(board_pieces, new_idx)
            if piece != 0 and piece * colour > 0:
                defended_pieces[new_idx] = 1


@njit
def _pawn_defends(board_pieces, sq, defended_pieces, colour):
    
    f = sq % 8
    r = sq // 8
    if colour == 1:
        if r < 7:
            if f > 0:
                idx = sq + 7
                piece = bt.get_piece(board_pieces, idx)
                if piece != 0 and piece > 0:
                    defended_pieces[idx] = 1
            if f < 7:
                idx = sq + 9
                piece = bt.get_piece(board_pieces, idx)
                if piece != 0 and piece > 0:
                    defended_pieces[idx] = 1
    else:
        if r > 0:
            if f > 0:
                idx = sq - 9
                piece = bt.get_piece(board_pieces, idx)
                if piece != 0 and piece < 0:
                    defended_pieces[idx] = 1
            if f < 7:
                idx = sq - 7
                piece = bt.get_piece(board_pieces, idx)
                if piece != 0 and piece < 0:
                    defended_pieces[idx] = 1


@njit
def _rook_defends(board_pieces, sq, defended_pieces, colour):
    f0 = sq % 8
    r0 = sq // 8
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for k in range(1, 8):
            f = f0 + df * k
            r = r0 + dr * k
            if f < 0 or f > 7 or r < 0 or r > 7:
                break
            idx = r * 8 + f
            piece = bt.get_piece(board_pieces, idx)
            if piece != 0:
                if piece * colour > 0:
                    defended_pieces[idx] = 1
                break


@njit
def _handle_white_knight(board_pieces, sq, white_score, defended_pieces):
    #_knight_defends(board_pieces, sq, defended_pieces, 1)
    if sq < 8:
        white_score -= 10
    else:
        white_score += CENTER_CONTROL[sq]
    white_score += KNIGHT_WHITE
    return white_score


@njit
def _bishop_visible_empty_squares(board_pieces, sq, defended_pieces, colour):
    f = sq % 8
    r = sq // 8
    visible_empty_squares = 0

    for i, j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for k in range(1, 8):
            if f + i * k > 7 or f + i * k < 0 or r + j * k > 7 or r + j * k < 0:
                break

            idx = (r + j * k) * 8 + f + i * k
            piece = bt.get_piece(board_pieces, idx)
            if piece != 0:
                if piece * colour > 0:
                    defended_pieces[idx] = 1 # also check if the bishop defends this piece
                break
            visible_empty_squares += 1

    return visible_empty_squares


@njit
def _handle_white_bishop(board_pieces, sq, white_score, defended_pieces):
    if sq < 8:
        white_score -= 10

    visible_empty_squares = _bishop_visible_empty_squares(board_pieces, sq, defended_pieces, 1)
    white_score += visible_empty_squares * 2
    white_score += BISHOP_WHITE
    return white_score


@njit
def _handle_white_rook(board_pieces, sq, white_score, defended_pieces):
    #_rook_defends(board_pieces, sq, defended_pieces, 1)
    if 48 <= sq <= 55:
        white_score += 10

    if sq < 8:
        black_pawn_block = False
        white_pawn_block = False

        for i in range(sq + 8, 64, 8):
            if bt.get_piece(board_pieces, i) == bt.BLACK_PAWN:
                black_pawn_block = True
            elif bt.get_piece(board_pieces, i) == bt.WHITE_PAWN:
                white_pawn_block = True

        if not black_pawn_block:
            white_score += 10
            if not white_pawn_block:
                white_score += 10

    white_score += ROOK_WHITE
    return white_score


@njit
def _handle_white_queen(board_pieces, sq, white_score, defended_pieces):
    
    #_rook_defends(board_pieces, sq, defended_pieces, 1)
    #_bishop_visible_empty_squares(board_pieces, sq, defended_pieces, 1)
    return white_score + QUEEN_WHITE


@njit
def _handle_white_king(board_pieces, sq, white_score, endgame_score_adjustment, defended_pieces):
    #_king_defends(board_pieces, sq, defended_pieces, 1)
    white_score += KING_WHITE
    endgame_score_adjustment += KING_CENTER_CONTROL[sq]

    if sq == 6 and bt.get_piece(board_pieces, 7) != bt.WHITE_ROOK:
        white_score += 15
        endgame_score_adjustment -= 15
        
    return white_score, endgame_score_adjustment


@njit
def _handle_black_pawn(sq, black_pawn_count, black_pawn_max_rank, black_score, endgame_score_adjustment):
    f = (sq % 8) + 1
    r = sq // 8
    black_pawn_count[f] += 1

    if r > black_pawn_max_rank[f]:
        black_pawn_max_rank[f] = r

    black_score -= CENTER_CONTROL[sq]
    black_score += PAWN_BLACK

    endgame_score_adjustment += CENTER_CONTROL[sq]
    endgame_score_adjustment -= (7 - r) * 2

    return black_score, endgame_score_adjustment


@njit
def _handle_black_knight(board_pieces, sq, black_score, defended_pieces):
    #_knight_defends(board_pieces, sq, defended_pieces, -1)
    if sq > 55:
        black_score += 10
    else:
        black_score -= CENTER_CONTROL[sq]
    black_score += KNIGHT_BLACK
    return black_score


@njit
def _handle_black_bishop(board_pieces, sq, black_score, defended_pieces):
    if sq > 55:
        black_score += 10

    visible_empty_squares = _bishop_visible_empty_squares(board_pieces, sq, defended_pieces, -1)
    black_score -= visible_empty_squares * 2
    black_score += BISHOP_BLACK
    return black_score


@njit
def _handle_black_rook(board_pieces, sq, black_score, defended_pieces):
    #_rook_defends(board_pieces, sq, defended_pieces, -1)
    if 8 <= sq <= 15:
        black_score -= 20

    if sq > 55:
        black_pawn_block = False
        white_pawn_block = False

        for i in range(sq - 8, -1, -8):
            if bt.get_piece(board_pieces, i) == bt.BLACK_PAWN:
                black_pawn_block = True
            elif bt.get_piece(board_pieces, i) == bt.WHITE_PAWN:
                white_pawn_block = True

        if not white_pawn_block:
            black_score -= 10
            if not black_pawn_block:
                black_score -= 10

    black_score += ROOK_BLACK
    return black_score


@njit
def _handle_black_queen(board_pieces, sq, black_score, defended_pieces):
    #_rook_defends(board_pieces, sq, defended_pieces, -1)
    _ = _bishop_visible_empty_squares(board_pieces, sq, defended_pieces, -1)
    return black_score + QUEEN_BLACK


@njit
def _handle_black_king(board_pieces, sq, black_score, endgame_score_adjustment, defended_pieces):
    #_king_defends(board_pieces, sq, defended_pieces, -1)
    endgame_score_adjustment -= KING_CENTER_CONTROL[sq]
    black_score += KING_BLACK

    if sq == 62 and bt.get_piece(board_pieces, 63) != bt.BLACK_ROOK:
        black_score -= 15
        endgame_score_adjustment += 15

    return black_score, endgame_score_adjustment


@njit
def _pawn_structure_adjustment(white_pawn_count, black_pawn_count, white_pawn_min_rank, black_pawn_max_rank):
    score_adjustment = 0

    current_min = 0
    right_min = white_pawn_min_rank[1]  

    for i in range(1, 9):
        n = white_pawn_count[i]

        if n != 0:
            left_min = current_min
            current_min = right_min
            right_min = white_pawn_min_rank[i + 1]

            if white_pawn_count[i + 1] == 0 and white_pawn_count[i - 1] == 0:  # Isolated pawn
                score_adjustment -= 10

            elif current_min < left_min and current_min < right_min:  # Backward pawn
                score_adjustment -= 5

            if n >= 2:  # Stacked pawns
                score_adjustment -= 10 * (n - 1)

            if black_pawn_count[i + 1] == 0 and black_pawn_count[i - 1] == 0 and black_pawn_count[i] == 0:  # Passed pawn
                score_adjustment += 10

    current_max = 0
    right_max = black_pawn_max_rank[1]  

    for i in range(1, 9):
        n = black_pawn_count[i]

        if n != 0:
            left_max = current_max
            current_max = right_max
            right_max = black_pawn_max_rank[i + 1]

            if black_pawn_count[i + 1] == 0 and black_pawn_count[i - 1] == 0:  # Isolated pawn
                score_adjustment += 10

            elif current_max > left_max and current_max > right_max:  # Backward pawn
                score_adjustment += 5

            if n >= 2:  # Stacked pawns
                score_adjustment += 10 * (n - 1)

            if white_pawn_count[i + 1] == 0 and white_pawn_count[i - 1] == 0 and white_pawn_count[i] == 0:  # Passed pawn
                score_adjustment -= 10

    return score_adjustment


@njit
def _loose_pieces_adjustment(board_pieces, loose_pieces):

    score_adjustment = 0

    for sq in range(64):
        piece = bt.get_piece(board_pieces, sq)
        if piece == bt.WHITE_BISHOP or piece == bt.WHITE_KNIGHT or piece == bt.WHITE_ROOK:
            if loose_pieces[sq] == 0:
                score_adjustment -= 10
        elif piece == bt.WHITE_PAWN:
            if loose_pieces[sq] == 0:
                score_adjustment -= 5
        elif piece == bt.BLACK_BISHOP or piece == bt.BLACK_KNIGHT or piece == bt.BLACK_ROOK:
            if loose_pieces[sq] == 0:
                score_adjustment += 10
        elif piece == bt.BLACK_PAWN:
            if loose_pieces[sq] == 0:
                score_adjustment += 5

    return score_adjustment


@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    """
    Args:
        board_pieces: Array of 12 bitboards (piece locations) – Do not modify
        board_occupancy: Array of 3 bitboards (White, Black, All) – Do not modify
        side_to_move: 0 for White, 1 for Black
    
    Returns:
        int32: The score from the perspective of the side to move
               (Positive = Current player (side to move) is winning)
    """
    
    
    black_score = 0
    white_score = 0

    # The right most and left most columns act as a buffer to prevent index out of bounds errors
    # Numba can't use Python lists in nopython mode. These arrays keep the same
    # information your structure logic uses later:
    # - counts for len(...)
    # - min rank per file for min(...)
    # - max rank per file for max(...)
    white_pawn_count = np.zeros(10, dtype=np.int32)
    black_pawn_count = np.zeros(10, dtype=np.int32)
    white_pawn_min_rank = np.full(10, 99, dtype=np.int32)   
    black_pawn_max_rank = np.full(10, -1, dtype=np.int32)   

    endgame_score_adjustment = 0

    defended_pieces = np.zeros(64, dtype=np.int8)

    
    for sq in range(64):
        piece_id = bt.get_piece(board_pieces, sq) 
        
        # Skip empty squares
        if piece_id == 0:
            continue







        if piece_id == bt.WHITE_PAWN: 
            #_pawn_defends(board_pieces, sq, defended_pieces, 1)
            white_score, endgame_score_adjustment = _handle_white_pawn(
                sq, white_pawn_count, white_pawn_min_rank, white_score, endgame_score_adjustment
            )






        elif piece_id == bt.WHITE_KNIGHT:  
            white_score = _handle_white_knight(board_pieces, sq, white_score, defended_pieces)






        elif piece_id == bt.WHITE_BISHOP:  
            white_score = _handle_white_bishop(board_pieces, sq, white_score, defended_pieces)





        elif piece_id == bt.WHITE_ROOK:  
            white_score = _handle_white_rook(board_pieces, sq, white_score, defended_pieces)






        elif piece_id == bt.WHITE_QUEEN:
            white_score = _handle_white_queen(board_pieces, sq, white_score, defended_pieces)






        elif piece_id == bt.WHITE_KING:        
            white_score, endgame_score_adjustment = _handle_white_king(
                board_pieces, sq, white_score, endgame_score_adjustment, defended_pieces
            )





        elif piece_id == bt.BLACK_PAWN:        
            #_pawn_defends(board_pieces, sq, defended_pieces, -1)
            black_score, endgame_score_adjustment = _handle_black_pawn(
                sq, black_pawn_count, black_pawn_max_rank, black_score, endgame_score_adjustment
            )






        elif piece_id == bt.BLACK_KNIGHT:  
            black_score = _handle_black_knight(board_pieces, sq, black_score, defended_pieces)






        elif piece_id == bt.BLACK_BISHOP:  
            black_score = _handle_black_bishop(board_pieces, sq, black_score, defended_pieces)
        





        elif piece_id == bt.BLACK_ROOK:    
            black_score = _handle_black_rook(board_pieces, sq, black_score, defended_pieces)






        elif piece_id == bt.BLACK_QUEEN:
            black_score = _handle_black_queen(board_pieces, sq, black_score, defended_pieces)






        elif piece_id == bt.BLACK_KING:    
            black_score, endgame_score_adjustment = _handle_black_king(
                board_pieces, sq, black_score, endgame_score_adjustment, defended_pieces
            )





    score = white_score + black_score


    if -1000 + KING_BLACK < black_score and white_score < 1000 + KING_WHITE:
        score += endgame_score_adjustment


    score += _pawn_structure_adjustment(white_pawn_count, black_pawn_count, white_pawn_min_rank, black_pawn_max_rank)

    #score += _loose_pieces_adjustment(board_pieces, defended_pieces)
        
        


    # The engine requires the score to be relative to the player whose turn it is
    # If absolute score is +100 (White is winning) but it's Black's turn (side_to_move = 1), we must return -100 so the engine knows Black is in a bad position.
    if side_to_move == BLACK_TO_MOVE:
        return -score
    return score
    



    
