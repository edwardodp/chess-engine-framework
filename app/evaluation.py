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

# Black Piece Values (Negative)
PAWN_BLACK   = -100
KNIGHT_BLACK = -320
BISHOP_BLACK = -330
ROOK_BLACK   = -500
QUEEN_BLACK  = -900
KING_BLACK   = -20000


@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    """
    Args:
        board_pieces: Array of 12 bitboards (piece locations)
        board_occupancy: Array of 3 bitboards (White, Black, All)
        side_to_move: 0 for White, 1 for Black
    
    Returns:
        int32: The score from the perspective of the side to move
               (Positive = Current player (side to move) is winning)
    """
    
    score = 0

    # Example: Basic Material Counting
    # This loop goes through every square (0-63) to check for pieces.
    
    for sq in range(64):
        piece_id = bt.get_piece(board_pieces, sq) # Helper function in `board_tools`
        
        # Skip empty squares
        if piece_id == 0:
            continue

        # Add value based on piece ID
        if piece_id == bt.WHITE_PAWN:      score += PAWN_WHITE
        elif piece_id == bt.WHITE_KNIGHT:  score += KNIGHT_WHITE
        elif piece_id == bt.WHITE_BISHOP:  score += BISHOP_WHITE
        elif piece_id == bt.WHITE_ROOK:    score += ROOK_WHITE
        elif piece_id == bt.WHITE_QUEEN:   score += QUEEN_WHITE
        elif piece_id == bt.WHITE_KING:    score += KING_WHITE
        elif piece_id == bt.BLACK_PAWN:    score += PAWN_BLACK
        elif piece_id == bt.BLACK_KNIGHT:  score += KNIGHT_BLACK
        elif piece_id == bt.BLACK_BISHOP:  score += BISHOP_BLACK
        elif piece_id == bt.BLACK_ROOK:    score += ROOK_BLACK
        elif piece_id == bt.BLACK_QUEEN:   score += QUEEN_BLACK
        elif piece_id == bt.BLACK_KING:    score += KING_BLACK


    # The engine requires the score to be relative to the player whose turn it is
    # If score is +100 (White is winning) but it's Black's turn (side_to_move = 1),
    # we must return -100 so the engine knows Black is in a bad position.
    
    if side_to_move == 1:
        return -score
    
    return score
