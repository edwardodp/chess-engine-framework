import board_tools as bt
from numba import njit, int64, int32, uint32
import numpy as np

# Piece type constants
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

# Color constants
WHITE = 0
BLACK = 1


"""
Piece square tables
"""

mg_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
], dtype=np.int16)

eg_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
], dtype=np.int16)

mg_knight_table = np.array([
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
], dtype=np.int16)

eg_knight_table = np.array([
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
], dtype=np.int16)

mg_bishop_table = np.array([
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
], dtype=np.int16)

eg_bishop_table = np.array([
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
], dtype=np.int16)

mg_rook_table = np.array([
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
], dtype=np.int16)

eg_rook_table = np.array([
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
], dtype=np.int16)

mg_queen_table = np.array([
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
], dtype=np.int16)

eg_queen_table = np.array([
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
], dtype=np.int16)

mg_king_table = np.array([
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
], dtype=np.int16)

eg_king_table = np.array([
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
], dtype=np.int16)

# array of midgame tables
mg_pesto_table = np.array([
    mg_pawn_table,
    mg_knight_table,
    mg_bishop_table,
    mg_rook_table,
    mg_queen_table,
    mg_king_table,
], dtype=np.int16)

# array of endgame tables
eg_pesto_table = np.array([
    eg_pawn_table,
    eg_knight_table,
    eg_bishop_table,
    eg_rook_table,
    eg_queen_table,
    eg_king_table,
], dtype=np.int16)
# pawn knight biship rook queen king 
MG_VALUE = np.array([82, 337, 365, 477, 1025, 0], dtype=np.int16)
EG_VALUE = np.array([94, 281, 297, 512,  936, 0], dtype=np.int16)

GAMEPHASE_INC = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)

# Initialize tables their layout is table[piece][square]
def init_tables():
    mg_table = np.zeros((12, 64), dtype=np.int32)
    eg_table = np.zeros((12, 64), dtype=np.int32)
        
    # For white pieces (indices 0-5)
    for piece_type in range(6):
        for sq in range(64):
            mg_table[piece_type][sq] = MG_VALUE[piece_type] + mg_pesto_table[piece_type][sq]
            eg_table[piece_type][sq] = EG_VALUE[piece_type] + eg_pesto_table[piece_type][sq]
    
    # For black pieces (indices 6-11), flip the square
    for piece_type in range(6):
        for sq in range(64):
            mg_table[piece_type + 6][sq] = MG_VALUE[piece_type] + mg_pesto_table[piece_type][sq ^ 56]
            eg_table[piece_type + 6][sq] = EG_VALUE[piece_type] + eg_pesto_table[piece_type][sq ^ 56]
    
    return mg_table, eg_table

# Pre-compute tables
MG_TABLE, EG_TABLE = init_tables()



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
    mg_white = 0
    mg_black = 0
    eg_white = 0
    eg_black = 0
    game_phase = 0
    
    score = 0


    def lsb_index(bb):
        idx = 0
        while (bb & 1) == 0:
            bb >>= 1
            idx += 1
        return idx
    
    
    # Evaluate white pieces (indices 0-5)
    for piece_type in range(6):
        # gets the bit board for the current piece type
        bitboard = board_pieces[piece_type]
        # Loops until the value stored in bitboard is 0, meaning there are no more pieces of that type to evaluate
        while bitboard:
            # Get the least significant bit position by AND the bitboard with its negation, finding its position using bit_length, and adjusting for 0-indexing.
            #sq = (bitboard & -bitboard).bit_length() - 1
            lsb = bitboard & -bitboard
            sq = lsb_index(lsb)
            mg_white += MG_TABLE[piece_type][sq]
            eg_white += EG_TABLE[piece_type][sq]
            game_phase += GAMEPHASE_INC[piece_type]
            # Clear the least significant bit
            bitboard &= bitboard - 1
        # Evaluate black pieces (indices 6-11)
        for piece_type in range(6):
            bitboard = board_pieces[piece_type + 6]
            while bitboard:
                # Get the least significant bit position
                #sq = (bitboard & -bitboard).bit_length() - 1
                lsb = bitboard & -bitboard
                sq = lsb_index(lsb)
                mg_black += MG_TABLE[piece_type + 6][sq]
                eg_black += EG_TABLE[piece_type + 6][sq]
                game_phase += GAMEPHASE_INC[piece_type]
                # Clear the least significant bit
                bitboard &= bitboard - 1 
    # Tapered eval
    #calculates the midgame and endgame score if the current turn is white
    if side_to_move == WHITE:
        mg_score = mg_white - mg_black
        eg_score = eg_white - eg_black
    #calculates the midgame and endgame score if the current turn is BLack
    else:
        mg_score = mg_black - mg_white
        eg_score = eg_black - eg_white
    
    
    mg_phase = game_phase
    if mg_phase > 24:
        mg_phase = 24
    eg_phase = 24 - mg_phase
    
    return (mg_score * mg_phase + eg_score * eg_phase) // 24
            
            
    """       
    # Example: Material value counting
    for sq in range(64):
        piece_id = bt.get_piece(board_pieces, sq) # Helper function in `board_tools` (use them)
        
        # Skip empty squares
        if piece_id == 0:
            continue

        # Add value based on piece ID (constants at top of file; use these too!)
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
    # If absolute score is +100 (White is winning) but it's Black's turn (side_to_move = 1), we must return -100 so the engine knows Black is in a bad position.
    if side_to_move == BLACK_TO_MOVE:
        return -score
    return score
    """
