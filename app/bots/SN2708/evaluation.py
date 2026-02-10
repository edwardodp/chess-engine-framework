import board_tools as bt
from numba import njit, int64, int32, uint32
import numpy as np

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

# Phase Piece Values
PHASE_MAX = 24
PHASE_KNIGHT = 1
PHASE_BISHOP = 1
PHASE_ROOK   = 2
PHASE_QUEEN  = 4

# Piece Attack Values
PAWN_ATTACK_WEIGHT    = 6
KNIGHT_ATTACK_WEIGHT  = 10
BISHOP_ATTACK_WEIGHT  = 8
ROOK_ATTACK_WEIGHT    = 12
QUEEN_ATTACK_WEIGHT   = 14

KING_SAFETY_SCALE = 256  # higher = softer penalty


"""
Turn Constants
"""
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1




# File masks (a..h)
FILE_MASKS = (
    np.uint64(0x0101010101010101),
    np.uint64(0x0202020202020202),
    np.uint64(0x0404040404040404),
    np.uint64(0x0808080808080808),
    np.uint64(0x1010101010101010),
    np.uint64(0x2020202020202020),
    np.uint64(0x4040404040404040),
    np.uint64(0x8080808080808080),
    )
# Adjacent files mask for each file (for isolated pawn test)
ADJ_FILE_MASKS = (
    FILE_MASKS[1],
    FILE_MASKS[0] | FILE_MASKS[2],
    FILE_MASKS[1] | FILE_MASKS[3],
    FILE_MASKS[2] | FILE_MASKS[4],
    FILE_MASKS[3] | FILE_MASKS[5],
    FILE_MASKS[4] | FILE_MASKS[6],
    FILE_MASKS[5] | FILE_MASKS[7],
    FILE_MASKS[6],
)

# Passed pawn masks:
PASSED_MASK_WHITE = np.zeros(64, dtype=np.uint64)
PASSED_MASK_BLACK = np.zeros(64, dtype=np.uint64)

for sq in range(64):
    file = sq & 7
    rank = sq >> 3

    mW = 0
    mB = 0

    # White passed pawn: enemy pawns ahead (higher ranks) on files f-1,f,f+1
    for i in (-1, 0, 1):
        checkingFile = file + i
        if 0 <= checkingFile <= 7:
            for j in range(rank + 1, 8):
                mW |= (1 << (j * 8 + checkingFile))

    # Black passed pawn: enemy pawns ahead for black (lower ranks) on files f-1,f,f+1
    for i in (-1, 0, 1):
        checkingFile = file + i
        if 0 <= checkingFile <= 7:
            for j in range(rank - 1, -1, -1):
                mB |= (1 << (j * 8 + checkingFile))

    PASSED_MASK_WHITE[sq] = np.uint64(mW)
    PASSED_MASK_BLACK[sq] = np.uint64(mB)





PAWN_MG_WHITE = np.array([
    # Rank 1 (a1..h1)
     0, 0, 0, 0, 0, 0, 0, 0,

    # Rank 2 (a2..h2)
     3,  3, 10, 19, 16, 19,  7, -5,

    # Rank 3
    -9, -15, 11, 15, 32, 22,  5, -22,

    # Rank 4
    -8, -23,  6, 20, 40, 17,  4, -12,

    # Rank 5
    13,  0, -13,  1, 11, -2, -13,  5,

    # Rank 6
    -5, -12, -7, 22, -8, -5, -15, -18,

    # Rank 7
    -7,  7, -3, -13,  5, -16, 10, -8,

    # Rank 8 (a8..h8)
     0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)

PAWN_MG_BLACK = PAWN_MG_WHITE[::-1]

PAWN_EG_WHITE = np.array([
    # Rank 1
     0, 0, 0, 0, 0, 0, 0, 0,

    # Rank 2
   -10,  -6, 10,  0, 14,  7, -5, -19,

    # Rank 3
   -10, -10,-10,  4,  4,  3, -6,  -4,

    # Rank 4
     6,  -2, -8, -4,-13,-12,-10,  -9,

    # Rank 5
     9,   4,  3,-12,-12, -6, 13,   8,

    # Rank 6
    28,  20, 21, 28, 30,  7,  6,  13,

    # Rank 7
     0, -11, 12, 21, 25, 19,  4,   7,

    # Rank 8
     0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)

PAWN_EG_BLACK = PAWN_EG_WHITE[::-1]

KNIGHT_MG_WHITE = np.array([
   -175,  -92,  -74,  -73,  -73,  -74,  -92, -175,
    -77,  -41,  -27,  -15,  -15,  -27,  -41,  -77,
    -61,  -17,    6,   12,   12,    6,  -17,  -61,
    -35,    8,   40,   49,   49,   40,    8,  -35,
    -34,   13,   44,   51,   51,   44,   13,  -34,
     -9,   22,   58,   53,   53,   58,   22,   -9,
    -67,  -27,    4,   37,   37,    4,  -27,  -67,
   -201,  -83,  -56,  -26,  -26,  -56,  -83, -201
], dtype=np.int32)

KNIGHT_MG_BLACK = KNIGHT_MG_WHITE[::-1]


KNIGHT_EG_WHITE = np.array([
    -96,  -65,  -49,  -21,  -21,  -49,  -65,  -96,
    -67,  -54,  -18,    8,    8,  -18,  -54,  -67,
    -40,  -27,   -8,   29,   29,   -8,  -27,  -40,
    -35,   -2,   13,   28,   28,   13,   -2,  -35,
    -45,  -16,    9,   39,   39,    9,  -16,  -45,
    -51,  -44,  -16,   17,   17,  -16,  -44,  -51,
    -69,  -50,  -51,   12,   12,  -51,  -50,  -69,
   -100,  -88,  -56,  -17,  -17,  -56,  -88, -100
], dtype=np.int32)

KNIGHT_EG_BLACK = KNIGHT_EG_WHITE[::-1]

BISHOP_MG_WHITE = np.array([
   -53,  -5,  -8, -23, -23,  -8,  -5, -53,
   -15,   8,  19,   4,   4,  19,   8, -15,
    -7,  21,  -5,  17,  17,  -5,  21,  -7,
    -5,  11,  25,  39,  39,  25,  11,  -5,
   -12,  29,  22,  31,  31,  22,  29, -12,
   -16,   6,   1,  11,  11,   1,   6, -16,
   -17, -14,   5,   0,   0,   5, -14, -17,
   -48,   1, -14, -23, -23, -14,   1, -48
], dtype=np.int32)

BISHOP_MG_BLACK = BISHOP_MG_WHITE[::-1]

BISHOP_EG_WHITE = np.array([
   -57, -30, -37, -12, -12, -37, -30, -57,
   -37, -13, -17,   1,   1, -17, -13, -37,
   -16,  -1,  -2,  10,  10,  -2,  -1, -16,
   -20,  -6,   0,  17,  17,   0,  -6, -20,
   -17,  -1, -14,  15,  15, -14,  -1, -17,
   -30,   6,   4,   6,   6,   4,   6, -30,
   -31, -20,  -1,   1,   1,  -1, -20, -31,
   -46, -42, -37, -24, -24, -37, -42, -46
], dtype=np.int32)

BISHOP_EG_BLACK = BISHOP_EG_WHITE[::-1]

ROOK_MG_WHITE = np.array([
   -31, -20, -14,  -5,  -5, -14, -20, -31,
   -21, -13,  -8,   6,   6,  -8, -13, -21,
   -25, -11,  -1,   3,   3,  -1, -11, -25,
   -13,  -5,  -4,  -6,  -6,  -4,  -5, -13,
   -27, -15,  -4,   3,   3,  -4, -15, -27,
   -22,  -2,   6,  12,  12,   6,  -2, -22,
    -2,  12,  16,  18,  18,  16,  12,  -2,
   -17, -19,  -1,   9,   9,  -1, -19, -17
], dtype=np.int32)

ROOK_MG_BLACK = ROOK_MG_WHITE[::-1]

ROOK_EG_WHITE = np.array([
   -9, -13, -10,  -9,  -9, -10, -13,  -9,
  -12,  -9,  -1,  -2,  -2,  -1,  -9, -12,
   6,  -8,  -2,  -6,  -6,  -2,  -8,   6,
  -6,   1,  -9,   7,   7,  -9,   1,  -6,
  -5,   8,   7,  -6,  -6,   7,   8,  -5,
   6,   1,  -7,  12,  12,  -7,   1,   6,
  16,  20,  16,  20,  20,  16,  20,  16,
  18,   0,  19,  13,  13,  19,   0,  18
], dtype=np.int32)

ROOK_EG_BLACK = ROOK_EG_WHITE[::-1]

QUEEN_MG_WHITE = np.array([
     3,  -5,  -5,   4,   4,  -5,  -5,   3,
    -3,   5,   8,  12,  12,   8,   5,  -3,
    -3,   6,  13,   7,   7,  13,   6,  -3,
     4,   5,   9,   8,   8,   9,   5,   4,
     0,  14,  12,   5,   5,  12,  14,   0,
    -4,  10,   6,   8,   8,   6,  10,  -4,
    -5,   6,  10,   8,   8,  10,   6,  -5,
    -2,  -2,   1,  -2,  -2,   1,  -2,  -2
], dtype=np.int32)

QUEEN_MG_BLACK = QUEEN_MG_WHITE[::-1]

QUEEN_EG_WHITE = np.array([
   -69, -57, -47, -26, -26, -47, -57, -69,
   -55, -31, -22,  -4,  -4, -22, -31, -55,
   -39, -18,  -9,   3,   3,  -9, -18, -39,
   -23,  -3,  13,  24,  24,  13,  -3, -23,
   -29,  -6,   9,  21,  21,   9,  -6, -29,
   -38, -18, -12,   1,   1, -12, -18, -38,
   -50, -27, -24,  -8,  -8, -24, -27, -50,
   -75, -52, -43, -36, -36, -43, -52, -75
], dtype=np.int32)

QUEEN_EG_BLACK = QUEEN_EG_WHITE[::-1]

KING_MG_WHITE = np.array([
    271, 327, 271, 198, 198, 271, 327, 271,
    278, 303, 234, 179, 179, 234, 303, 278,
    195, 258, 169, 120, 120, 169, 258, 195,
    164, 190, 138,  98,  98, 138, 190, 164,
    154, 179, 105,  70,  70, 105, 179, 154,
    123, 145,  81,  31,  31,  81, 145, 123,
     88, 120,  65,  33,  33,  65, 120,  88,
     59,  89,  45,  -1,  -1,  45,  89,  59
], dtype=np.int32)

KING_MG_BLACK = KING_MG_WHITE[::-1]

KING_EG_WHITE = np.array([
      1,  45,  85,  76,  76,  85,  45,   1,
     53, 100, 133, 135, 135, 133, 100,  53,
     88, 130, 169, 175, 175, 169, 130,  88,
    103, 156, 172, 172, 172, 172, 156, 103,
     96, 166, 199, 199, 199, 199, 166,  96,
    92, 172, 184, 191, 191, 184, 172,  92,
     47, 121, 116, 131, 131, 116, 121,  47,
     11,  59,  73,  78,  78,  73,  59,  11
], dtype=np.int32)

KING_EG_BLACK = KING_EG_WHITE[::-1]

ROOK_MG_OPENFILE = 47
ROOK_MG_SEMIOPENFILE = 21

ROOK_EG_OPENFILE = 25
ROOK_EG_SEMIOPENFILE = 4


MobilityBonus_Knight = np.array([ (-62, -81), (-53, -56), (-12, -30), (-4, -14),(3, 8), (13, 15), (22, 23), (28, 27), (33, 33)], dtype=np.int32)
MobilityBonus_Bishop = np.array([
    (-48, -59), (-20, -23), (16, -3), (26, 13),
    (38, 24), (51, 42), (55, 54), (63, 57),
    (63, 65), (68, 73), (81, 78), (81, 86),
    (91, 88), (98, 97)], dtype=np.int32)
MobilityBonus_Rook = np.array([
    (-58, -76), (-27, -18), (-15, 28), (-10, 55),
    (-5, 69), (-2, 82), (9, 112), (16, 118),
    (30, 132), (29, 142), (32, 155), (38, 165),
    (46, 166), (48, 169), (58, 171)], dtype=np.int32)
MobilityBonus_Queen = np.array([
    (-39, -36), (-21, -15), (3, 8), (3, 18),
    (14, 34), (22, 54), (28, 61), (41, 73),
    (43, 79), (48, 92), (56, 94), (60, 104),
    (60, 113), (66, 120), (67, 123), (70, 126),
    (71, 133), (73, 136), (79, 140), (88, 143),
    (88, 148), (99, 166), (102, 170), (102, 175),
    (106, 184), (109, 191), (113, 206), (116, 212)], dtype=np.int32)


# Pawn structure constants (MG, EG)
BACKWARD       = np.array([9, 24], dtype=np.int32)
DOUBLED        = np.array([11, 56], dtype=np.int32)
ISOLATED       = np.array([5, 15], dtype=np.int32)
PASSED         = np.array([20, 60], dtype=np.int32)  



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
    
    mg_score = 0
    eg_score = 0
    score = 0
    whiteMaterial = 0
    blackMaterial = 0
    totalMaterial = 0
    phase = 0
    WhitePiece = 0
    BlackPiece = 0
    
    whitePawns = np.uint64(board_pieces[0])
    blackPawns = np.uint64(board_pieces[6])

    # Example: Material value counting
    for sq in range(64):
        piece_id = bt.get_piece(board_pieces, sq) # Helper function in `board_tools` (use them)

        WhitePiece = 0
        BlackPiece = 0
        mobility = 0
        file = sq % 8
        rank = sq // 8

        
        # Skip empty squares
        if piece_id == 0:
            continue

        # Add value based on piece ID (constants at top of file; use these too!)
        if piece_id == bt.WHITE_PAWN:      
            whiteMaterial += PAWN_WHITE 
            mg_score += PAWN_MG_WHITE[sq]
            eg_score += PAWN_EG_WHITE[sq]

            # -------------------------
            # Fast pawn structure (bitboards)
            # -------------------------
            bit = (np.uint64(1) << sq)

            # Isolated: no friendly pawn on adjacent files
            if (whitePawns & ADJ_FILE_MASKS[file]) == 0:
                mg_score -= ISOLATED[0]
                eg_score -= ISOLATED[1]

            # Doubled: there exists another friendly pawn on same file
            if (whitePawns & (FILE_MASKS[file] & ~bit)) != 0:
                mg_score -= DOUBLED[0]
                eg_score -= DOUBLED[1]

            # Passed: no enemy pawn ahead on same/adjacent files
            if (blackPawns & PASSED_MASK_WHITE[sq]) == 0:
                mg_score += PASSED[0]
                eg_score += PASSED[1]



        elif piece_id == bt.WHITE_KNIGHT:  
            whiteMaterial += KNIGHT_WHITE
            mg_score += KNIGHT_MG_WHITE[sq]
            eg_score += KNIGHT_EG_WHITE[sq]
            phase += PHASE_KNIGHT
            for offset in (17, 15, 10, 6, -17, -15, -10, -6):
                to_sq = sq + offset
                if 0 <= to_sq < 64:
                    to_file = to_sq % 8
                    if abs(to_file - file) <= 2:
                        # not occupied by white piece
                        if bt.check_square(board_occupancy, to_sq, 0) == 0:
                            mobility += 1

            mg_score += MobilityBonus_Knight[mobility][0]
            eg_score += MobilityBonus_Knight[mobility][1]
            


        elif piece_id == bt.WHITE_BISHOP:  
            whiteMaterial += BISHOP_WHITE
            mg_score += BISHOP_MG_WHITE[sq]
            eg_score += BISHOP_EG_WHITE[sq]
            phase += PHASE_BISHOP

            # Diagonal directions: NE(+9), NW(+7), SE(-7), SW(-9)
            directions = (9, 7, -7, -9)
            for i in directions:
                duplicate_sq = sq
                while True:
                    duplicate_sq += i
                    if not (0 <= duplicate_sq < 64):
                        break
                    to_rank = duplicate_sq // 8
                    to_file = duplicate_sq % 8


                    if abs(to_file - file) != abs(to_rank - rank):
                        break

                    # Stop if blocked by white piece
                    if bt.check_square(board_occupancy, duplicate_sq, 0) != 0:
                        break

                    mobility += 1

                    # Stop if blocked by any piece (cannot jump over)
                    if bt.check_square(board_occupancy, duplicate_sq, 2) != 0:
                        break

            if mobility >= len(MobilityBonus_Bishop):
                mobility = len(MobilityBonus_Bishop) - 1

            mg_score += MobilityBonus_Bishop[mobility][0]
            eg_score += MobilityBonus_Bishop[mobility][1]

     

        elif piece_id == bt.WHITE_ROOK:    
            whiteMaterial += ROOK_WHITE
            mg_score += ROOK_MG_WHITE[sq]
            eg_score += ROOK_EG_WHITE[sq]
            phase += PHASE_ROOK
            for i in range(8):
                if (whitePawns >> (sq%8 + 8*i)) & 1:
                    WhitePiece += 1
                elif (blackPawns >> (sq%8 + 8*i)) & 1:
                    BlackPiece += 1
            if WhitePiece == 0 and BlackPiece ==0:
                mg_score += ROOK_MG_OPENFILE
                eg_score += ROOK_EG_OPENFILE
            elif WhitePiece == 0 and BlackPiece > 0:
                mg_score += ROOK_MG_SEMIOPENFILE
                eg_score += ROOK_EG_SEMIOPENFILE


        elif piece_id == bt.WHITE_QUEEN:   
            whiteMaterial += QUEEN_WHITE
            mg_score += QUEEN_MG_WHITE[sq]
            eg_score += QUEEN_EG_WHITE[sq]
            phase += PHASE_QUEEN



        elif piece_id == bt.WHITE_KING:    
            whiteMaterial += KING_WHITE
            mg_score += KING_MG_WHITE[sq]
            eg_score += KING_EG_WHITE[sq]


        elif piece_id == bt.BLACK_PAWN:    
            blackMaterial += PAWN_BLACK
            mg_score -= PAWN_MG_BLACK[sq]
            eg_score -= PAWN_EG_BLACK[sq]

            # -------------------------
            # Fast pawn structure (bitboards)
            # -------------------------
            bit = (np.uint64(1) << sq)

            # Isolated: no friendly pawn on adjacent files
            if (blackPawns & ADJ_FILE_MASKS[file]) == 0:
                mg_score += ISOLATED[0]
                eg_score += ISOLATED[1]

            # Doubled: another black pawn exists on same file
            if (blackPawns & (FILE_MASKS[file] & ~bit)) != 0:
                mg_score += DOUBLED[0]
                eg_score += DOUBLED[1]

            # Passed: for black, check squares "ahead" toward rank 1
            if (whitePawns & PASSED_MASK_BLACK[sq]) == 0:
                mg_score -= PASSED[0]
                eg_score -= PASSED[1]





        elif piece_id == bt.BLACK_KNIGHT:  
            blackMaterial += KNIGHT_BLACK
            mg_score -= KNIGHT_MG_BLACK[sq]
            eg_score -= KNIGHT_EG_BLACK[sq]
            phase += PHASE_KNIGHT
            mobility = 0

            for offset in (17, 15, 10, 6, -17, -15, -10, -6):
                to_sq = sq + offset
                if 0 <= to_sq < 64:
                    to_file = to_sq % 8
                    if abs(to_file - file) <= 2:
                        # not occupied by black piece
                        if bt.check_square(board_occupancy, to_sq, 1) == 0:
                            mobility += 1

            mg_score -= MobilityBonus_Knight[mobility][0]
            eg_score -= MobilityBonus_Knight[mobility][1]

        elif piece_id == bt.BLACK_BISHOP:  
            blackMaterial += BISHOP_BLACK
            mg_score -= BISHOP_MG_BLACK[sq]
            eg_score -= BISHOP_EG_BLACK[sq]
            phase += PHASE_BISHOP

            # Diagonal directions: NE(+9), NW(+7), SE(-7), SW(-9)
            directions = (9, 7, -7, -9)
            for i in directions:
                duplicate_sq = sq
                while True:
                    duplicate_sq += i
                    if not (0 <= duplicate_sq < 64):
                        break
                    to_rank = duplicate_sq // 8
                    to_file = duplicate_sq % 8

                    # Check we didn’t wrap around the board
                    if abs(to_file - file) != abs(to_rank - rank):
                        break

                    # Stop if blocked by black piece
                    if bt.check_square(board_occupancy, duplicate_sq, 1) != 0:
                        break

                    mobility += 1

                    # Stop if blocked by any piece (cannot jump over)
                    if bt.check_square(board_occupancy, duplicate_sq, 2) != 0:
                        break

            if mobility >= len(MobilityBonus_Bishop):
                mobility = len(MobilityBonus_Bishop) - 1

            mg_score -= MobilityBonus_Bishop[mobility][0]
            eg_score -= MobilityBonus_Bishop[mobility][1]


        elif piece_id == bt.BLACK_ROOK:    
            blackMaterial += ROOK_BLACK
            mg_score -= ROOK_MG_BLACK[sq]
            eg_score -= ROOK_EG_BLACK[sq]
            phase += PHASE_ROOK
            for i in range(8):
                if (whitePawns >> (sq%8 + 8*i)) & 1:
                    WhitePiece += 1
                elif (blackPawns >> (sq%8 + 8*i)) & 1:
                    BlackPiece += 1
            if WhitePiece == 0 and BlackPiece ==0:
                mg_score -= ROOK_MG_OPENFILE
                eg_score -= ROOK_EG_OPENFILE
            elif WhitePiece > 0 and BlackPiece == 0:
                mg_score -= ROOK_MG_SEMIOPENFILE
                eg_score -= ROOK_EG_SEMIOPENFILE

        elif piece_id == bt.BLACK_QUEEN:   
            blackMaterial += QUEEN_BLACK
            mg_score -= QUEEN_MG_BLACK[sq]
            eg_score -= QUEEN_EG_BLACK[sq]
            phase += PHASE_QUEEN

        elif piece_id == bt.BLACK_KING:    
            blackMaterial += KING_BLACK
            mg_score -= KING_MG_BLACK[sq]
            eg_score -= KING_EG_BLACK[sq]



    totalMaterial = whiteMaterial + blackMaterial

    score = totalMaterial + (mg_score * phase + eg_score * (PHASE_MAX - phase)) // PHASE_MAX
    

    
    # The engine requires the score to be relative to the player whose turn it is
    # If absolute score is +100 (White is winning) but it's Black's turn (side_to_move = 1), we must return -100 so the engine knows Black is in a bad position.
    if side_to_move == BLACK_TO_MOVE:
        return -score
    else:
        return score
