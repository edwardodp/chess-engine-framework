import board_tools as bt
from numba import njit, int64, int32, uint32, bool_
import numpy as np

"""
Piece Value Constants
"""
# White Piece Values
PAWN   = 100
KNIGHT = 320
BISHOP = 330
ROOK   = 500
QUEEN  = 900
KING   = 20000

PAWN_TABLE = np.array([0,  0,  0,  0,  0,  0,  0,  0,
                      75, 75, 75, 75, 75, 75, 75, 75,
                      25, 25, 29, 29, 29, 29, 25, 25,
                      4,  8, 12, 21, 21, 12,  8,  4,
                      0,  4,  8, 17, 17,  8,  4,  0,
                      4, -4, -8,  4,  4, -8, -4,  4,
                      4,  8,  8,-17,-17,  8,  8,  4,
                      0,  0,  0,  0,  0,  0,  0,  0])
LATE_PAWN_TABLE = np.array([0,  0,  0,  0,  0,  0,  0,  0,
                      75, 75, 75, 75, 75, 75, 75, 75,
                      25, 25, 29, 29, 29, 29, 25, 25,
                      20, 20, 21, 21, 21, 21,  20,  20,
                      10,  10,  17, 17, 17,  17,  10,  10,
                      8, 8, 8,  8,  8, 8, 8,  8,
                      4,  8,  8,-17,-17,  8,  8,  4,
                      0,  0,  0,  0,  0,  0,  0,  0])
KNIGHT_TABLE = np.array([-8, -4, -4, -4, -4, -4, -4, -8,
                      -4, 0, 0, 0, 0, 0, 0, -4,
                      -4, 0, 17, 17, 17, 17, 0, -4,
                      -4, 0, 17, 21, 21, 17, 0, -4,
                      -4, 0, 17, 21, 21, 17, 0, -4,
                      -4, 0, 17, 17, 17, 17, 0, -4,
                      -4, 0, 0, 0, 0, 0, 0,  -4,
                      -8, -4, -4, -4, -4, -4, -4, -8])
BISHOP_TABLE = np.array([-59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10])
ROOK_TABLE = np.array([
    0,   0,   0,   0,   0,   0,   0,   0,
    20,   20,   20,   20,   20,   20,   20,   20,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   10,   10,   5,   0,   0
])
EARLY_KING = np.array([0, 8, 8, -20, -4, 8, 21, 0,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10,
                      -10, -10, -10, -10, -10, -10, -10, -10])
LATE_KING = np.array([
    -95,  -95,  -90,  -90,  -90,  -90,  -95,  -95,
    -95,  -50,  -50,  -50,  -50,  -50,  -50,  -95,
    -90,  -50,  -20,  -20,  -20,  -20,  -50,  -90,
    -90,  -50,  -20,    0,    0,  -20,  -50,  -90,
    -90,  -50,  -20,    0,    0,  -20,  -50,  -90,
    -90,  -50,  -20,  -20,  -20,  -20,  -50,  -90,
    -95,  -50,  -50,  -50,  -50,  -50,  -50,  -95,
    -95,  -95,  -90,  -90,  -90,  -90,  -95,  -95,
])
                      
WHITE_PAWN_SHIELD = np.array([                 768,                 1792,                 3584,
                 7168,                14336,                28672,
                57344,                49152,               196608,
               458752,               917504,              1835008,
              3670016,              7340032,             14680064,
             12582912,             50331648,            117440512,
            234881024,            469762048,            939524096,
           1879048192,           3758096384,           3221225472,
          12884901888,          30064771072,          60129542144,
         120259084288,         240518168576,         481036337152,
         962072674304,         824633720832,        3298534883328,
        7696581394432,       15393162788864,       30786325577728,
       61572651155456,      123145302310912,      246290604621824,
      211106232532992,      844424930131968,     1970324836974592,
     3940649673949184,     7881299347898368,    15762598695796736,
    31525197391593472,    63050394783186944,    54043195528445952,
   216172782113783808,   504403158265495552,  1008806316530991104,
  2017612633061982208,  4035225266123964416,  8070450532247928832,
 16140901064495857664, 13835058055282163712,                    0,
                    0,                    0,                    0,
                    0,                    0,                    0,
                    0], dtype=np.uint64)
                    
BLACK_PAWN_SHIELD = np.array([                0,                 0,                 0,                 0,
                 0,                 0,                 0,                 0,
                 3,                 7,                14,                28,
                56,               112,               224,               192,
               768,              1792,              3584,              7168,
             14336,             28672,             57344,             49152,
            196608,            458752,            917504,           1835008,
           3670016,           7340032,          14680064,          12582912,
          50331648,         117440512,         234881024,         469762048,
         939524096,        1879048192,        3758096384,        3221225472,
       12884901888,       30064771072,       60129542144,      120259084288,
      240518168576,      481036337152,      962072674304,      824633720832,
     3298534883328,     7696581394432,    15393162788864,    30786325577728,
    61572651155456,   123145302310912,   246290604621824,   211106232532992,
   844424930131968,  1970324836974592,  3940649673949184,  7881299347898368,
 15762598695796736, 31525197391593472, 63050394783186944, 54043195528445952], dtype=np.uint64)
                      
FILE_MASKS = np.array([
    0x0101010101010101,  # file A
    0x0202020202020202,  # file B
    0x0404040404040404,  # file C
    0x0808080808080808,
    0x1010101010101010,
    0x2020202020202020,
    0x4040404040404040,
    0x8080808080808080,  # file H
], dtype=np.uint64)

BISHOP_VISION = np.array([[7,14,21,28,35,42,49],[9,18,27,36,45,54,63]])

"""
Turn Constants
"""
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1

@njit(int32(int64))
def bit_scan_forward(bb):
    sq = 0
    while (bb & 1) == 0:
        bb >>= 1
        sq += 1
    return sq
    
@njit(int32(int64))
def popcount(bb):
    count = 0
    while bb != 0 and bb != -1:
        count += bb & 1
        bb >>= 1
        if bb == -1:
            count += 1
            break
    return count

@njit(bool_(int32, int64, int64))
def is_open_file(file, wp, bp):
    file_mask = FILE_MASKS[file]
    return (wp | bp) & file_mask == 0

@njit(int32(int32, int64[:]))
def visible_squares(sq, board_occupancy):
    squares = 0
    for direction in BISHOP_VISION:
        for off in direction:
            if sq + off <= 63:
                if bt.check_square(board_occupancy, sq + off, 2) == 1:
                    break
                else:
                   squares += 1
            else:
                break
    for direction in BISHOP_VISION:
        for off in direction:
            if sq - off >= 0:
                if bt.check_square(board_occupancy, sq - off, 2) == 1:
                    break
                else:
                   squares += 1
            else:
                break
    return squares


@njit(bool_(int32, int64))
def is_doubled_pawn(file, bb):
    file_mask = FILE_MASKS[file]
    return popcount(bb & file_mask) > 1

@njit(bool_(int32, int64))
def isolated_pawn(sq, bb):
    file = sq % 8
    if file == 7:
        if is_doubled_pawn((sq - 1) & 7, bb):
            return False
    elif file == 0:
        if is_doubled_pawn((sq + 1) & 7, bb):
            return False
    else:
        if is_doubled_pawn((sq - 1) & 7, bb) or is_doubled_pawn((sq + 1) & 7, bb):
            return False
    return True
    
@njit(bool_(int32, bool_, int64, int64))
def is_passed_pawn(sq, white, wp, bp):
    file = sq & 7
    rank = sq >> 3

    # Step 1: file + adjacent files
    file_mask = FILE_MASKS[file]
    if file > 0:
        file_mask |= FILE_MASKS[file - 1]
    if file < 7:
        file_mask |= FILE_MASKS[file + 1]

    if white:
        # squares in front (higher ranks)
        ahead_mask = file_mask & (~((1 << (sq + 1)) - 1))
        return (bp & ahead_mask) == 0
    else:
        # squares in front for black (lower ranks)
        ahead_mask = file_mask & ((1 << sq) - 1)
        return (wp & ahead_mask) == 0

@njit(int32(int32, int64, bool_))
def king_pawns(kingSq, bb, white):
    if white:
        return popcount(bb & WHITE_PAWN_SHIELD[kingSq])
    else:
        return popcount(bb & BLACK_PAWN_SHIELD[kingSq])
    
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
    
    score = 0
    material = 0

    # Unpack bitboards (replace with your indices from board_tools)
    wp = board_pieces[0]  # WHITE_PAWN
    wn = board_pieces[1]  # WHITE_KNIGHT
    wb = board_pieces[2]  # WHITE_BISHOP
    wr = board_pieces[3]  # WHITE_ROOK
    wq = board_pieces[4]  # WHITE_QUEEN
    wk = board_pieces[5]  # WHITE_KING

    bp = board_pieces[6]  # BLACK_PAWN
    bn = board_pieces[7]  # BLACK_KNIGHT
    bB = board_pieces[8]  # BLACK_BISHOP
    br = board_pieces[9]  # BLACK_ROOK
    bq = board_pieces[10] # BLACK_QUEEN
    bk = board_pieces[11] # BLACK_KING
    
    white_material = (
        PAWN   * popcount(wp) +
        KNIGHT * popcount(wn) +
        BISHOP * popcount(wb) +
        ROOK   * popcount(wr) +
        QUEEN  * popcount(wq)
    )
    
    black_material = (
        PAWN   * popcount(bp) +
        KNIGHT * popcount(bn) +
        BISHOP * popcount(bB) +
        ROOK   * popcount(br) +
        QUEEN  * popcount(bq)
    )
    
    material = white_material + black_material
    
    score = white_material - black_material
    
    """
    Pawn positioning
    """
    if material <= 2800:
        table = LATE_PAWN_TABLE
        scale = 1.0
    elif material >= 7500:
        table = PAWN_TABLE
        scale = 1.2
    else:
        table = PAWN_TABLE
        scale = 1.0

    bb = wp
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score += table[63 - sq] * scale
        if not is_doubled_pawn(sq & 7, wp):
            score += 8
        if not isolated_pawn(sq, wp):
            score += 8
        if is_passed_pawn(sq, True, wp, bp):
            score += 25
        bb &= bb - 1
    
    bb = bp
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score -= table[sq] * scale
        if not is_doubled_pawn(sq & 7, bp):
            score -= 8
        if not isolated_pawn(sq, bp):
            score -= 8
        if is_passed_pawn(sq, False, wp, bp):
            score -= 25
        bb &= bb - 1
        
    """
    Knight positioning
    """
    bb = wn
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score += KNIGHT_TABLE[sq] * (3 if material >= 7700 else 1)
        bb &= bb - 1
        
    bb = bn
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score -= KNIGHT_TABLE[sq] * (3 if material >= 7700 else 1)
        bb &= bb - 1
        
    """
    Bishop positioning
    """
    bb = wb
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score += 7 * visible_squares(sq, board_occupancy)
        if sq <= 7:
            score -= 10
        bb &= bb - 1
        
    bb = bB
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        score -= 7 * visible_squares(sq, board_occupancy)
        if sq >= 56:
            score += 10
        bb &= bb - 1
        
    """
    Rook positioning
    """
    bb = wr
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        if is_open_file(sq & 7, wp, bp):
            score += 15
        score += ROOK_TABLE[sq]
        bb &= bb - 1
        
    bb = br
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        if is_open_file(sq & 7, wp, bp):
            score -= 15
        score -= ROOK_TABLE[sq]
        bb &= bb - 1
        
    """
    King positioning
    """
    bb = wk
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        if material <= 2800:
            score += KNIGHT_TABLE[sq]
        else:
            score += EARLY_KING[sq]
        if (sq <= 2 or (sq >= 6 and sq <= 7)) and material > 2800:
            score += king_pawns(sq, wp, True) * 40
        score += KING
        bb &= bb - 1
        
    bb = bk
    while bb:
        lsb = bb & -bb
        sq = bit_scan_forward(lsb)
        if material <= 2800:
            score -= KNIGHT_TABLE[sq]
        else:
            score -= EARLY_KING[63-sq]
        if (sq >= 62 or (sq <= 58 and sq >= 56)) and material > 2800:
            score -= king_pawns(sq, bp, False) * 40
        score -= KING
        bb &= bb - 1
        
    if white_material >= black_material + 200:
        score += 750 / material
    elif white_material + 200 <= black_material:
        score -= 750 / material

    # The engine requires the score to be relative to the player whose turn it is
    # If absolute score is +100 (White is winning) but it's Black's turn (side_to_move = 1), we must return -100 so the engine knows Black is in a bad position.
    if side_to_move == BLACK_TO_MOVE:
        return -score
    return score
