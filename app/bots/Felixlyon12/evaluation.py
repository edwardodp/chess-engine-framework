import board_tools as bt
from numba import njit, int64, int32, uint32, bool_
import numpy as np

# NOTE (Marker review — Felixlyon12):
# Overall: Good structure. All functions were already @njit, PST tables are globals,
# bitboard iteration pattern is correct. Significantly better than average.
#
# Fixes applied:
# 1. popcount was O(64) bit-by-bit shift — replaced with bt.count_bits (O(set bits)).
# 2. visible_squares had no file-wrapping guard — bishop on h-file with +9 offset
#    would wrap to a-file. Added file distance check.
# 3. isolated_pawn used is_doubled_pawn (>1 pawn) to check neighbours, but isolation
#    means 0 pawns on adjacent files (>=1 should return not-isolated). Fixed to check
#    for any pawn on adjacent files.
# 4. 750/material always truncated to 0 (material is ~5000-8000). Scaled to 75000.
# 5. Float scale (1.0/1.2) promoted score to float — replaced with integer math
#    (multiply by 10 or 12, divide by 10).
# 6. LATE_KING and BISHOP_TABLE were defined but never used — removed.

"""
Piece Value Constants
"""
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

@njit(bool_(int32, int64, int64))
def is_open_file(file, wp, bp):
    file_mask = FILE_MASKS[file]
    return (wp | bp) & file_mask == 0

# FIX: Added file-distance check to prevent wrapping around board edges.
@njit(int32(int32, int64[:]))
def visible_squares(sq, board_occupancy):
    squares = int32(0)
    rank = sq // 8
    file = sq % 8
    # Four diagonal directions: (dr, df)
    dirs_r = (1, 1, -1, -1)
    dirs_f = (1, -1, 1, -1)
    for d in range(4):
        r = rank + dirs_r[d]
        f = file + dirs_f[d]
        while 0 <= r < 8 and 0 <= f < 8:
            target = r * 8 + f
            if bt.check_square(board_occupancy, int32(target), int32(2)) == 1:
                break
            squares += 1
            r += dirs_r[d]
            f += dirs_f[d]
    return squares

@njit(bool_(int32, int64))
def is_doubled_pawn(file, bb):
    file_mask = FILE_MASKS[file]
    return bt.count_bits(bb & file_mask) > 1

# FIX: Was using is_doubled_pawn (>1) to check for neighbours, which missed files
# with exactly 1 pawn. Now checks for any pawn (>=1) on adjacent files.
@njit(bool_(int32, int64))
def has_pawn_on_file(file, bb):
    file_mask = FILE_MASKS[file]
    return (bb & file_mask) != 0

@njit(bool_(int32, int64))
def isolated_pawn(sq, bb):
    file = sq % 8
    if file == 0:
        return not has_pawn_on_file(int32(1), bb)
    elif file == 7:
        return not has_pawn_on_file(int32(6), bb)
    else:
        return not (has_pawn_on_file(int32(file - 1), bb) or
                    has_pawn_on_file(int32(file + 1), bb))

@njit(bool_(int32, bool_, int64, int64))
def is_passed_pawn(sq, white, wp, bp):
    file = sq & 7
    rank = sq >> 3

    file_mask = FILE_MASKS[file]
    if file > 0:
        file_mask |= FILE_MASKS[file - 1]
    if file < 7:
        file_mask |= FILE_MASKS[file + 1]

    if white:
        ahead_mask = file_mask & (~((int64(1) << int64(sq + 1)) - int64(1)))
        return (bp & ahead_mask) == 0
    else:
        ahead_mask = file_mask & ((int64(1) << int64(sq)) - int64(1))
        return (wp & ahead_mask) == 0

@njit(int32(int32, int64, bool_))
def king_pawns(kingSq, bb, white):
    if white:
        return bt.count_bits(bb & int64(WHITE_PAWN_SHIELD[kingSq]))
    else:
        return bt.count_bits(bb & int64(BLACK_PAWN_SHIELD[kingSq]))

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    score = int32(0)

    wp = board_pieces[0]
    wn = board_pieces[1]
    wb = board_pieces[2]
    wr = board_pieces[3]
    wq = board_pieces[4]
    wk = board_pieces[5]

    bp = board_pieces[6]
    bn = board_pieces[7]
    bB = board_pieces[8]
    br = board_pieces[9]
    bq = board_pieces[10]
    bk = board_pieces[11]

    # FIX: Replaced custom popcount (O(64)) with bt.count_bits (O(set bits))
    white_material = (
        PAWN   * bt.count_bits(wp) +
        KNIGHT * bt.count_bits(wn) +
        BISHOP * bt.count_bits(wb) +
        ROOK   * bt.count_bits(wr) +
        QUEEN  * bt.count_bits(wq)
    )

    black_material = (
        PAWN   * bt.count_bits(bp) +
        KNIGHT * bt.count_bits(bn) +
        BISHOP * bt.count_bits(bB) +
        ROOK   * bt.count_bits(br) +
        QUEEN  * bt.count_bits(bq)
    )

    material = white_material + black_material
    score = white_material - black_material

    # FIX: Replaced float scale (1.0/1.2) with integer math (* 10 or * 12, // 10)
    if material <= 2800:
        table = LATE_PAWN_TABLE
        scale_num = int32(10)
    elif material >= 7500:
        table = PAWN_TABLE
        scale_num = int32(12)
    else:
        table = PAWN_TABLE
        scale_num = int32(10)

    bb = wp
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score += table[63 - sq] * scale_num // 10
        if not is_doubled_pawn(sq & 7, wp):
            score += 8
        if not isolated_pawn(int32(sq), wp):
            score += 8
        if is_passed_pawn(int32(sq), True, wp, bp):
            score += 25
        bb &= bb - 1

    bb = bp
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score -= table[sq] * scale_num // 10
        if not is_doubled_pawn(sq & 7, bp):
            score -= 8
        if not isolated_pawn(int32(sq), bp):
            score -= 8
        if is_passed_pawn(int32(sq), False, wp, bp):
            score -= 25
        bb &= bb - 1

    # Knights
    knight_scale = int32(3) if material >= 7700 else int32(1)
    bb = wn
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score += KNIGHT_TABLE[sq] * knight_scale
        bb &= bb - 1

    bb = bn
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score -= KNIGHT_TABLE[sq] * knight_scale
        bb &= bb - 1

    # Bishops
    bb = wb
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score += 7 * visible_squares(int32(sq), board_occupancy)
        if sq <= 7:
            score -= 10
        bb &= bb - 1

    bb = bB
    while bb:
        sq = bit_scan_forward(bb & -bb)
        score -= 7 * visible_squares(int32(sq), board_occupancy)
        if sq >= 56:
            score += 10
        bb &= bb - 1

    # Rooks
    bb = wr
    while bb:
        sq = bit_scan_forward(bb & -bb)
        if is_open_file(sq & 7, wp, bp):
            score += 15
        score += ROOK_TABLE[sq]
        bb &= bb - 1

    bb = br
    while bb:
        sq = bit_scan_forward(bb & -bb)
        if is_open_file(sq & 7, wp, bp):
            score -= 15
        score -= ROOK_TABLE[sq]
        bb &= bb - 1

    # Kings
    bb = wk
    while bb:
        sq = bit_scan_forward(bb & -bb)
        if material <= 2800:
            score += KNIGHT_TABLE[sq]
        else:
            score += EARLY_KING[sq]
        if (sq <= 2 or (sq >= 6 and sq <= 7)) and material > 2800:
            score += king_pawns(int32(sq), wp, True) * 40
        score += KING
        bb &= bb - 1

    bb = bk
    while bb:
        sq = bit_scan_forward(bb & -bb)
        if material <= 2800:
            score -= KNIGHT_TABLE[sq]
        else:
            score -= EARLY_KING[63 - sq]
        if (sq >= 62 or (sq <= 58 and sq >= 56)) and material > 2800:
            score -= king_pawns(int32(sq), bp, False) * 40
        score -= KING
        bb &= bb - 1

    # FIX: 750/material was always 0 (int truncation). Scaled numerator to 75000
    # so a 200cp advantage with ~8000 material yields ~9cp tempo bonus.
    if material > 0:
        if white_material >= black_material + 200:
            score += 75000 // material
        elif white_material + 200 <= black_material:
            score -= 75000 // material

    if side_to_move == BLACK_TO_MOVE:
        return int32(-score)
    return int32(score)
