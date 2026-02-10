import board_tools as bt
from numba import njit, int64, int32, uint32

# NOTE (Fixed by marker):
# 1. bishop_activity / rook_activity were passed the full board_occupancy array
#    instead of board_occupancy[2]. Fixed.
# 2. ALL helper functions were plain Python (no @njit), causing interpreted-speed
#    execution at every leaf node. Rewritten to Numba-compatible bitboard ops.

PHASE_MAX = 24
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1


# ================= BIT HELPERS =================

@njit
def bitscan(bb):
    """Returns index of least significant set bit."""
    if bb == 0:
        return int32(0)
    sq = int32(0)
    while (bb & int64(1)) == 0:
        sq += 1
        bb >>= 1
    return sq


# ================= ATTACK BITBOARDS =================

@njit
def knight_attacks_bb(sq):
    bb = int64(0)
    rank = sq // 8
    file = sq % 8
    dr = (-2, -2, -1, -1, 1, 1, 2, 2)
    df = (-1, 1, -2, 2, -2, 2, -1, 1)
    for i in range(8):
        r = rank + dr[i]
        f = file + df[i]
        if 0 <= r < 8 and 0 <= f < 8:
            bb |= int64(1) << int64(r * 8 + f)
    return bb


@njit
def bishop_attacks_bb(sq, occupancy):
    attacks = int64(0)
    rank = sq // 8
    file = sq % 8
    dirs_r = (1, 1, -1, -1)
    dirs_f = (1, -1, 1, -1)
    for d in range(4):
        r = rank + dirs_r[d]
        f = file + dirs_f[d]
        while 0 <= r < 8 and 0 <= f < 8:
            bit = int64(1) << int64(r * 8 + f)
            attacks |= bit
            if occupancy & bit:
                break
            r += dirs_r[d]
            f += dirs_f[d]
    return attacks


@njit
def rook_attacks_bb(sq, occupancy):
    attacks = int64(0)
    rank = sq // 8
    file = sq % 8
    dirs_r = (0, 0, 1, -1)
    dirs_f = (1, -1, 0, 0)
    for d in range(4):
        r = rank + dirs_r[d]
        f = file + dirs_f[d]
        while 0 <= r < 8 and 0 <= f < 8:
            bit = int64(1) << int64(r * 8 + f)
            attacks |= bit
            if occupancy & bit:
                break
            r += dirs_r[d]
            f += dirs_f[d]
    return attacks


@njit
def queen_attacks_bb(sq, occupancy):
    return bishop_attacks_bb(sq, occupancy) | rook_attacks_bb(sq, occupancy)


@njit
def pawn_attacks_bb(sq, side):
    attacks = int64(0)
    rank = sq // 8
    file = sq % 8
    direction = 1 if side == 0 else -1
    for df in (-1, 1):
        r = rank + direction
        f = file + df
        if 0 <= r < 8 and 0 <= f < 8:
            attacks |= int64(1) << int64(r * 8 + f)
    return attacks


# ================= ACTIVITY =================

@njit
def knight_activity_score(knight_bb):
    score = int32(0)
    bb = knight_bb
    while bb:
        sq = bitscan(bb)
        score += bt.count_bits(knight_attacks_bb(sq))
        bb &= bb - int64(1)
    return score


@njit
def bishop_activity_score(bishop_bb, occ):
    score = int32(0)
    bb = bishop_bb
    while bb:
        sq = bitscan(bb)
        score += bt.count_bits(bishop_attacks_bb(sq, occ))
        bb &= bb - int64(1)
    return score


@njit
def rook_activity_score(rook_bb, occ):
    score = int32(0)
    bb = rook_bb
    while bb:
        sq = bitscan(bb)
        score += bt.count_bits(rook_attacks_bb(sq, occ))
        bb &= bb - int64(1)
    return score


# ================= PAWNS =================

@njit
def column_count(pawn_bb, file):
    count = int32(0)
    for rank in range(8):
        if (pawn_bb >> int64(rank * 8 + file)) & int64(1):
            count += 1
    return count


@njit
def count_passed_pawns(pawns, enemy_pawns, side):
    count = int32(0)
    direction = int32(1) if side == 0 else int32(-1)
    bb = pawns
    while bb:
        sq = bitscan(bb)
        rank = sq // 8
        file = sq % 8
        is_passed = True
        r = rank + direction
        while 0 <= r < 8:
            for df in (-1, 0, 1):
                f = file + df
                if 0 <= f < 8:
                    if (enemy_pawns >> int64(r * 8 + f)) & int64(1):
                        is_passed = False
            r += direction
        if is_passed:
            count += 1
        bb &= bb - int64(1)
    return count


@njit
def count_isolated_pawns(pawns):
    count = int32(0)
    bb = pawns
    while bb:
        sq = bitscan(bb)
        file = sq % 8
        has_neighbor = False
        for df in (-1, 1):
            f = file + df
            if 0 <= f < 8:
                for r in range(8):
                    if (pawns >> int64(r * 8 + f)) & int64(1):
                        has_neighbor = True
        if not has_neighbor:
            count += 1
        bb &= bb - int64(1)
    return count


# ================= ATTACK MAP =================

@njit
def all_attacks_bb(board_pieces, side):
    """Returns a bitboard of all squares attacked by the given side."""
    attacked = int64(0)
    occupancy = int64(0)
    for i in range(12):
        occupancy |= board_pieces[i]

    offset = int32(0) if side == 0 else int32(6)

    # Pawns
    bb = board_pieces[offset]
    while bb:
        sq = bitscan(bb)
        attacked |= pawn_attacks_bb(sq, side)
        bb &= bb - int64(1)

    # Knights
    bb = board_pieces[offset + 1]
    while bb:
        sq = bitscan(bb)
        attacked |= knight_attacks_bb(sq)
        bb &= bb - int64(1)

    # Bishops
    bb = board_pieces[offset + 2]
    while bb:
        sq = bitscan(bb)
        attacked |= bishop_attacks_bb(sq, occupancy)
        bb &= bb - int64(1)

    # Rooks
    bb = board_pieces[offset + 3]
    while bb:
        sq = bitscan(bb)
        attacked |= rook_attacks_bb(sq, occupancy)
        bb &= bb - int64(1)

    # Queens
    bb = board_pieces[offset + 4]
    while bb:
        sq = bitscan(bb)
        attacked |= queen_attacks_bb(sq, occupancy)
        bb &= bb - int64(1)

    return attacked


@njit
def hanging_piece_score(board_pieces):
    score = int32(0)
    white_att = all_attacks_bb(board_pieces, int32(0))
    black_att = all_attacks_bb(board_pieces, int32(1))

    values = (20, 60, 60, 100, 180)

    # White pieces hanging (attacked by black, not defended by white)
    for i in range(5):
        bb = board_pieces[i]
        while bb:
            sq = bitscan(bb)
            bit = int64(1) << int64(sq)
            if (black_att & bit) and not (white_att & bit):
                score -= values[i]
            bb &= bb - int64(1)

    # Black pieces hanging (attacked by white, not defended by black)
    for i in range(5):
        bb = board_pieces[i + 6]
        while bb:
            sq = bitscan(bb)
            bit = int64(1) << int64(sq)
            if (white_att & bit) and not (black_att & bit):
                score += values[i]
            bb &= bb - int64(1)

    return score


# ================= MAIN EVALUATION =================

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    mg_score = int32(0)
    eg_score = int32(0)

    # -------- MATERIAL COUNTS --------
    wp = bt.count_bits(board_pieces[0])
    wn = bt.count_bits(board_pieces[1])
    wb = bt.count_bits(board_pieces[2])
    wr = bt.count_bits(board_pieces[3])
    wq = bt.count_bits(board_pieces[4])

    bp = bt.count_bits(board_pieces[6])
    bn = bt.count_bits(board_pieces[7])
    bb_ = bt.count_bits(board_pieces[8])
    br = bt.count_bits(board_pieces[9])
    bq = bt.count_bits(board_pieces[10])

    # -------- PHASE --------
    phase = (wn + bn) + (wb + bb_) + 2 * (wr + br) + 4 * (wq + bq)
    if phase > PHASE_MAX:
        phase = PHASE_MAX

    # -------- MATERIAL --------
    material = (100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb_) +
                500 * (wr - br) + 900 * (wq - bq))
    mg_score += material
    eg_score += material

    # -------- CENTER --------
    center = (27, 28, 35, 36)
    for i in range(4):
        piece = bt.get_piece(board_pieces, int32(center[i]))
        if piece != 0:
            if piece > 0:
                mg_score += 20
            else:
                mg_score -= 20

    # -------- DOUBLED PAWNS --------
    for f in range(8):
        if column_count(board_pieces[0], int32(f)) > 1:
            mg_score -= 10
        if column_count(board_pieces[6], int32(f)) > 1:
            mg_score += 10

    # -------- PASSED PAWNS --------
    eg_score += 20 * count_passed_pawns(board_pieces[0], board_pieces[6], int32(0))
    eg_score -= 20 * count_passed_pawns(board_pieces[6], board_pieces[0], int32(1))

    # -------- ISOLATED PAWNS --------
    mg_score -= 10 * count_isolated_pawns(board_pieces[0])
    mg_score += 10 * count_isolated_pawns(board_pieces[6])

    # -------- ACTIVITY --------
    occ = board_occupancy[2]
    mg_score += 5 * knight_activity_score(board_pieces[1])
    mg_score -= 5 * knight_activity_score(board_pieces[7])
    mg_score += 5 * bishop_activity_score(board_pieces[2], occ)
    mg_score -= 5 * bishop_activity_score(board_pieces[8], occ)
    mg_score += 5 * rook_activity_score(board_pieces[3], occ)
    mg_score -= 5 * rook_activity_score(board_pieces[9], occ)

    # -------- HANGING PIECES --------
    mg_score += hanging_piece_score(board_pieces)

    # -------- TAPER --------
    score = (mg_score * phase + eg_score * (PHASE_MAX - phase)) // PHASE_MAX

    if side_to_move == BLACK_TO_MOVE:
        return int32(-score)
    return int32(score)
