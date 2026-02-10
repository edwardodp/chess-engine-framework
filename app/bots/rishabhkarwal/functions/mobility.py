import board_tools as bt
from numba import njit, int64, int32, uint32, uint64
import numpy as np

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

MV_MG = np.array([82, 337, 365, 477, 1025, 0], dtype=np.int32)
MV_EG = np.array([94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_SCORES = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)

# trapped penalty (applied if safe moves == 0)
TRAPPED_PENALTY = np.array([0, 50, 50, 35, 50, 0], dtype=np.int32)

mg_p = (0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0)
mg_n = (-167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23)
mg_b = (-29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21)
mg_r = (32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26)
mg_q = (-28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50)
mg_k = (-65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, 3, -14, -13, -45, -28, -4, -41, -27, -27, -16, -17, -25, -18, -3, -19, -74, -35, -18, -18, -11, 15, 4, -17)

eg_p = (0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0)
eg_n = (-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64)
eg_b = (-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -36, -13, -5, -18, -4, 16, 13, 16, 17, 27, 20, -5, 7, 17, 32, 25, 24, 15, 22, 15, 6, 20, 26, 28, 30, 24, 16, 2, 7, 15, 16, 19, 22, 22, 13, 2, -1, 11, 19, 18, 20, 22, 17, 9, -23, -9, -23, -5, -9, -16, -5, -17)
eg_r = (13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20)
eg_q = (-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41)
eg_k = (-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43)

def prepare_tables(tuples, materials):
    table_12 = np.zeros((12, 64), dtype=np.int32)
    for i, t in enumerate(tuples):
        arr = np.array(t, dtype=np.int32).reshape(8, 8)[::-1].flatten() + materials[i]
        table_12[i] = arr
        table_12[i + 6] = -(arr.reshape(8, 8)[::-1].flatten())
    return table_12

MG_TABLES = prepare_tables((mg_p, mg_n, mg_b, mg_r, mg_q, mg_k), MV_MG)
EG_TABLES = prepare_tables((eg_p, eg_n, eg_b, eg_r, eg_q, eg_k), MV_EG)
PHASE_WEIGHTS = np.concatenate((PHASE_SCORES, PHASE_SCORES))

KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
RAYS = np.zeros((64, 8), dtype=np.uint64)

for s in range(64):
    mask = 0
    x, y = s % 8, s // 8
    for dx, dy in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 8 and 0 <= ny < 8: mask |= (1 << (ny * 8 + nx))
    KNIGHT_ATTACKS[s] = mask

    for i, (dx, dy) in enumerate([(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]):
        r_mask = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < 8 and 0 <= cy < 8:
            r_mask |= (1 << (cy * 8 + cx))
            cx += dx; cy += dy
        RAYS[s, i] = r_mask

DE_BRUIJN_MAGIC = uint64(0x03f79d71b4cb0a89)
DE_BRUIJN_LOOKUP = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], dtype=np.int32)

@njit(int32(uint64), inline='always')
def count_bits(n):
    c = 0
    while n:
        n &= n - uint64(1)
        c += 1
    return c

@njit(int32(int64[:], int64[:], uint32), boundscheck=False, fastmath=True)
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    mg_score = 0
    eg_score = 0
    
    white_occ = uint64(board_occupancy[0])
    black_occ = uint64(board_occupancy[1])
    all_occ = uint64(board_occupancy[2])
    
    wp = uint64(board_pieces[0])
    bp = uint64(board_pieces[6])

    phase = 0
    phase += (count_bits(uint64(board_pieces[1])) + count_bits(uint64(board_pieces[7]))) * 1
    phase += (count_bits(uint64(board_pieces[2])) + count_bits(uint64(board_pieces[8]))) * 1
    phase += (count_bits(uint64(board_pieces[3])) + count_bits(uint64(board_pieces[9]))) * 2
    phase += (count_bits(uint64(board_pieces[4])) + count_bits(uint64(board_pieces[10]))) * 4
    
    do_mobility = phase <= 24

    # pawn Attacks (safe mobility)
    not_a = uint64(0xFEFEFEFEFEFEFEFE)
    not_h = uint64(0x7F7F7F7F7F7F7F7F)
    wp_attacks = ((wp << uint64(9)) & not_a) | ((wp << uint64(7)) & not_h)
    bp_attacks = ((bp >> uint64(9)) & not_h) | ((bp >> uint64(7)) & not_a)

    mg_tab = MG_TABLES
    eg_tab = EG_TABLES
    lookup = DE_BRUIJN_LOOKUP
    magic = DE_BRUIJN_MAGIC
    rays = RAYS
    trap_pen = TRAPPED_PENALTY
    
    for p_type in range(12):
        bb = uint64(board_pieces[p_type])
        if not bb: continue

        piece_idx = p_type % 6
        is_white = (p_type < 6)
        my_occ = white_occ if is_white else black_occ
        enemy_pawn_attacks = bp_attacks if is_white else wp_attacks
        safe_mask = (~my_occ) & (~enemy_pawn_attacks)

        while bb:
            lsb = bb & (~bb + uint64(1))
            sq = lookup[((lsb * magic) >> uint64(58))]
            
            mg_score += mg_tab[p_type, sq]
            eg_score += eg_tab[p_type, sq]

            if do_mobility and 1 <= piece_idx <= 4:
                attacks = uint64(0)
                
                if piece_idx == 1:
                    attacks = KNIGHT_ATTACKS[sq]
                else:
                    if piece_idx == 2 or piece_idx == 4:
                        for d in range(4, 6):
                            r = rays[sq, d]
                            b = r & all_occ
                            if b:
                                blocker = b & (~b + uint64(1))
                                b_sq = lookup[((blocker * magic) >> uint64(58))]
                                r ^= rays[b_sq, d]
                            attacks |= r
                        for d in range(6, 8):
                            r = rays[sq, d]
                            b = r & all_occ
                            if b:
                                n = b
                                n|=n>>1; n|=n>>2; n|=n>>4; n|=n>>8; n|=n>>16; n|=n>>32
                                b_sq = lookup[((n * magic) >> uint64(58))]
                                r ^= rays[b_sq, d]
                            attacks |= r

                    if piece_idx == 3 or piece_idx == 4:
                        for d in (0, 2):
                            r = rays[sq, d]
                            b = r & all_occ
                            if b:
                                blocker = b & (~b + uint64(1))
                                b_sq = lookup[((blocker * magic) >> uint64(58))]
                                r ^= rays[b_sq, d]
                            attacks |= r
                        for d in (1, 3):
                            r = rays[sq, d]
                            b = r & all_occ
                            if b:
                                n = b
                                n|=n>>1; n|=n>>2; n|=n>>4; n|=n>>8; n|=n>>16; n|=n>>32
                                b_sq = lookup[((n * magic) >> uint64(58))]
                                r ^= rays[b_sq, d]
                            attacks |= r

                # if 0 safe moves -> penalty
                if (attacks & safe_mask) == 0:
                    penalty = trap_pen[piece_idx]
                    if is_white:
                        mg_score -= penalty
                        eg_score -= penalty
                    else:
                        mg_score += penalty
                        eg_score += penalty

            bb ^= lsb

    if phase > 24: phase = 24
    final = ((mg_score * phase) + (eg_score * (24 - phase))) // 24

    return -final if side_to_move == 1 else final