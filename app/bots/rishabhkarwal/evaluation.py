# NOTE (Fixed by marker): Original evaluation.py was a one-liner:
#   from functions.hce import evaluation_function
# This required a 'functions/' subpackage with __init__.py and hce.py, which is
# fragile in multiprocessing contexts and caused noisy import warnings from
# unrelated files (uci.py, driver.py, match.py) that depend on python-chess.
# Flattened by copying hce.py contents directly into evaluation.py.
# No code changes were made to the evaluation logic itself.

from numba import njit, int64, int32, uint32, uint64
import numpy as np

## Constants

# Piece Indices
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
BLACK = 6 # offset for black piece index

# Material Values
MATERIAL_VALUE_MG = np.array([82, 337, 365, 477, 1025, 0], dtype=np.int32)
MATERIAL_VALUE_EG = np.array([94, 281, 297, 512, 936, 0], dtype=np.int32)

# Phase Values
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = PHASE_WEIGHTS[KNIGHT] * 4 + PHASE_WEIGHTS[BISHOP] * 4 + PHASE_WEIGHTS[ROOK] * 4 + PHASE_WEIGHTS[QUEEN] * 2
ENDGAME_PHASE = int(round(MAX_PHASE * 0.25, 0)) # when phase > this
OPENING_PHASE = int(round(MAX_PHASE * 0.33, 0)) # when phase < this

## Penalties

# Mobility
TRAPPED_PENALTY = np.array([0, 30, 40, 50, 50, 0], dtype=np.int32) # no pseudo-legal moves

# Pawn Structure
DOUBLED_PAWN_PENALTY_MG = 12
DOUBLED_PAWN_PENALTY_EG = 17
ISOLATED_PAWN_PENALTY_MG = 15
ISOLATED_PAWN_PENALTY_EG = 20
PAWN_ISLAND_PENALTY_MG = 6
PAWN_ISLAND_PENALTY_EG = 15

# King Safety
KING_ZONE_ATTACK_PENALTY = 11 # per piece attacking squares near king

## Bonuses

# Mobility
MOBILITY_BONUS_MG = np.array([0, 4, 4, 2, 1, 0], dtype=np.int32) # per psuedo-legal move
MOBILITY_BONUS_EG = np.array([0, 3, 3, 7, 5, 0], dtype=np.int32)
PASSED_PAWN_BONUS_MG = np.array([0, 2, 4, 8, 15, 24, 47, 0], dtype=np.int32)
PASSED_PAWN_BONUS_EG = np.array([0, 7, 15, 26, 59, 89, 145, 0], dtype=np.int32)

# Pieces
BISHOP_PAIR_BONUS_MG = 24
BISHOP_PAIR_BONUS_EG = 35
ROOK_OPEN_FILE_BONUS_MG = 7
ROOK_OPEN_FILE_BONUS_EG = 10
ROOK_SEMI_OPEN_FILE_BONUS_MG = 5
ROOK_SEMI_OPEN_FILE_BONUS_EG = 8

# King Safety
PAWN_SHIELD_BONUS_MG = 12 # per pawn in front of king

# Batteries
ROOK_BATTERY_BONUS_MG = 15
ROOK_BATTERY_BONUS_EG = 25
ROOK_QUEEN_BATTERY_BONUS_MG = 14
ROOK_QUEEN_BATTERY_BONUS_EG = 19
QUEEN_BISHOP_BATTERY_BONUS_MG = 12
QUEEN_BISHOP_BATTERY_BONUS_EG = 8

# Mop-up Evaluation
KING_EDGE_DISTANCE_BONUS = 15  # push losing king to edge
KING_PROXIMITY_BONUS = 10  # bring winning king close to losing king


## Piece-Square Tables
mg_pawn_table = (
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0
)

mg_knight_table = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23
)

mg_bishop_table = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21
)

mg_rook_table = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26
)

mg_queen_table = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50
)

mg_king_table = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14,   3, -14, -13, -45, -28,  -4, -41,
    -27, -27, -16, -17, -25, -18,  -3, -19,
    -74, -35, -18, -18, -11,  15,   4, -17
)

eg_pawn_table = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0
)

eg_knight_table = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64
)

eg_bishop_table = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12, -36, -13,  -5, -18,
     -4,  16,  13,  16,  17,  27,  20,  -5,
      7,  17,  32,  25,  24,  15,  22,  15,
      6,  20,  26,  28,  30,  24,  16,   2,
      7,  15,  16,  19,  22,  22,  13,   2,
     -1,  11,  19,  18,  20,  22,  17,   9,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17
)

eg_rook_table = (
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20
)

eg_queen_table = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41
)

eg_king_table = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
)

def prepare_piece_square_tables(table_tuples, material_values):
    psqt_all_pieces = np.zeros((12, 64), dtype=np.int32)
    
    for i, pst in enumerate(table_tuples):
        # white pieces: flip board vertically and add material value
        white_table = np.array(pst, dtype=np.int32).reshape(8, 8)[::-1].flatten()
        white_table += material_values[i]
        psqt_all_pieces[i] = white_table
        
        # black pieces: negate and flip vertically
        black_table = -(white_table.reshape(8, 8)[::-1].flatten())
        psqt_all_pieces[i + BLACK] = black_table
    
    return psqt_all_pieces

PSQT_MIDDLEGAME = prepare_piece_square_tables(
    (mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table),
    MATERIAL_VALUE_MG
)

PSQT_ENDGAME = prepare_piece_square_tables(
    (eg_pawn_table, eg_knight_table, eg_bishop_table, eg_rook_table, eg_queen_table, eg_king_table),
    MATERIAL_VALUE_EG
)

## Precomputed Attack Tables

KNIGHT_ATTACK_MASKS = np.zeros(64, dtype=np.uint64)
SLIDING_RAYS = np.zeros((64, 8), dtype=np.uint64) # sliding piece rays in 8 cardinal directions
KING_ATTACK_MASKS = np.zeros(64, dtype=np.uint64)

# knight attacks
for square in range(64):
    attack_mask = 0
    file, rank = square % 8, square // 8
    knight_deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    
    for delta_file, delta_rank in knight_deltas:
        target_file, target_rank = file + delta_file, rank + delta_rank
        if 0 <= target_file < 8 and 0 <= target_rank < 8:
            attack_mask |= (1 << (target_rank * 8 + target_file))
    
    KNIGHT_ATTACK_MASKS[square] = attack_mask

# sliding rays
for square in range(64):
    file, rank = square % 8, square // 8
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    for direction_idx, (delta_file, delta_rank) in enumerate(directions):
        ray_mask = 0
        current_file, current_rank = file + delta_file, rank + delta_rank
        
        while 0 <= current_file < 8 and 0 <= current_rank < 8:
            ray_mask |= (1 << (current_rank * 8 + current_file))
            current_file += delta_file
            current_rank += delta_rank
        
        SLIDING_RAYS[square, direction_idx] = ray_mask

# king attacks
for square in range(64):
    attack_mask = 0
    file, rank = square % 8, square // 8
    king_deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for delta_file, delta_rank in king_deltas:
        target_file, target_rank = file + delta_file, rank + delta_rank
        if 0 <= target_file < 8 and 0 <= target_rank < 8:
            attack_mask |= (1 << (target_rank * 8 + target_file))
    
    KING_ATTACK_MASKS[square] = attack_mask

# file and rank masks for pawn structure and battery detection
FILE_MASKS = np.array([
    0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
    0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080
], dtype=np.uint64)

RANK_MASKS = np.array([
    0x00000000000000FF, 0x000000000000FF00, 0x0000000000FF0000, 0x00000000FF000000,
    0x000000FF00000000, 0x0000FF0000000000, 0x00FF000000000000, 0xFF00000000000000
], dtype=np.uint64)

# diagonal masks (for battery detection)
# main diagonal and anti-diagonal
DIAGONAL_A1H8 = uint64(0x8040201008040201)
DIAGONAL_A8H1 = uint64(0x0102040810204080)

## Helpers

# de bruijn sequence for fast bit scanning
DE_BRUIJN_MAGIC = uint64(0x03f79d71b4cb0a89)
DE_BRUIJN_INDEX_TABLE = np.array([
     0,  1, 48,  2, 57, 49, 28,  3,
    61, 58, 50, 42, 38, 29, 17,  4,
    62, 55, 59, 36, 53, 51, 43, 22,
    45, 39, 33, 30, 24, 18, 12,  5,
    63, 47, 56, 27, 60, 41, 37, 16,
    54, 35, 52, 21, 44, 32, 23, 11,
    46, 26, 40, 15, 34, 20, 31, 10,
    25, 14, 19,  9, 13,  8,  7,  6
], dtype=np.int32)

@njit(int32(uint64), inline='always')
def popcount(bitboard):
    """count set bits using brian kernighan's algorithm"""
    count = 0
    while bitboard:
        bitboard &= bitboard - uint64(1)
        count += 1
    return count

@njit(int32(uint64), inline='always')
def get_lsb_index(bitboard):
    """extract least significant bit index using de bruijn multiplication"""
    lsb = bitboard & (~bitboard + uint64(1))
    return DE_BRUIJN_INDEX_TABLE[((lsb * DE_BRUIJN_MAGIC) >> uint64(58))]

@njit(int32(int32, int32), inline='always')
def chebyshev_distance(square1, square2):
    """calculate chebyshev distance (king distance) between two squares"""
    file1, rank1 = square1 % 8, square1 // 8
    file2, rank2 = square2 % 8, square2 // 8
    file_distance = abs(file1 - file2)
    rank_distance = abs(rank1 - rank2)
    return max(file_distance, rank_distance)

@njit(int32(int32), inline='always')
def edge_distance(square):
    """calculate manhattan distance from square to nearest board edge"""
    file, rank = square % 8, square // 8
    file_dist = min(file, 7 - file)
    rank_dist = min(rank, 7 - rank)
    return min(file_dist, rank_dist)

## Main Evaluation Function

@njit(int32(int64[:], int64[:], uint32), boundscheck=False, fastmath=True)
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    # initialise middlegame and endgame scores
    score_mg = 0
    score_eg = 0
    
    # convert occupancy bitboards to unsigned for bitwise operations
    white_pieces = uint64(board_occupancy[0])
    black_pieces = uint64(board_occupancy[1])
    all_pieces = uint64(board_occupancy[2])
    
    # extract pawn bitboards for structure evaluation
    white_pawns = uint64(board_pieces[0])
    black_pawns = uint64(board_pieces[BLACK])
    
    ## Phase Calculation
    game_phase = 0
    for piece in [KNIGHT, BISHOP, ROOK, QUEEN]:
        game_phase += (popcount(uint64(board_pieces[piece])) + popcount(uint64(board_pieces[piece + BLACK]))) * PHASE_WEIGHTS[piece]
    game_phase = min(game_phase, MAX_PHASE)

    # determine if we should calculate expensive features
    is_opening = game_phase >= OPENING_PHASE
    is_endgame = game_phase <= ENDGAME_PHASE
    
    ## Pawn Attacks
    not_a_file = uint64(0xFEFEFEFEFEFEFEFE)
    not_h_file = uint64(0x7F7F7F7F7F7F7F7F)
    
    white_pawn_attacks = ((white_pawns << uint64(9)) & not_a_file) | ((white_pawns << uint64(7)) & not_h_file)
    black_pawn_attacks = ((black_pawns >> uint64(9)) & not_h_file) | ((black_pawns >> uint64(7)) & not_a_file)
    

    # cache lookup tables for performance
    psqt_mg = PSQT_MIDDLEGAME
    psqt_eg = PSQT_ENDGAME
    de_bruijn_lookup = DE_BRUIJN_INDEX_TABLE
    de_bruijn_magic = DE_BRUIJN_MAGIC
    sliding_rays = SLIDING_RAYS
    knight_attacks = KNIGHT_ATTACK_MASKS
    king_attacks = KING_ATTACK_MASKS
    trapped_penalties = TRAPPED_PENALTY
    mobility_mg = MOBILITY_BONUS_MG
    mobility_eg = MOBILITY_BONUS_EG
    passed_bonus_mg = PASSED_PAWN_BONUS_MG
    passed_bonus_eg = PASSED_PAWN_BONUS_EG
    
    # track piece squares for later heuristics
    white_king_square = -1
    black_king_square = -1
    white_rook_squares = np.zeros(2, dtype=np.int32)
    black_rook_squares = np.zeros(2, dtype=np.int32)
    white_rook_count = 0
    black_rook_count = 0
    white_queen_square = -1
    black_queen_square = -1
    white_bishop_squares = np.zeros(2, dtype=np.int32)
    black_bishop_squares = np.zeros(2, dtype=np.int32)
    white_bishop_count = 0
    black_bishop_count = 0
    
    # evaluate all pieces
    for piece_type in range(12):
        piece_bitboard = uint64(board_pieces[piece_type])
        if not piece_bitboard:
            continue
        
        piece_index = piece_type % 6
        is_white = (piece_type < 6)
        own_pieces = white_pieces if is_white else black_pieces
        enemy_pawn_attacks = black_pawn_attacks if is_white else white_pawn_attacks
        safe_squares_mask = (~own_pieces) & (~enemy_pawn_attacks)
        
        while piece_bitboard:
            # extract least significant bit (next piece)
            lsb = piece_bitboard & (~piece_bitboard + uint64(1))
            square = de_bruijn_lookup[((lsb * de_bruijn_magic) >> uint64(58))]
            
            # record king positions
            if piece_index == KING:
                if is_white:
                    white_king_square = square
                else:
                    black_king_square = square
            
            elif piece_index == ROOK:
                # record rook positions for battery detection
                if is_white and white_rook_count < 2:
                    white_rook_squares[white_rook_count] = square
                    white_rook_count += 1
                elif not is_white and black_rook_count < 2:
                    black_rook_squares[black_rook_count] = square
                    black_rook_count += 1

                file_idx = square % 8
                file_mask = FILE_MASKS[file_idx]
                
                # check for pawns on this file
                is_own_pawn = (own_pieces & file_mask) & (white_pawns if is_white else black_pawns)
                is_enemy_pawn = (enemy_pawn_attacks & file_mask) if is_white else (white_pawn_attacks & file_mask)

                has_own_pawns = (white_pawns if is_white else black_pawns) & file_mask
                has_enemy_pawns = (black_pawns if is_white else white_pawns) & file_mask
                
                if not has_own_pawns:
                    if not has_enemy_pawns:
                        # open file
                        score_mg += ROOK_OPEN_FILE_BONUS_MG if is_white else -ROOK_OPEN_FILE_BONUS_MG
                        score_eg += ROOK_OPEN_FILE_BONUS_EG if is_white else -ROOK_OPEN_FILE_BONUS_EG
                    else:
                        # semi-open file
                        score_mg += ROOK_SEMI_OPEN_FILE_BONUS_MG if is_white else -ROOK_SEMI_OPEN_FILE_BONUS_MG
                        score_eg += ROOK_SEMI_OPEN_FILE_BONUS_EG if is_white else -ROOK_SEMI_OPEN_FILE_BONUS_EG
            
            # record queen positions
            elif piece_index == QUEEN:
                if is_white:
                    white_queen_square = square
                else:
                    black_queen_square = square
            
            # record bishop positions
            elif piece_index == BISHOP:
                if is_white and white_bishop_count < 2:
                    white_bishop_squares[white_bishop_count] = square
                    white_bishop_count += 1
                elif not is_white and black_bishop_count < 2:
                    black_bishop_squares[black_bishop_count] = square
                    black_bishop_count += 1
            
            # add piece-square table values (includes material)
            score_mg += psqt_mg[piece_type, square]
            score_eg += psqt_eg[piece_type, square]
            
            ## Mobility Evaluation

            if is_opening and 1 <= piece_index <= 4:
                attack_bitboard = uint64(0)
                
                if piece_index == KNIGHT:
                    # knights use precomputed attack pattern
                    attack_bitboard = knight_attacks[square]
                
                elif piece_index == BISHOP:
                    # bishops move diagonally (directions 4 - 7)
                    for direction in range(4, 8):
                        ray = sliding_rays[square, direction]
                        blockers = ray & all_pieces
                        
                        if blockers:
                            # find first blocker in this direction
                            if direction >= 6:  # northwest or southwest (negative directions)
                                # find most significant bit (furthest blocker)
                                temp = blockers
                                temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                                temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                                blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                            else: # northeast or southeast (positive directions)
                                # find least significant bit (nearest blocker)
                                blocker = blockers & (~blockers + uint64(1))
                                blocker_square = de_bruijn_lookup[((blocker * de_bruijn_magic) >> uint64(58))]
                            
                            # remove squares beyond blocker
                            ray ^= sliding_rays[blocker_square, direction]
                        
                        attack_bitboard |= ray
                
                elif piece_index == ROOK:
                    # rooks move orthogonally (directions 0 - 3)
                    for direction in (0, 2):  # north, east (positive directions)
                        ray = sliding_rays[square, direction]
                        blockers = ray & all_pieces
                        
                        if blockers:
                            blocker = blockers & (~blockers + uint64(1))
                            blocker_square = de_bruijn_lookup[((blocker * de_bruijn_magic) >> uint64(58))]
                            ray ^= sliding_rays[blocker_square, direction]
                        
                        attack_bitboard |= ray
                    
                    for direction in (1, 3):  # south, west (negative directions)
                        ray = sliding_rays[square, direction]
                        blockers = ray & all_pieces
                        
                        if blockers:
                            temp = blockers
                            temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                            temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                            blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                            ray ^= sliding_rays[blocker_square, direction]
                        
                        attack_bitboard |= ray
                
                elif piece_index == QUEEN:
                    # queens move both diagonally and orthogonally (directions 0 - 7)
                    for direction in range(8):
                        ray = sliding_rays[square, direction]
                        blockers = ray & all_pieces
                        
                        if blockers:
                            if direction in (1, 3, 6, 7): # negative directions
                                temp = blockers
                                temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                                temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                                blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                            else: # positive directions
                                blocker = blockers & (~blockers + uint64(1))
                                blocker_square = de_bruijn_lookup[((blocker * de_bruijn_magic) >> uint64(58))]
                            
                            ray ^= sliding_rays[blocker_square, direction]
                        
                        attack_bitboard |= ray
                
                # count safe moves (not occupied by own pieces, not attacked by enemy pawns)
                safe_moves = attack_bitboard & safe_squares_mask
                mobility_count = popcount(safe_moves)
                
                if mobility_count == 0:
                    # trapped piece penalty
                    penalty = trapped_penalties[piece_index]
                    if is_white:
                        score_mg -= penalty
                        score_eg -= penalty
                    else:
                        score_mg += penalty
                        score_eg += penalty
                else:
                    # mobility bonus
                    bonus_mg = mobility_count * mobility_mg[piece_index]
                    bonus_eg = mobility_count * mobility_eg[piece_index]
                    
                    if is_white:
                        score_mg += bonus_mg
                        score_eg += bonus_eg
                    else:
                        score_mg -= bonus_mg
                        score_eg -= bonus_eg
            
            # clear processed bit
            piece_bitboard ^= lsb
    
    if white_bishop_count >= 2:
        score_mg += BISHOP_PAIR_BONUS_MG
        score_eg += BISHOP_PAIR_BONUS_EG
    if black_bishop_count >= 2:
        score_mg -= BISHOP_PAIR_BONUS_MG
        score_eg -= BISHOP_PAIR_BONUS_EG
    
    # doubled pawns penalty
    for file_index in range(8):
        file_mask = FILE_MASKS[file_index]
        white_pawns_on_file = popcount(white_pawns & file_mask)
        black_pawns_on_file = popcount(black_pawns & file_mask)
        
        if white_pawns_on_file > 1:
            penalty_mg = (white_pawns_on_file - 1) * DOUBLED_PAWN_PENALTY_MG
            penalty_eg = (white_pawns_on_file - 1) * DOUBLED_PAWN_PENALTY_EG
            score_mg -= penalty_mg
            score_eg -= penalty_eg
        
        if black_pawns_on_file > 1:
            penalty_mg = (black_pawns_on_file - 1) * DOUBLED_PAWN_PENALTY_MG
            penalty_eg = (black_pawns_on_file - 1) * DOUBLED_PAWN_PENALTY_EG
            score_mg += penalty_mg
            score_eg += penalty_eg
    
    # isolated pawns penalty (no friendly pawns on adjacent files)
    for file_index in range(8):
        file_mask = FILE_MASKS[file_index]
        adjacent_files_mask = uint64(0)
        
        if file_index > 0:
            adjacent_files_mask |= FILE_MASKS[file_index - 1]
        if file_index < 7:
            adjacent_files_mask |= FILE_MASKS[file_index + 1]
        
        # white isolated pawns
        if (white_pawns & file_mask) and not (white_pawns & adjacent_files_mask):
            score_mg -= ISOLATED_PAWN_PENALTY_MG
            score_eg -= ISOLATED_PAWN_PENALTY_EG
        
        # black isolated pawns
        if (black_pawns & file_mask) and not (black_pawns & adjacent_files_mask):
            score_mg += ISOLATED_PAWN_PENALTY_MG
            score_eg += ISOLATED_PAWN_PENALTY_EG
    
    # pawn islands penalty (groups of connected pawns)
    # count transitions from occupied to empty files
    white_pawn_islands = 0
    black_pawn_islands = 0
    prev_white = False
    prev_black = False
    
    for file_index in range(8):
        file_mask = FILE_MASKS[file_index]
        has_white = bool(white_pawns & file_mask)
        has_black = bool(black_pawns & file_mask)
        
        if has_white and not prev_white:
            white_pawn_islands += 1
        if has_black and not prev_black:
            black_pawn_islands += 1
        
        prev_white = has_white
        prev_black = has_black
    
    # penalty for multiple pawn islands (more islands => worse structure)
    if white_pawn_islands > 1:
        penalty_mg = (white_pawn_islands - 1) * PAWN_ISLAND_PENALTY_MG
        penalty_eg = (white_pawn_islands - 1) * PAWN_ISLAND_PENALTY_EG
        score_mg -= penalty_mg
        score_eg -= penalty_eg
    
    if black_pawn_islands > 1:
        penalty_mg = (black_pawn_islands - 1) * PAWN_ISLAND_PENALTY_MG
        penalty_eg = (black_pawn_islands - 1) * PAWN_ISLAND_PENALTY_EG
        score_mg += penalty_mg
        score_eg += penalty_eg

    ## Passed Pawns

    # White Passed Pawns
    temp_wp = white_pawns
    while temp_wp:
        lsb = temp_wp & (~temp_wp + uint64(1))
        sq = de_bruijn_lookup[((lsb * de_bruijn_magic) >> uint64(58))]
        rank = sq // 8
        file_idx = sq % 8
        
        if rank >= 3: # only check if sufficiently advanced
            ahead_mask = FILE_MASKS[file_idx]
            # add adjacent files
            if file_idx > 0: ahead_mask |= FILE_MASKS[file_idx - 1]
            if file_idx < 7: ahead_mask |= FILE_MASKS[file_idx + 1]
            
            rank_forward_mask = uint64(0)
            for r in range(rank + 1, 8):
                rank_forward_mask |= RANK_MASKS[r]
            
            ahead_mask &= rank_forward_mask
            
            if (black_pawns & ahead_mask) == 0:
                bonus_m = passed_bonus_mg[rank]
                bonus_e = passed_bonus_eg[rank]
                score_mg += bonus_m
                score_eg += bonus_e
        
        temp_wp ^= lsb

    # Black Passed Pawns
    temp_bp = black_pawns
    while temp_bp:
        lsb = temp_bp & (~temp_bp + uint64(1))
        sq = de_bruijn_lookup[((lsb * de_bruijn_magic) >> uint64(58))]
        rank = sq // 8
        file_idx = sq % 8
        
        if rank <= 4: 
            ahead_mask = FILE_MASKS[file_idx]
            if file_idx > 0: ahead_mask |= FILE_MASKS[file_idx - 1]
            if file_idx < 7: ahead_mask |= FILE_MASKS[file_idx + 1]
            
            rank_forward_mask = uint64(0)
            for r in range(0, rank):
                rank_forward_mask |= RANK_MASKS[r]
            
            ahead_mask &= rank_forward_mask
            
            if (white_pawns & ahead_mask) == 0:
                bonus_m = passed_bonus_mg[7 - rank]
                bonus_e = passed_bonus_eg[7 - rank]
                score_mg -= bonus_m
                score_eg -= bonus_e
        
        temp_bp ^= lsb
    
    ## King Safety
    if is_opening:
        # white king
        if white_king_square >= 0:
            king_file = white_king_square % 8
            king_rank = white_king_square // 8
            
            # pawn shield
            if king_file <= 2 or king_file >= 5:
                shield_count = 0
                
                for file_offset in range(-1, 2):
                    check_file = king_file + file_offset
                    if 0 <= check_file < 8:
                        for rank_offset in range(1, 3):
                            check_rank = king_rank + rank_offset
                            if check_rank < 8:
                                check_square = check_rank * 8 + check_file
                                if (white_pawns >> uint64(check_square)) & uint64(1):
                                    shield_count += 1
                
                score_mg += shield_count * PAWN_SHIELD_BONUS_MG
            
            # king zone attacks (IMPROVED)
            king_zone = king_attacks[white_king_square]
            enemy_attackers = 0
            
            for piece_type in range(7, 11):
                piece_bb = uint64(board_pieces[piece_type])
                
                while piece_bb:
                    lsb = piece_bb & (~piece_bb + uint64(1))
                    piece_square = de_bruijn_lookup[((lsb * de_bruijn_magic) >> uint64(58))]
                    piece_index = piece_type % 6
                    
                    attacks_zone = False
                    
                    if piece_index == KNIGHT:
                        if knight_attacks[piece_square] & king_zone:
                            attacks_zone = True
                    
                    elif piece_index == BISHOP:
                        # check diagonal rays
                        for direction in range(4, 8):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                # check if path is clear
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction >= 6:
                                        temp = blockers
                                        temp |= temp >> uint64(1); temp |= temp >> uint64(2); temp |= temp >> uint64(4)
                                        temp |= temp >> uint64(8); temp |= temp >> uint64(16); temp |= temp >> uint64(32)
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    # check if any king zone square is reached before blocker
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    elif piece_index == ROOK:
                        # check orthogonal rays
                        for direction in range(4):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction in (1, 3):
                                        temp = blockers
                                        temp |= temp >> uint64(1); temp |= temp >> uint64(2); temp |= temp >> uint64(4)
                                        temp |= temp >> uint64(8); temp |= temp >> uint64(16); temp |= temp >> uint64(32)
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    elif piece_index == QUEEN:
                        # check all 8 rays
                        for direction in range(8):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction in (1, 3, 6, 7):
                                        temp = blockers
                                        temp |= temp >> uint64(1); temp |= temp >> uint64(2); temp |= temp >> uint64(4)
                                        temp |= temp >> uint64(8); temp |= temp >> uint64(16); temp |= temp >> uint64(32)
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    if attacks_zone:
                        enemy_attackers += 1
                    
                    piece_bb ^= lsb
            
            score_mg -= enemy_attackers * KING_ZONE_ATTACK_PENALTY
        
        # black king
        if black_king_square >= 0:
            king_file = black_king_square % 8
            king_rank = black_king_square // 8
            
            if king_file <= 2 or king_file >= 5:
                shield_count = 0
                
                for file_offset in range(-1, 2):
                    check_file = king_file + file_offset
                    if 0 <= check_file < 8:
                        for rank_offset in range(1, 3):
                            check_rank = king_rank - rank_offset
                            if check_rank >= 0:
                                check_square = check_rank * 8 + check_file
                                if (black_pawns >> uint64(check_square)) & uint64(1):
                                    shield_count += 1
                
                score_mg -= shield_count * PAWN_SHIELD_BONUS_MG
            
            king_zone = king_attacks[black_king_square]
            enemy_attackers = 0
            
            for piece_type in range(1, 5):
                piece_bb = uint64(board_pieces[piece_type])
                
                while piece_bb:
                    lsb = piece_bb & (~piece_bb + uint64(1))
                    piece_square = de_bruijn_lookup[((lsb * de_bruijn_magic) >> uint64(58))]
                    piece_index = piece_type % 6
                    
                    attacks_zone = False
                    
                    if piece_index == KNIGHT:
                        if knight_attacks[piece_square] & king_zone:
                            attacks_zone = True
                    
                    elif piece_index == BISHOP:
                        for direction in range(4, 8):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction >= 6:
                                        temp = blockers
                                        temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                                        temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    elif piece_index == ROOK:
                        for direction in range(4):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction in (1, 3):
                                        temp = blockers
                                        temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                                        temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    elif piece_index == QUEEN:
                        for direction in range(8):
                            ray = sliding_rays[piece_square, direction]
                            if ray & king_zone:
                                blockers = ray & all_pieces
                                if blockers:
                                    if direction in (1, 3, 6, 7):
                                        temp = blockers
                                        temp |= temp >> 1; temp |= temp >> 2; temp |= temp >> 4
                                        temp |= temp >> 8; temp |= temp >> 16; temp |= temp >> 32
                                        first_blocker_square = de_bruijn_lookup[((temp * de_bruijn_magic) >> uint64(58))]
                                    else:
                                        first_blocker = blockers & (~blockers + uint64(1))
                                        first_blocker_square = de_bruijn_lookup[((first_blocker * de_bruijn_magic) >> uint64(58))]
                                    
                                    clear_ray = ray ^ sliding_rays[first_blocker_square, direction]
                                    if clear_ray & king_zone:
                                        attacks_zone = True
                                        break
                                else:
                                    attacks_zone = True
                                    break
                    
                    if attacks_zone:
                        enemy_attackers += 1
                    
                    piece_bb ^= lsb
            
            score_mg += enemy_attackers * KING_ZONE_ATTACK_PENALTY
    
    ## Piece Batteries
    
    # rook battery (two rooks on same file or rank)
    if white_rook_count == 2:
        rook1_file = white_rook_squares[0] % 8
        rook1_rank = white_rook_squares[0] // 8
        rook2_file = white_rook_squares[1] % 8
        rook2_rank = white_rook_squares[1] // 8
        
        if rook1_file == rook2_file or rook1_rank == rook2_rank:
            score_mg += ROOK_BATTERY_BONUS_MG
            score_eg += ROOK_BATTERY_BONUS_EG
    
    if black_rook_count == 2:
        rook1_file = black_rook_squares[0] % 8
        rook1_rank = black_rook_squares[0] // 8
        rook2_file = black_rook_squares[1] % 8
        rook2_rank = black_rook_squares[1] // 8
        
        if rook1_file == rook2_file or rook1_rank == rook2_rank:
            score_mg -= ROOK_BATTERY_BONUS_MG
            score_eg -= ROOK_BATTERY_BONUS_EG
    
    # rook + queen battery (same file or rank)
    if white_queen_square >= 0 and white_rook_count > 0:
        queen_file = white_queen_square % 8
        queen_rank = white_queen_square // 8
        
        for i in range(white_rook_count):
            rook_file = white_rook_squares[i] % 8
            rook_rank = white_rook_squares[i] // 8
            
            if queen_file == rook_file or queen_rank == rook_rank:
                score_mg += ROOK_QUEEN_BATTERY_BONUS_MG
                score_eg += ROOK_QUEEN_BATTERY_BONUS_EG
                break
    
    if black_queen_square >= 0 and black_rook_count > 0:
        queen_file = black_queen_square % 8
        queen_rank = black_queen_square // 8
        
        for i in range(black_rook_count):
            rook_file = black_rook_squares[i] % 8
            rook_rank = black_rook_squares[i] // 8
            
            if queen_file == rook_file or queen_rank == rook_rank:
                score_mg -= ROOK_QUEEN_BATTERY_BONUS_MG
                score_eg -= ROOK_QUEEN_BATTERY_BONUS_EG
                break
    
    # queen + bishop battery (same diagonal)
    if white_queen_square >= 0 and white_bishop_count > 0:
        queen_file = white_queen_square % 8
        queen_rank = white_queen_square // 8
        
        for i in range(white_bishop_count):
            bishop_file = white_bishop_squares[i] % 8
            bishop_rank = white_bishop_squares[i] // 8
            
            # same diagonal if |file difference| == |rank_difference|
            file_diff = abs(queen_file - bishop_file)
            rank_diff = abs(queen_rank - bishop_rank)
            
            if file_diff == rank_diff and file_diff > 0:
                score_mg += QUEEN_BISHOP_BATTERY_BONUS_MG
                score_eg += QUEEN_BISHOP_BATTERY_BONUS_EG
                break
    
    if black_queen_square >= 0 and black_bishop_count > 0:
        queen_file = black_queen_square % 8
        queen_rank = black_queen_square // 8
        
        for i in range(black_bishop_count):
            bishop_file = black_bishop_squares[i] % 8
            bishop_rank = black_bishop_squares[i] // 8
            
            file_diff = abs(queen_file - bishop_file)
            rank_diff = abs(queen_rank - bishop_rank)
            
            if file_diff == rank_diff and file_diff > 0:
                score_mg -= QUEEN_BISHOP_BATTERY_BONUS_MG
                score_eg -= QUEEN_BISHOP_BATTERY_BONUS_EG
                break
    
    # Mop-up Evaluation

    if is_endgame:
        # calculate material difference
        white_material = 0
        black_material = 0
        
        for piece_idx in range(KING + 1):
            white_material += popcount(uint64(board_pieces[piece_idx])) * MATERIAL_VALUE_EG[piece_idx]
            black_material += popcount(uint64(board_pieces[piece_idx + 6])) * MATERIAL_VALUE_EG[piece_idx]
        
        material_difference = abs(white_material - black_material)
        
        # only apply mop-up if there's significant material advantage
        if material_difference >= MATERIAL_VALUE_EG[ROOK] - 10: # roughly equivalent to a rook
            winning_side_is_white = white_material > black_material
            
            if winning_side_is_white:
                # push black king to edge
                if black_king_square >= 0:
                    edge_bonus = (3 - edge_distance(black_king_square)) * KING_EDGE_DISTANCE_BONUS
                    score_eg += edge_bonus
                
                # bring white king closer to black king
                if white_king_square >= 0 and black_king_square >= 0:
                    distance = chebyshev_distance(white_king_square, black_king_square)
                    proximity_bonus = (7 - distance) * KING_PROXIMITY_BONUS
                    score_eg += proximity_bonus
            else:
                # push white king to edge
                if white_king_square >= 0:
                    edge_bonus = (3 - edge_distance(white_king_square)) * KING_EDGE_DISTANCE_BONUS
                    score_eg -= edge_bonus
                
                # bring black king closer to white king
                if white_king_square >= 0 and black_king_square >= 0:
                    distance = chebyshev_distance(white_king_square, black_king_square)
                    proximity_bonus = (7 - distance) * KING_PROXIMITY_BONUS
                    score_eg -= proximity_bonus

    # Tapered Evaluation
    final_score = ((score_mg * game_phase) + (score_eg * (MAX_PHASE - game_phase))) // MAX_PHASE
    
    return -final_score if side_to_move == 1 else final_score
