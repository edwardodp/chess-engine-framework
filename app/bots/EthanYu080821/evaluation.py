from itertools import count
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

"""
Turn Constants
"""
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1


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




# ======= CODE======
PHASE_MAX = 24
CENTER_SQUARES = (27, 28, 35, 36)

WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1


# ================= BITBOARD =================
def bitboard_to_squares(bitboard):
    squares = []
    for sq in range(64):
        if (bitboard >> sq) & 1:
            squares.append(sq)
    return squares


# ================= KNIGHTS =================
def knight_moves(sq):
    moves = []
    rank = sq // 8
    file = sq % 8
    offsets = [(-2,-1),(-2,1),(-1,-2),(-1,2),
               (1,-2),(1,2),(2,-1),(2,1)]
    for dr, df in offsets:
        r = rank + dr
        f = file + df
        if 0 <= r < 8 and 0 <= f < 8:
            moves.append(r*8 + f)
    return moves


def knight_activity(knight_bb):
    score = 0
    for sq in bitboard_to_squares(knight_bb):
        score += len(knight_moves(sq))
    return score


# ================= Attacking Pieces =================
def bishop_attacks(sq, occupancy):
    attacks = []
    for d in (7, 9, -7, -9):
        t = sq + d
        while 0 <= t < 64 and abs((t % 8) - ((t - d) % 8)) == 1:
            attacks.append(t)
            if (occupancy >> t) & 1:
                break
            t += d
    return attacks


def rook_attacks(sq, occupancy):
    attacks = []
    for d in (1, -1, 8, -8):
        t = sq + d
        while 0 <= t < 64 and (d in (8,-8) or t//8 == (t-d)//8):
            attacks.append(t)
            if (occupancy >> t) & 1:
                break
            t += d
    return attacks


def queen_attacks(sq, occupancy):
    return bishop_attacks(sq, occupancy) + rook_attacks(sq, occupancy)


def bishop_activity(bb, occ):
    score = 0
    for sq in bitboard_to_squares(bb):
        score += len(bishop_attacks(sq, occ))
    return score


def rook_activity(bb, occ):
    score = 0
    for sq in bitboard_to_squares(bb):
        score += len(rook_attacks(sq, occ))
    return score


# ================= PAWNS =================
def pawn_attacks(sq, side):
    attacks = []
    rank = sq // 8
    file = sq % 8
    direction = 1 if side == WHITE_TO_MOVE else -1
    for df in (-1, 1):
        r = rank + direction
        f = file + df
        if 0 <= r < 8 and 0 <= f < 8:
            attacks.append(r*8 + f)
    return attacks


def column_count(pawn_bb, file):
    count = 0
    for rank in range(8):
        if (pawn_bb >> (rank*8 + file)) & 1:
            count += 1
    return count


def passed_pawns(pawns, enemy_pawns, side):
    passed = []
    direction = 1 if side == WHITE_TO_MOVE else -1
    for sq in range(64):
        if not (pawns >> sq) & 1:
            continue
        rank, file = divmod(sq, 8)
        is_passed = True
        r = rank + direction
        while 0 <= r < 8:
            for f in (file-1, file, file+1):
                if 0 <= f < 8:
                    if (enemy_pawns >> (r*8 + f)) & 1:
                        is_passed = False
            r += direction
        if is_passed:
            passed.append(sq)
    return passed


def isolated_pawns(pawns):
    isolated = []
    for sq in range(64):
        if not (pawns >> sq) & 1:
            continue
        file = sq % 8
        has_neighbor = False
        for f in (file-1, file+1):
            if 0 <= f < 8:
                for r in range(8):
                    if (pawns >> (r*8 + f)) & 1:
                        has_neighbor = True
        if not has_neighbor:
            isolated.append(sq)
    return isolated


# ================= ATTACK MAP =================
def attacked_squares(board_pieces, side):
    attacked = [0]*64
    occupancy = 0
    for bb in board_pieces:
        occupancy |= bb

    pieces = [0,1,2,3,4] if side == WHITE_TO_MOVE else [6,7,8,9,10]

    for idx in pieces:
        for sq in bitboard_to_squares(board_pieces[idx]):
            if idx in (1,7):
                for t in knight_moves(sq):
                    attacked[t] = 1
            elif idx in (2,8):
                for t in bishop_attacks(sq, occupancy):
                    attacked[t] = 1
            elif idx in (3,9):
                for t in rook_attacks(sq, occupancy):
                    attacked[t] = 1
            elif idx in (4,10):
                for t in queen_attacks(sq, occupancy):
                    attacked[t] = 1
            elif idx in (0,6):
                for t in pawn_attacks(sq, side):
                    attacked[t] = 1
    return attacked


def hanging_piece_score(board_pieces):
    score = 0
    white_att = attacked_squares(board_pieces, WHITE_TO_MOVE)
    black_att = attacked_squares(board_pieces, BLACK_TO_MOVE)

    VALUES = {0:20,1:60,2:60,3:100,4:180}

    for i in range(5):
        for sq in bitboard_to_squares(board_pieces[i]):
            if black_att[sq] and not white_att[sq]:
                score -= VALUES[i]

    for i in range(5):
        for sq in bitboard_to_squares(board_pieces[i+6]):
            if white_att[sq] and not black_att[sq]:
                score += VALUES[i]

    return score


# ================= Evaluation Function=================
def evaluation_function(board_pieces, board_occupancy, side_to_move):

    mg_score = 0
    eg_score = 0

    # -------- MATERIAL COUNTS --------
    PAWN_WHITE   = bt.popcount(board_pieces[0])
    KNIGHT_WHITE = bt.popcount(board_pieces[1])
    BISHOP_WHITE = bt.popcount(board_pieces[2])
    ROOK_WHITE   = bt.popcount(board_pieces[3])
    QUEEN_WHITE  = bt.popcount(board_pieces[4])

    PAWN_BLACK   = bt.popcount(board_pieces[6])
    KNIGHT_BLACK = bt.popcount(board_pieces[7])
    BISHOP_BLACK = bt.popcount(board_pieces[8])
    ROOK_BLACK   = bt.popcount(board_pieces[9])
    QUEEN_BLACK  = bt.popcount(board_pieces[10])

    # -------- PHASE --------
    phase = (
        (KNIGHT_WHITE + KNIGHT_BLACK) +
        (BISHOP_WHITE + BISHOP_BLACK) +
        2*(ROOK_WHITE + ROOK_BLACK) +
        4*(QUEEN_WHITE + QUEEN_BLACK)
    )
    phase = min(phase, PHASE_MAX)

    # -------- MATERIAL --------
    material = (
        100*(PAWN_WHITE - PAWN_BLACK) +
        320*(KNIGHT_WHITE - KNIGHT_BLACK) +
        330*(BISHOP_WHITE - BISHOP_BLACK) +
        500*(ROOK_WHITE - ROOK_BLACK) +
        900*(QUEEN_WHITE - QUEEN_BLACK)
    )
    mg_score += material
    eg_score += material

    # -------- CENTER --------
    for sq in CENTER_SQUARES:
        piece = bt.get_piece(board_pieces, sq)
        if piece:
            mg_score += 20 if piece <= bt.WHITE_KING else -20

    # -------- PAWNS --------
    for f in range(8):
        if column_count(board_pieces[0], f) > 1: mg_score -= 10
        if column_count(board_pieces[6], f) > 1: mg_score += 10

    for sq in passed_pawns(board_pieces[0], board_pieces[6], WHITE_TO_MOVE): eg_score += 20
    for sq in passed_pawns(board_pieces[6], board_pieces[0], BLACK_TO_MOVE): eg_score -= 20

    for sq in isolated_pawns(board_pieces[0]): mg_score -= 10
    for sq in isolated_pawns(board_pieces[6]): mg_score += 10

    # -------- ACTIVITY --------
    mg_score += 5 * knight_activity(board_pieces[1])
    mg_score -= 5 * knight_activity(board_pieces[7])

    mg_score += 5 * bishop_activity(board_pieces[2], board_occupancy)
    mg_score -= 5 * bishop_activity(board_pieces[8], board_occupancy)

    mg_score += 5 * rook_activity(board_pieces[3], board_occupancy)
    mg_score -= 5 * rook_activity(board_pieces[9], board_occupancy)

    # -------- HANGING --------
    mg_score += hanging_piece_score(board_pieces)

    # -------- FINAL Score--------
    score = (mg_score*phase + eg_score*(PHASE_MAX-phase)) // PHASE_MAX
    return score if side_to_move == WHITE_TO_MOVE else -score
