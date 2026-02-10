from numba import njit, int64, int32


"""Piece IDs"""
WHITE_PAWN = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_ROOK = 4
WHITE_QUEEN = 5
WHITE_KING = 6

BLACK_PAWN = -1
BLACK_KNIGHT = -2
BLACK_BISHOP = -3
BLACK_ROOK = -4
BLACK_QUEEN = -5
BLACK_KING = -6


@njit(int32(int64[:], int32))
def get_piece(board_pieces, sq):
    """
    Returns the Piece ID at a specific square (0-63).
    Returns 0 if empty.
    """
    mask = 1 << sq

    if board_pieces[0] & mask: return 1
    if board_pieces[1] & mask: return 2
    if board_pieces[2] & mask: return 3
    if board_pieces[3] & mask: return 4
    if board_pieces[4] & mask: return 5
    if board_pieces[5] & mask: return 6

    if board_pieces[6] & mask: return -1
    if board_pieces[7] & mask: return -2
    if board_pieces[8] & mask: return -3
    if board_pieces[9] & mask: return -4
    if board_pieces[10] & mask: return -5
    if board_pieces[11] & mask: return -6

    return 0

@njit(int32(int64[:], int32, int32))
def check_square(board_occupancy, sq, color_idx):
    """
    Checks if a specific color occupies a square
    color_idx: 0 for White, 1 for Black, 2 for B&W
    """
    mask = 1 << sq
    if board_occupancy[color_idx] & mask:
        return 1
    return 0

@njit(int32(int64))
def count_bits(bitboard):
    """
    Counts the number of pieces in a bitboard
    """
    count = 0
    while bitboard:
        bitboard &= bitboard - 1
        count += 1
    return count
