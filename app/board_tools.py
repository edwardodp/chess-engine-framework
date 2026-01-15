from numba import njit, int64, int32

@njit(int32(int64[:], int32))
def get_piece(pieces, sq):
    """
    Returns the piece ID at a given square (0-63).
    Positive = White, Negative = Black, 0 = Empty.
    IDs: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
    """
    bit_mask = 1 << sq
    
    # Check White Pieces (Indices 0-5)
    if pieces[0] & bit_mask: return 1  # White Pawn
    if pieces[1] & bit_mask: return 2  # White Knight
    if pieces[2] & bit_mask: return 3  # White Bishop
    if pieces[3] & bit_mask: return 4  # White Rook
    if pieces[4] & bit_mask: return 5  # White Queen
    if pieces[5] & bit_mask: return 6  # White King

    # Check Black Pieces (Indices 6-11)
    if pieces[6] & bit_mask: return -1 # Black Pawn
    if pieces[7] & bit_mask: return -2 # Black Knight
    if pieces[8] & bit_mask: return -3 # Black Bishop
    if pieces[9] & bit_mask: return -4 # Black Rook
    if pieces[10] & bit_mask: return -5 # Black Queen
    if pieces[11] & bit_mask: return -6 # Black King
    
    return 0 # Empty
