from numba import njit, int32, int64, uint32

# We force a return of 5 no matter what the board looks like.
@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    return 5
