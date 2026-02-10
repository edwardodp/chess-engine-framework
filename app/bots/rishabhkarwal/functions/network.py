import board_tools as bt
import numpy as np
from numba import njit, int64, int32, uint32

# Constants
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1

# scaling factor from training (1 / 0.004)
# needs to be adjusted if changed in training script
SCALE_FACTOR = 250.0 

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    hidden_state = np.copy(bt.B1) 
    
    for piece_idx in range(12):
        bitboard = board_pieces[piece_idx]
        
        while bitboard != 0:
            sq = bt.bit_scan_forward(bitboard)
            bitboard = bt.pop_lsb(bitboard)
            
            feature_index = (piece_idx * 64) + sq

            start_idx = feature_index * 32
            end_idx = start_idx + 32

            hidden_state += bt.W1_FLAT[start_idx : end_idx]
            
    for i in range(32):
        if hidden_state[i] < 0.0:
            hidden_state[i] = 0.0
            
    raw_output = 0.0
    for i in range(32):
        raw_output += hidden_state[i] * bt.W2[i]

    raw_output += bt.B2[0]
    

    if raw_output > 100.0: raw_output = 100.0
    if raw_output < -100.0: raw_output = -100.0
    
    score_cp = int32(raw_output * SCALE_FACTOR)
    
    if side_to_move == BLACK_TO_MOVE:
        return -score_cp
        
    return score_cp