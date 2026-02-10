import numpy as np
import base64
import math
from numba import njit, int64, int32, float32


WEIGHTS_ENCODED = ""

def decode_weights():
    """
    Decodes the Base64 string into Numpy arrays
    """
    clean_str = WEIGHTS_ENCODED.strip()


    try:
        binary_data = base64.b64decode(clean_str)
   
        if len(binary_data) % 4 != 0:
            binary_data = binary_data[:len(binary_data) - (len(binary_data) % 4)]

        all_weights = np.frombuffer(binary_data, dtype=np.float32)
        
        offset = 0
        
        # Layer 1 Weights
        w1_size = 768 * 32
        if len(all_weights) < w1_size: raise ValueError("Data too short for W1")
        w1 = all_weights[offset : offset + w1_size]
        offset += w1_size
        
        # Layer 1 Bias
        b1_size = 32
        if len(all_weights) < offset + b1_size: raise ValueError("Data too short for B1")
        b1 = all_weights[offset : offset + b1_size]
        offset += b1_size
        
        # Layer 2 Weights
        w2_size = 32
        if len(all_weights) < offset + w2_size: raise ValueError("Data too short for W2")
        w2 = all_weights[offset : offset + w2_size]
        offset += w2_size
        
        # Layer 2 Bias
        if len(all_weights) < offset + 1: raise ValueError("Data too short for B2")
        b2 = all_weights[offset : offset + 1]
        
        return w1, b1, w2, b2

    except Exception as e:
        print(e)


# W1_FLAT, B1, W2, B2 = decode_weights()


@njit(int32(int64))
def bit_scan_forward(bitboard):
    if bitboard == 0:
        return -1

    lsb = bitboard & -bitboard
    
    return int32(math.log2(float(lsb)))

@njit(int64(int64))
def pop_lsb(bitboard):
    return bitboard & (bitboard - 1)


# Piece IDs
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