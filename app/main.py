import sys
import os
import ctypes
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluation_function


COMPETITION_DEPTH = 5;

# Returns: int32
# Args: int64* (pieces), int64* (occupancy), uint32 (move_count)
@cfunc(types.int32(types.CPointer(types.int64), types.CPointer(types.int64), types.uint32))
def evaluation_wrapper(board_pieces_ptr, board_occupancy_ptr, move_count):
    board_pieces = carray(board_pieces_ptr, (12,), np.int64)
    board_occupancy = carray(board_occupancy_ptr, (3,), np.int64)
    
    score = evaluation_function(board_pieces, board_occupancy, move_count)
    
    return np.int32(score)

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = "ChessLib.dll" if sys.platform == "win32" else "libChessLib.so"
    lib_path = os.path.join(curr_dir, "..", "bindings", lib_name)
    lib_path = os.path.abspath(lib_path)

    if not os.path.exists(lib_path):
        print(f"[Error] Shared library not found at: {lib_path}")
        print("Please compile the C++ core and move the output to the 'bindings/' folder.")
        sys.exit(1)

    try:
        chess_lib = ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"[Error] Failed to load library: {e}")
        sys.exit(1)

    chess_lib.startEngine.argtypes = [ctypes.c_void_p, ctypes.c_int]
    
    print(f"--- Chess Engine Interface Loaded ---")
    print(f"Library: {lib_path}")
    
    engine_callback = evaluation_wrapper

    print("Starting Engine...")
    chess_lib.startEngine(engine_callback.address, COMPETITION_DEPTH)

if __name__ == "__main__":
    main()
