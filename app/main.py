import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluation_function
import ctypes
import sys 
import os

# return 32 bit signed int, 1 parameter - pointer of type 64 bit unsigned int (pointing to C board data array) another 64 bit unsigned int to the occupancy bitboards. 3 ints which say which squares have white peices, black and any colour peice. unsigned 32 bit which is number of moves.
@cfunc(types.int32(types.CPointer(types.uint64), types.CPointer(types.uint64), types.uint32))
# wrapper evalutation function
def _evalutation_function(board_pieces_ptr, board_occupancy_ptr, move_count):

    # convert c array to numpy array
    board_pieces_data = carray(board_pieces_ptr, (12,), np.int64)
    board_occupancy_data = carray(board_occupancy_ptr, (3,), np.int64)
    
    evaluation = np.int32(evaluation_function(board_pieces_data, board_occupancy_data, move_count))
    
    return evaluation

curr_dir = os.path.dirname(os.path.abspath(__file__))

# check OS to fetch corresponding library
if sys.platform == "win32":
    dll_path = os.path.join(curr_dir, "..", "bindings","example.dll")
else:
    dll_path = os.path.join(curr_dir, "..", "bindings","example.so")

init_path = os.path.abspath(dll_path)
init = ctypes.CDLL(init_path)

# define parameter type and initialise engine
init.startEngine.argtypes = [ctypes.c_void_p]
init.startEngine(_evalutation_function.address)

