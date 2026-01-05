""" The main entry point and the 'backend' of sorts """
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluationFunction
import ctypes
import sys 
import os

# 12 64 bit integers, each integer is a type of piece e.g white pawn, black pawn. and each bit represents a cell on board where 1 means it is on that cell. a1, b1 , ... h1, a2, ... , h2
# white are first 6, pawn knight bishop rook queen king, black are last 6 with same order

# return 32 bit signed int, 1 parameter - pointer of type 8 bit signed int (pointing to C board data array)
@cfunc(types.int32(types.CPointer(types.int8)))
# wrapper evalutation function
def _evalutation_function(dataPtr):

    # convert c array to numpy 2d (8x8) array
    boardData = np.reshape(carray(dataPtr, (64,), np.int8), (8,8))
    
    evaluation = np.int32(evaluationFunction(boardData))    
    
    return 0

# check OS to fetch corresponding library
if sys.platform == "win32":
    initPath = os.path.abspath("example.dll")
    init = ctypes.CDLL(libPath)
else:
    libPath = os.path.abspath("example.so")
    init = ctypes.CDLL(libPath)

# define parameter type and initialise engine
init.startEngine.argtypes = [ctypes.c_void_p]
init.startEngine(_evalutation_function.address)

