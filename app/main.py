""" The main entry point and the 'backend' of sorts """
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluationFunction
import ctypes
import sys 
import os

# return 32 bit signed int, 1 parameter - pointer of type 64 bit unsigned int (pointing to C board data array)
@cfunc(types.int32(types.CPointer(types.uint64)))
# wrapper evalutation function
def _evalutation_function(dataPtr):

    # convert c array to numpy array
    boardData = carray(dataPtr, (12,), np.uint64)
    
    evaluation = np.int32(evaluationFunction(boardData))
    
    return evaluation

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

