""" The main entry point and the 'backend' of sorts """
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluationFunction

# return 32 bit signed int, 1 parameter - pointer of type 8 bit signed int (pointing to C board data array)
@cfunc(types.int32(types.CPointer(types.int8)))
# wrapper evalutation function
def _evalutation_function(dataPtr):

    # convert c array to numpy 2d (8x8) array
    boardData = np.reshape(carray(dataPtr, (4,), np.int8), (2,2))
    
    evaluation = np.int32(evaluationFunction(boardData))    
    
    return evaluation



