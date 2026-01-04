""" The main entry point and the 'backend' of sorts """
import numpy as np
from numba import cfunc, types, carray, njit

@cfunc(types.int32(types.CPointer(types.int8)))
def _evalutation_function(dataPtr):
    

