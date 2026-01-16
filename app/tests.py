from evaluation import evaluation_function
import numpy as np

# Position 0: Starting positions
# All pieces on home squares
starting_position = np.array([65280, 66, 36, 129, 8, 16, 71776119061217280, 4755801206503243776, 2594073385365405696, -9151314442816847872, 576460752303423488, 1152921504606846976], dtype=np.int64)
starting_occupancy = np.array([65535, -281474976710656, -281474976645121], dtype=np.int64)
evaluation_function(starting_position, starting_occupancy, 0)
