import sys
import os
import ctypes
import numpy as np
from numba import cfunc, types, carray
import chess
import time
from functools import partial
print = partial(print, flush=True)

def compile_wrapper(evaluation_function):
    @cfunc(types.int32(types.CPointer(types.int64), types.CPointer(types.int64), types.uint32))
    def wrapper(board_pieces_ptr, board_occupancy_ptr, side_to_move):

        board_pieces = carray(board_pieces_ptr, (12,), np.int64)
        board_occupancy = carray(board_occupancy_ptr, (3,), np.int64)
        
        score = evaluation_function(board_pieces, board_occupancy, side_to_move)
        
        return np.int32(score)
    return wrapper

# the c structures in python
class SearchContext(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('depth', ctypes.c_int32),
        ('padding', ctypes.c_int32),
        ('callback_ptr', ctypes.c_void_p) # pointer to the python function
    ]

# context manager to capture c output and parse
class OutputCapturer:
    def __init__(self):
        self.captured = ''

    def __enter__(self):
        # flush buffer so we don't mix old data
        sys.stdout.flush()
        
        # save original stdout
        self.save_stdout = os.dup(1)
        
        # create a read and write pipe
        self.pipe_r, self.pipe_w = os.pipe()
        
        # redirect stdout to the write end of the pipe
        # anything the c code prints goes into the pipe
        os.dup2(self.pipe_w, 1)
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # close the write end of the pipe
        os.close(self.pipe_w)
        
        # restore stdout so python printing works again
        os.dup2(self.save_stdout, 1)
        os.close(self.save_stdout)
        
        # read everything from the pipe
        with os.fdopen(self.pipe_r) as pipe_reader:
            self.captured = pipe_reader.read()

    def get_lines(self):
        return self.captured.splitlines()

class Engine:
    def __init__(self, evaluation_function, depth):
        base_address = self._init_dll()._handle # base address of the loaded module in memory
        self._init_tables(base_address)

        self.capturer = OutputCapturer()
        
        BRAIN_OFFSET = 0x4EE0 # magic offset for search function (windows)
        
        # calculate the hidden function address
        self.brain_address = base_address + BRAIN_OFFSET

        # define the function signature
        # 'short* (__fastcall *Brain)(short* out, uint64* board, Context* ctx, int* stats)'
        self.BrainFuncType = ctypes.CFUNCTYPE(
            ctypes.POINTER(ctypes.c_int16), # return type (short*)
            ctypes.POINTER(ctypes.c_int16), # arg 1: output move
            ctypes.POINTER(ctypes.c_uint64), # arg 2: board array
            ctypes.POINTER(SearchContext), # arg 3: context struct
            ctypes.POINTER(ctypes.c_int32) # arg 4: stats array
        )

        self.evaluation = compile_wrapper(evaluation_function)
        self.depth = depth

    def _init_dll(self):
        lib = 'ChessLib.dll'
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(curr_dir, '..', 'bindings', lib)
        
        try: return ctypes.CDLL(lib_path)
        except OSError: print(f'info string (error): couldn\'t load {lib}')

    def _init_tables(self, base_address):
        INIT_OFFSET = 0x2100 # magic offset for precomputed move tables (windows)
        # needed so engine can make non-pawn moves
        try:
            ctypes.CFUNCTYPE(None)(base_address + INIT_OFFSET)()
            print(f'info string: loaded precomputed tables')
        except Exception as e:
            print(f'info string (error): initialisation failed')

    @staticmethod
    def move_to_str(move: int) -> str:
        if move == 0: return '0000'

        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        files = 'abcdefgh'
        move_str = f'{files[from_sq % 8]}{from_sq // 8 + 1}{files[to_sq % 8]}{to_sq // 8 + 1}'

        if flags & 0x8: # if the 4th bit is set, it's a promotion
            kind = flags & 0x3 # bottom 2 bits
            if kind == 0: move_str += 'n' # 00 = knight
            elif kind == 1: move_str += 'b' # 01 = bishop
            elif kind == 2: move_str += 'r' # 10 = rook
            elif kind == 3: move_str += 'q' # 11 = queen

        return move_str

    @staticmethod
    def get_board(board: chess.Board):
        arr = (ctypes.c_uint64 * 32)()

        # pieces
        for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
            arr[i] = board.pieces_mask(pt, chess.WHITE)
            arr[i + 6] = board.pieces_mask(pt, chess.BLACK)

        # occupancy 12-14
        arr[12] = board.occupied_co[chess.WHITE]
        arr[13] = board.occupied_co[chess.BLACK]
        arr[14] = board.occupied

        # state 15-16
        ep = (board.ep_square + 1) if board.ep_square is not None else 0
        arr[15] = (0 if board.turn == chess.WHITE else 1) | (ep << 32)
        
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE): castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK): castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
        arr[16] = castling

        return arr

    def get_best_move(self, board: chess.Board):
        run_search = self.BrainFuncType(self.brain_address)
        
        c_board = Engine.get_board(board)

        # setup call arguments
        best_move_out = ctypes.c_int16(0)
        stats_out = (ctypes.c_int32 * 4)()
        
        ctx = SearchContext()
        ctx.depth = self.depth
        ctx.padding = 0
        ctx.callback_ptr = ctypes.cast(self.evaluation.address, ctypes.c_void_p)

        print(f'info string: {board.fen()} @ depth {self.depth}')

        start_time = time.perf_counter()
        with self.capturer:
            run_search(ctypes.byref(best_move_out), c_board, ctypes.byref(ctx), stats_out)
        end_time = time.perf_counter()
        duration = int((end_time - start_time) * 1000) # in milliseconds
        
        # parse the captured lines
        for line in self.capturer.get_lines():
            # minimal parsing for speed
            # format: 'Info: Depth 5 Score: 500'
            parts = line.split() 
            print(f'info depth {parts[2]} score cp {parts[4]}')
        print(f'info time {duration}')
        
        return Engine.move_to_str(best_move_out.value)