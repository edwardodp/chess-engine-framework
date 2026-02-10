import sys
import chess
from driver import Engine
from functools import partial
import importlib

NAME = 'Engine'
AUTHOR = 'Rish'
COMPETITION_DEPTH = 5

print = partial(print, flush=True)

def uci_loop(evaluation_function):
    engine = None
    board = chess.Board()

    # disable buffering to ensure GUI get messages immediately
    sys.stdout.reconfigure(line_buffering=True)

    while True:
        try:
            command_line = input().strip()
        except EOFError:
            break

        if not command_line:
            continue

        parts = command_line.split()
        cmd = parts[0]

        if cmd == 'uci':
            print(f'id name {NAME}')
            print(f'id author {AUTHOR}')
            print('uciok')

        elif cmd == 'isready':
            if engine is None: engine = Engine(evaluation_function, depth = COMPETITION_DEPTH)
            print('readyok')

        elif cmd == 'ucinewgame':
            board = chess.Board()

        elif cmd == 'position':
            if 'startpos' in parts:
                board = chess.Board()
                moves_idx = parts.index('moves') + 1 if 'moves' in parts else -1
            elif 'fen' in parts:
                # reconstruct FEN from parts
                fen_parts = []
                moves_idx = -1
                for i in range(parts.index('fen') + 1, len(parts)):
                    if parts[i] == 'moves':
                        moves_idx = i + 1
                        break
                    fen_parts.append(parts[i])
                board = chess.Board(' '.join(fen_parts))
            
            # apply moves if present
            if moves_idx > 0 and moves_idx < len(parts):
                for move_uci in parts[moves_idx:]:
                    board.push(chess.Move.from_uci(move_uci))

        elif cmd == 'go':
            # parse 'depth' if provided
            if 'depth' in parts:
                idx = parts.index('depth')
                if idx + 1 < len(parts):
                    engine.depth = int(parts[idx + 1])
                    # update engine depth
            
            best_move = engine.get_best_move(board)
            print(f'bestmove {best_move}')

        elif cmd == 'd':
            print(board)

        elif cmd == 'quit': break

if __name__ == '__main__':
    strategy = 'constant'
    if len(sys.argv) > 1: strategy = sys.argv[1]

    try:
        # import 'evaluation.X'
        module_name = f'evaluation.{strategy}'
        module = importlib.import_module(module_name)
        
        # get the function
        function = module.evaluation_function
        
        # start UCI loop with this function
        uci_loop(function)
        
    except ImportError as e:
        print(f'info string (error): cannot load strategy \'{strategy}\' || {str(e).lower()}')