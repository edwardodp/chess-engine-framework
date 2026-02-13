# Chess Engine Framework

A chess engine platform built for the University of Warwick Computing Society, where students write evaluation functions in Python that plug into a shared C++ search backend. The framework was used to run the Chess Engine Showdown -- a live event sponsored by Optiver in which student-built engines played against rated human players from the Warwick Chess Society.

## Overview

The framework separates the chess engine into two layers:

- **Search backend (C++):** Handles move generation, negamax search with principal variation search (PVS), quiescence search, killer move heuristic, history heuristic, and board state management. Compiled into a shared library that is loaded at runtime.
- **Evaluation frontend (Python):** Students write a single `evaluation_function` in Python that scores a chess position given the bitboard representation. These functions are compiled with Numba and passed as native callbacks into the C++ backend, avoiding the overhead of crossing the Python/C++ boundary on every node evaluation.

This architecture means all competitors benefit from the same search optimisations, and the competition is purely about who can write the best position evaluation.

## Features

- **Human vs Bot:** Play against any loaded bot through an SFML-based graphical interface with piece rendering, move highlighting, and promotion UI.
- **Bot vs Bot (Headed):** Watch two bots play each other in the GUI with full visualisation.
- **Headless Tournaments:** Run batch matches between bots in parallel using multiprocessing, with results displayed in a live-updating scoreboard. Useful for ranking bots before an event.
- **PGN Export:** Games played in the GUI are recorded and can be exported as PGN files after the game ends, including support for games starting from custom FEN positions.
- **Undo Move:** In Human vs Bot mode, an undo button allows taking back moves (undoes both the human move and the bot's response).
- **Cross-Platform:** Shared libraries are built for Windows (DLL), Linux (SO), macOS ARM (dylib), and macOS Intel (dylib) via GitHub Actions. The Python launcher auto-detects the correct binary for the current platform.
- **Custom Openings:** Games can be started from any FEN position, allowing tournaments to use specific openings for fairer matchups.

## Getting Started

### Installation

       git clone <repository-url>
       cd chess-engine-framework

       python -m venv .venv
       source .venv/bin/activate # not windows...
       .venv\Scripts\activate # windows

       pip install -r requirements.txt

Download the shared library for your platform from the GitHub Actions build artifacts and place it in a `bindings/` directory at the project root:

       bindings/
           ChessLib.dll                # Windows
           libChessLib.so              # Linux
           libChessLib_arm64.dylib     # macOS Silicon
           libChessLib_intel.dylib     # macOS Intel

   You only need the one that matches your platform.

5. Run the launcher:

       python app/main.py

### Adding a Bot

To add a new bot, create a directory under `app/bots/` containing an `evaluation.py` file:

    app/bots/
        YourBotName/
            board_tools.py      # Optional: shared utilities
            evaluation.py       # Required: must define evaluation_function

The evaluation function must have the following signature:

```python
from numba import njit, int32, int64, uint32
import numpy as np

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    # board_pieces: int64 array of length 12
    #   [0] = white pawns,   [1] = white knights, [2] = white bishops,
    #   [3] = white rooks,   [4] = white queens,  [5] = white king,
    #   [6] = black pawns,   [7] = black knights,  [8] = black bishops,
    #   [9] = black rooks,   [10] = black queens,  [11] = black king
    #
    # board_occupancy: int64 array of length 3
    #   [0] = all white pieces, [1] = all black pieces, [2] = all pieces
    #
    # side_to_move: 0 = white, 1 = black
    #
    # Returns: integer score from the perspective of side_to_move
    #          (positive = good for side to move)

    score = 0
    # ... your evaluation logic here ...
    return np.int32(score)
```

Each bitboard is a 64-bit integer where bit N corresponds to square N (A1=0, B1=1, ..., H8=63). The function must be decorated with `@njit` and an explicit type signature so that Numba compiles it ahead of time into native code. This is what allows a Python evaluation function to be called millions of times per second from the C++ search without significant overhead.

The launcher will automatically discover and load any bot directory that contains an `evaluation.py` file when the program starts.

## How It Works

### Numba Compilation

When the launcher loads a bot, it compiles the evaluation function using Numba's `@cfunc` mechanism to produce a native function pointer. This pointer is passed directly to the C++ shared library via ctypes, meaning the C++ search calls the Python evaluation at native speed with no interpreter overhead. This was critical for making the framework viable -- without Numba compilation, the Python/C++ boundary crossing would have made the engine orders of magnitude slower.

### Search Architecture

The C++ search implements:

- **Iterative deepening** with configurable depth
- **Principal Variation Search (PVS)** for more efficient alpha-beta pruning
- **Quiescence search** with delta pruning and an 8-ply depth cap for balanced speed
- **Move ordering:** MVV-LVA for captures, killer moves and history heuristic for quiet moves
- **Previous-best-move ordering** at the root (from iterative deepening)
- **Draw detection:** Threefold repetition and fifty-move rule

### Dispatcher

Since the C++ search only accepts a single evaluation callback, a dispatcher function routes calls to the correct bot's evaluation based on which side is currently being searched. This allows two different student evaluations to play against each other through the same search function.

## Project Structure

    chess-engine-framework/
        app/
            main.py                 # Python launcher, GUI, tournament system
            bots/
                TemplateBot/        # Example bot
                    board_tools.py
                    evaluation.py
                ...
        src/
            Main.cpp                # C++ entry points (startEngine, runHeadlessGame)
            Interface.cpp           # SFML GUI, game loop, move history, undo
            Search.cpp              # iterative deepening, quiscence
            MoveGen.cpp             # Legal move generation
            Attacks.cpp             # Attack detection
            Zobrist.cpp             # Position hashing
        include/
            BoardState.hpp          # Board representation, make/undo move, draw detection
            Types.hpp               # Move encoding, piece types, squares
            ...
        bindings/                   # Shared libraries (needs to be added)
        assets/                     # Piece images, font
        requirements.txt

## The Chess Engine Showdown

The framework was built for a live event where three finalist bots (selected through a headless round-robin tournament) played against rated human players from the Warwick Chess Society. The human players were rated approximately 2200, 1600, and 1000 ECF. Each finalist played one game against each human opponent.

At search depth 7 with the PVS and move ordering optimisations, the top-seeded student bot defeated the 2200-rated player. The results across all matchups were competitive, with each rating bracket producing decisive and drawn results -- exactly the kind of balance that made for engaging spectating.

The event was funded by a £500 sponsorship from Optiver. Thanks Optiver :)
