# Chess Engine Competition
Welcome to the UWCS x Chess Soc x Optiver Chess Engine Competition!

This project will allow you to make your own chess engine by implementing your the evaluation function (the rest is already done for you!). All participants are required to use the base chess engine implementation found in this repo.

---
## Competition details
The competition will run in 2 parts. The first part is development. This is when you will be implementing your evaluation functions and will run from W1 Friday - W4 Sunday. Submissions will be open during the last week of development. These submissions will be judged based on technical prowess and how creative the solution is. 

From these submissions, a handful will be selected to enter the next part of the competition where the chess engines will be pitted against chess players from Chess Soc on W5 Thursday! The chess engines will then by judged by Chess Soc based on playing style and general performance against real opponents. 

There are £400 in prizes to be won so get coding!

---

## File Structure
```
app/
│   main.py             (DO NOT EDIT)
│   evaluation.py       (your code goes here)
│   board_tools.py      (helper functions)
bindings/               (make this directory)
│   (.dll, .so, or .dylib file from releases goes here)
```
---
## How to run
Firstly you will need to setup your python virtual environment. Make sure you are always working in your python virtual environment. Make sure you are using Python 3.12.4 or some newer stable release.

### Linux/macOS
1. Create the virtual environment in the root of this project `python -m venv .venv`
2. Activate the environment `source .venv/bin/activate`
3. Install the necessary packages `pip install -r requirements.txt`

### Windows Powershell
1. Create the virtual environment in the root of this project `py -m venv .venv`
2. Activate the environment `.venv\Scripts\Activate.ps1`
3. Install the necessary packages `pip install -r requirements.txt`

### Downloading shared library
Under the **Releases** section of the GitHub repository, you'll find the library needed to run the project. Place those files inside your `bindings/` directory. For Windows you'll only need the `.dll`, for Linux the `.so`, for Mac the `.dylib` corresponding to your device's architecture.

Once set up, simply run `python app/main.py` in your project root.

For Mac users: you will get a pop up telling you the library cannot be verified. Go to System Settings > Privacy & Security > Scroll down to Security > Always Allow (after attempting to run once).

---
## Restrictions
To make your python code fast enough to run inside the chess engine, we are using a compiler called Numba. Your function has the `@njit` decorator which enforces "No Python" mode.
This is so that your code can be compiled to machine code when it first runs the function. Because of this, **you cannot use standard python objects**.

### Cannot do/use
- Python objects `dict`, `list`, `set`, or any classes
- Dynamic typing (e.g. `x = 1` then later `x = "hello"`)
- No external libraries like `math`, `random`, or `pandas` (you can use `numpy` selectively, however)
- No mutable global variables (you must simply treat any globals as constants)

### Can and should do/use
- Use NumPy arrays (instead of lists), two of which are passed into the evaluation function you create
- Use primitives like `int`, `float`, `bool`. Python does treat everything as objects but Numba compiles these as primitives
- Built in operations like `+`, `-`, `*`, `&`, `|`, `^`

### Example of segfault (crash)
```py
def evaluation(bitboards, ...):
    magic_list = [1, 2, 3] # Cannot use lists becuase they're an object
    return sum(magic_list)
```

### Example of working code
```py
def evaluation(bitboards, ...):
    score = 0
    score += bitboards[0] * 10 # bitboards is a NumPy array so it can be accessed
    return score
```

---
## What is a chess engine
There are two main parts to a chess engine:
- Search (provided) which explore the tree of future moves. The shared library you place in `bindings/` is this search done in C++.
- Evaluation (which you write) which is used by the search when it reaches a board state it cannot search past (for performance or because the game is over). The evaluation tells the search which side is winning and by how much. These evaluations solely determine which move is picked.

Your evaluation function needs to return a single integer for any board state (where that board state exists in the search's game tree is not relevant):
- Score of zero means the position is equal
- Positive score means the player whose turn it is is winning
- Negative score means the player waiting for their turn is winning

*NOTE: This is different to mini-max search where positive means white is winning and vice versa, because this is a negamax search*

---
## What is an evaluation function
The evaluation function needs to translate a board state to a score representative of who's winning.

In `evaluation.py` you are given `board_pieces`, `board_occupancy`, and `side_to_move`. The first two arguments are NumPy arrays of bitboards which collectively represent the board state. The third parameter tells you whose turn it is (0 = White, 1 = Black).

### What is a bitboard?
A bitboard is simply a 64-bit integer where each bit represents a square on the chess board (0 to 63). If a bit is set to `1`, it means "something" is there. If it is `0`, the square is empty.

In this engine, the board state is split into two arrays:

**`board_pieces` (Where specific pieces are)**
This is an array of 12 integers. Each integer is a map for one specific piece type.
- Indices 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Indices 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)

**`board_occupancy` (Where *any* piece is)**
This is an array of 3 integers that combines the above data for quick lookups. You will use these often to check if a square is blocked.
- Index 0: **White Occupancy** (All white pieces combined)
- Index 1: **Black Occupancy** (All black pieces combined)
- Index 2: **All Occupancy** (Both White and Black combined)

#### Visual Example
Imagine the starting position. The **White Rooks** bitboard (Index 3 in `board_pieces`) would look like this in binary (formatted as a board):
```text
1 0 0 0 0 0 0 1  <- A1 and H1 are set (Rooks)
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
```
Indices for `board_pieces` are as follows:
- 0 - White pawns
- 1 - White knights
- 2 - White bishops
- 3 - White rooks
- 4 - White queens
- 5 - White king
- 6 - Black pawns
- 7 - Black knights
- 8 - Black bishops
- 9 - Black rooks
- 10 - Black queens
- 11 - Black king

Your function must calculate a score using these inputs. Basic strategies often count material, while advanced ones consider position (e.g., center control, pawn structure).

---
## Example evaluation functions

### The "Random" mover
*Terrible, but valid code. Since the score is always the same, the search engine sees every move as equal and will pick randomly (semantically speaking).*
```py
@njit
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    return 42
```

### Material value counter
This is what's provided in the repository as an example. It's a good starting point but severely lacks nuance
