import sys
import os
import glob
import ctypes
import importlib.util
import tkinter as tk
from tkinter import ttk
import numpy as np
import multiprocessing
import time

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
COMPETITION_DEPTH = 5
MAX_MOVES_PER_GAME = 200
LOADED_BOTS_CACHE = {}

# 4 balanced openings — replace with Stockfish-verified FENs later
TOURNAMENT_OPENINGS = [
    ("Starting Position", "startpos"),
    ("Italian Game",      "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Queen's Gambit",    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
    ("Sicilian Defence",  "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
]

# ctypes fallback type (only used if @cfunc fails)
EVAL_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint32,
)

class CallbackWrapper:
    """Holds a native callback and its address. Works with both @cfunc and ctypes."""
    def __init__(self, cb):
        self._cb = cb  # prevent garbage collection
        if hasattr(cb, 'address'):
            self.address = cb.address
        else:
            self.address = ctypes.cast(cb, ctypes.c_void_p).value

# -----------------------------------------------------------
# DUMMY EVAL (native via @cfunc)
# -----------------------------------------------------------
from numba import cfunc, types, carray

_c_sig = types.int32(types.CPointer(types.uint64), types.CPointer(types.uint64), types.uint32)

@cfunc(_c_sig)
def _dummy_cfunc(pieces_ptr, occupancy_ptr, side):
    return 0

dummy_wrapper = CallbackWrapper(_dummy_cfunc)

# -----------------------------------------------------------
# @cfunc wrapper source — exec'd inside each bot's module
# namespace so evaluation_function is a module-level name,
# NOT a closure capture. This compiles to a direct native call.
# -----------------------------------------------------------
_CFUNC_WRAPPER_SOURCE = """
from numba import cfunc, types, carray
import numpy as np

_c_sig = types.int32(types.CPointer(types.uint64), types.CPointer(types.uint64), types.uint32)

@cfunc(_c_sig)
def _native_eval_wrapper(pieces_ptr, occupancy_ptr, side):
    pieces = carray(pieces_ptr, (12,), dtype=np.uint64).astype(np.int64)
    occupancy = carray(occupancy_ptr, (3,), dtype=np.uint64).astype(np.int64)
    return evaluation_function(pieces, occupancy, side)
"""

# -----------------------------------------------------------
# BOT LOADING
# -----------------------------------------------------------
def load_bot_safely(bot_name, bot_path):
    print(f"Attempting to load: {bot_name}...")

    common_names = ["board_tools", "evaluation", "search", "move_gen", "uci"]
    for name in common_names:
        if name in sys.modules: del sys.modules[name]

    original_sys_path = sys.path[:]
    sys.path.insert(0, bot_path)
    local_modules = {}

    try:
        py_files = glob.glob(os.path.join(bot_path, "*.py"))
        for filepath in py_files:
            filename = os.path.basename(filepath)
            module_short_name = filename.replace(".py", "")
            unique_name = f"{bot_name}_{module_short_name}"

            spec = importlib.util.spec_from_file_location(unique_name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[unique_name] = module
                try:
                    spec.loader.exec_module(module)
                    local_modules[module_short_name] = module
                except Exception as e:
                    print(f"  -> Warning: Failed to load {filename}: {e}")

        for name, mod in local_modules.items(): sys.modules[name] = mod

        if "evaluation" in local_modules:
            eval_mod = local_modules["evaluation"]
            if hasattr(eval_mod, "evaluation_function"):

                # --- FAST PATH: compile @cfunc in the bot's own namespace ---
                try:
                    exec(_CFUNC_WRAPPER_SOURCE, eval_mod.__dict__)
                    native_cb = eval_mod.__dict__['_native_eval_wrapper']
                    cb = CallbackWrapper(native_cb)
                    LOADED_BOTS_CACHE[bot_path] = cb
                    print(f"  -> Success (native @cfunc). Address: {hex(cb.address)}")
                    return cb
                except Exception as e:
                    print(f"  -> @cfunc failed ({e}), falling back to ctypes...")

                # --- SLOW FALLBACK: ctypes wrapper ---
                real_eval_func = eval_mod.evaluation_function
                def make_wrapper(fn):
                    def wrapper(pieces_ptr, occupancy_ptr, side):
                        pieces = np.ctypeslib.as_array(pieces_ptr, shape=(12,)).astype(np.int64)
                        occupancy = np.ctypeslib.as_array(occupancy_ptr, shape=(3,)).astype(np.int64)
                        return int(fn(pieces, occupancy, np.int32(side)))
                    return CallbackWrapper(EVAL_FUNC_TYPE(wrapper))

                cb = make_wrapper(real_eval_func)
                LOADED_BOTS_CACHE[bot_path] = cb
                print(f"  -> Success (ctypes fallback). Address: {hex(cb.address)}")
                return cb
            else:
                print(f"  -> Error: missing 'evaluation_function'")
        else:
            print(f"  -> Error: No evaluation.py found")

    except Exception as e:
        print(f"  -> Critical Error loading {bot_name}: {e}")

    finally:
        sys.path = original_sys_path
        for name in local_modules.keys():
            if name in sys.modules: del sys.modules[name]

    return None


# -----------------------------------------------------------
# HEADLESS GAME WORKER (runs in child process)
# -----------------------------------------------------------
def _run_game_worker(result_queue, game_id, white_name, white_path,
                     black_name, black_path, fen, opening_name, depth, max_moves):
    """
    Each child process independently loads both bots and the C++ library,
    runs a single headless game, and sends the result back via the queue.
    """
    try:
        # Load bots fresh in this process
        white_cb = load_bot_safely(f"{white_name}_w", white_path)
        black_cb = load_bot_safely(f"{black_name}_b", black_path)

        if white_cb is None or black_cb is None:
            result_queue.put((game_id, opening_name, white_name, black_name,
                              "ERROR", "Bot failed to load"))
            return

        # Load C++ library
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bindings_dir = os.path.join(script_dir, "..", "bindings")
        lib_name = "libChessLib.dylib"
        lib_path = os.path.abspath(os.path.join(bindings_dir, lib_name))

        chess_lib = ctypes.CDLL(lib_path)
        chess_lib.runHeadlessGame.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int
        ]
        chess_lib.runHeadlessGame.restype = ctypes.c_int

        fen_bytes = fen.encode('utf-8')
        t0 = time.time()
        result_code = chess_lib.runHeadlessGame(
            white_cb.address, black_cb.address,
            depth, fen_bytes, max_moves
        )
        elapsed = time.time() - t0

        result_map = {
            0:  "Draw",
            1:  f"{white_name} wins",
            2:  f"{black_name} wins",
            -1: "Draw (max moves)",
        }
        result_str = result_map.get(result_code, f"Unknown ({result_code})")

        # Determine points: winner gets 1, draw gives 0.5 each
        w_pts = 0.5 if result_code in (0, -1) else (1.0 if result_code == 1 else 0.0)
        b_pts = 0.5 if result_code in (0, -1) else (1.0 if result_code == 2 else 0.0)

        result_queue.put((game_id, opening_name, white_name, black_name,
                          result_str, f"{elapsed:.1f}s", w_pts, b_pts))

    except Exception as e:
        result_queue.put((game_id, opening_name, white_name, black_name,
                          "CRASH", str(e), 0.0, 0.0))


# -----------------------------------------------------------
# COLOURS & THEME
# -----------------------------------------------------------
BG_DARK    = "#1e1e2e"
BG_CARD    = "#2a2a3d"
BG_INPUT   = "#363650"
FG_PRIMARY = "#cdd6f4"
FG_DIM     = "#6c7086"
FG_ACCENT  = "#89b4fa"
FG_GREEN   = "#a6e3a1"
FG_RED     = "#f38ba8"
FG_GOLD    = "#f9e2af"
FG_YELLOW  = "#f9e2af"
BTN_BG     = "#89b4fa"
BTN_FG     = "#1e1e2e"
BTN_HOVER  = "#74c7ec"
BTN_GREEN  = "#a6e3a1"
BTN_GREEN_HOVER = "#94e2d5"

CHESS_PIECES = "♔ ♕ ♖ ♗ ♘ ♙"


# -----------------------------------------------------------
# RESULTS WINDOW
# -----------------------------------------------------------
class ResultsWindow:
    """Popup window that displays tournament results as they come in."""

    def __init__(self, parent, bot_a, bot_b, num_games):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Headless Match: {bot_a} vs {bot_b}")
        self.top.configure(bg=BG_DARK)
        self.top.resizable(False, False)

        w, h = 700, 520
        ws = self.top.winfo_screenwidth()
        hs = self.top.winfo_screenheight()
        self.top.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")

        self.bot_a = bot_a
        self.bot_b = bot_b

        # Header
        tk.Label(self.top, text="⚔  Headless Match-Up", font=("Helvetica", 16, "bold"),
                 fg=FG_GOLD, bg=BG_DARK).pack(pady=(16, 2))
        tk.Label(self.top, text=f"{bot_a}  vs  {bot_b}",
                 font=("Helvetica", 12), fg=FG_PRIMARY, bg=BG_DARK).pack(pady=(0, 12))

        # Table frame
        table_frame = tk.Frame(self.top, bg=BG_CARD, padx=2, pady=2)
        table_frame.pack(padx=20, fill="x")

        # Column headers
        headers = ["#", "Opening", "White", "Black", "Result", "Time"]
        col_widths = [3, 18, 14, 14, 16, 6]
        header_row = tk.Frame(table_frame, bg=BG_INPUT)
        header_row.pack(fill="x")
        for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
            tk.Label(header_row, text=hdr, font=("Helvetica", 10, "bold"),
                     fg=FG_ACCENT, bg=BG_INPUT, width=cw, anchor="w",
                     padx=6, pady=4).pack(side="left")

        # Row labels (filled in as results arrive)
        self.row_frames = []
        self.row_labels = []
        for g in range(num_games):
            bg = BG_CARD if g % 2 == 0 else BG_INPUT
            row = tk.Frame(table_frame, bg=bg)
            row.pack(fill="x")
            labels = []
            for cw in col_widths:
                lbl = tk.Label(row, text="…", font=("Courier", 10),
                               fg=FG_DIM, bg=bg, width=cw, anchor="w", padx=6, pady=3)
                lbl.pack(side="left")
            labels = list(row.winfo_children())
            self.row_frames.append(row)
            self.row_labels.append(labels)
            # Pre-fill game number
            labels[0].config(text=str(g + 1))

        # Score summary
        self.summary_var = tk.StringVar(value="Running games…")
        self.summary = tk.Label(self.top, textvariable=self.summary_var,
                                font=("Helvetica", 14, "bold"),
                                fg=FG_PRIMARY, bg=BG_DARK)
        self.summary.pack(pady=(16, 4))

        self.detail_var = tk.StringVar(value="")
        tk.Label(self.top, textvariable=self.detail_var,
                 font=("Helvetica", 11), fg=FG_DIM, bg=BG_DARK).pack(pady=(0, 12))

    def update_row(self, game_id, opening, white, black, result, elapsed):
        if game_id < len(self.row_labels):
            labels = self.row_labels[game_id]
            values = [str(game_id + 1), opening, white, black, result, elapsed]
            # Pick colour based on result
            for lbl, val in zip(labels, values):
                lbl.config(text=val, fg=FG_PRIMARY)

            # Highlight the result column
            result_lbl = labels[4]
            if "wins" in result:
                winner = result.replace(" wins", "")
                if winner == self.bot_a:
                    result_lbl.config(fg=FG_GREEN)
                elif winner == self.bot_b:
                    result_lbl.config(fg=FG_RED)
            elif "Draw" in result:
                result_lbl.config(fg=FG_YELLOW)
            elif result in ("ERROR", "CRASH"):
                result_lbl.config(fg=FG_RED)

    def set_summary(self, text, detail=""):
        self.summary_var.set(text)
        self.detail_var.set(detail)


# -----------------------------------------------------------
# GUI
# -----------------------------------------------------------
class LauncherApp:
    def __init__(self, root, bots_dict):
        self.root = root
        self.bots_dict = bots_dict
        self.root.title("Chess Engine")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        w, h = 520, 620
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = int((ws / 2) - (w / 2))
        y = int((hs / 2) - (h / 2))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.options = ["Human"] + list(self.bots_dict.keys())

        # --- Header ---
        header = tk.Frame(root, bg=BG_DARK)
        header.pack(fill="x", pady=(28, 0))

        tk.Label(header, text="♚", font=("Arial", 36), fg=FG_GOLD, bg=BG_DARK).pack()
        tk.Label(header, text="Chess Engine", font=("Helvetica", 20, "bold"),
                 fg=FG_PRIMARY, bg=BG_DARK).pack(pady=(2, 0))
        tk.Label(header, text=CHESS_PIECES, font=("Arial", 12),
                 fg=FG_DIM, bg=BG_DARK).pack(pady=(2, 0))

        # --- Card ---
        card = tk.Frame(root, bg=BG_CARD, highlightbackground=BG_INPUT,
                        highlightthickness=1, padx=30, pady=24)
        card.pack(padx=40, pady=(20, 0), fill="x")

        # FEN
        tk.Label(card, text="START POSITION", font=("Helvetica", 9, "bold"),
                 fg=FG_DIM, bg=BG_CARD, anchor="w").pack(fill="x", pady=(0, 4))
        self.fen_var = tk.StringVar(value="startpos")
        fen_entry = tk.Entry(card, textvariable=self.fen_var, font=("Courier", 12),
                             bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                             relief="flat", highlightthickness=0)
        fen_entry.pack(fill="x", ipady=6)

        # Separator
        tk.Frame(card, bg=BG_INPUT, height=1).pack(fill="x", pady=(18, 14))

        # Players row
        players_frame = tk.Frame(card, bg=BG_CARD)
        players_frame.pack(fill="x")
        players_frame.columnconfigure(0, weight=1, uniform="col")
        players_frame.columnconfigure(1, weight=0)
        players_frame.columnconfigure(2, weight=1, uniform="col")

        # White
        w_frame = tk.Frame(players_frame, bg=BG_CARD)
        w_frame.grid(row=0, column=0, sticky="nsew")
        tk.Label(w_frame, text="♔", font=("Arial", 22), fg="#ffffff", bg=BG_CARD).pack()
        tk.Label(w_frame, text="WHITE", font=("Helvetica", 9, "bold"),
                 fg=FG_DIM, bg=BG_CARD).pack(pady=(0, 4))
        self.white_var = tk.StringVar(value="Human")
        w_cb = ttk.Combobox(w_frame, textvariable=self.white_var,
                            values=self.options, state="readonly", width=16,
                            font=("Helvetica", 11))
        w_cb.pack()

        # VS
        tk.Label(players_frame, text="vs", font=("Helvetica", 14, "bold"),
                 fg=FG_DIM, bg=BG_CARD).grid(row=0, column=1, padx=14)

        # Black
        b_frame = tk.Frame(players_frame, bg=BG_CARD)
        b_frame.grid(row=0, column=2, sticky="nsew")
        tk.Label(b_frame, text="♚", font=("Arial", 22), fg=FG_DIM, bg=BG_CARD).pack()
        tk.Label(b_frame, text="BLACK", font=("Helvetica", 9, "bold"),
                 fg=FG_DIM, bg=BG_CARD).pack(pady=(0, 4))
        self.black_var = tk.StringVar(value="Human")
        b_cb = ttk.Combobox(b_frame, textvariable=self.black_var,
                            values=self.options, state="readonly", width=16,
                            font=("Helvetica", 11))
        b_cb.pack()

        # --- Buttons ---
        btn_frame = tk.Frame(root, bg=BG_DARK)
        btn_frame.pack(pady=(20, 0))

        self.btn = tk.Button(btn_frame, text="▶  START MATCH", font=("Helvetica", 13, "bold"),
                             bg=BTN_BG, fg=BTN_FG, activebackground=BTN_HOVER,
                             activeforeground=BTN_FG, relief="flat", cursor="hand2",
                             padx=20, pady=10, command=self.launch)
        self.btn.pack(side="left", padx=(0, 8))
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=BTN_HOVER))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=BTN_BG))

        self.headless_btn = tk.Button(btn_frame, text="⚔  HEADLESS MATCH",
                                      font=("Helvetica", 13, "bold"),
                                      bg=BTN_GREEN, fg=BTN_FG,
                                      activebackground=BTN_GREEN_HOVER,
                                      activeforeground=BTN_FG, relief="flat",
                                      cursor="hand2", padx=20, pady=10,
                                      command=self.launch_headless)
        self.headless_btn.pack(side="left")
        self.headless_btn.bind("<Enter>", lambda e: self.headless_btn.config(bg=BTN_GREEN_HOVER))
        self.headless_btn.bind("<Leave>", lambda e: self.headless_btn.config(bg=BTN_GREEN))

        # --- Status Bar ---
        self.status_var = tk.StringVar(value=f"{len(self.bots_dict)} bot(s) loaded  ·  Ready")
        self.status = tk.Label(root, textvariable=self.status_var, font=("Helvetica", 10),
                               fg=FG_DIM, bg=BG_DARK)
        self.status.pack(side=tk.BOTTOM, pady=12)

    def set_status(self, text, colour=FG_DIM):
        self.status_var.set(text)
        self.status.config(fg=colour)

    # -----------------------------------------------------------
    # REGULAR GUI MATCH
    # -----------------------------------------------------------
    def launch(self):
        w_name = self.white_var.get()
        b_name = self.black_var.get()

        if w_name == "Human" and b_name == "Human":
            self.set_status("✗  Human vs Human not supported", FG_RED)
            return

        w_func = self.bots_dict.get(w_name) if w_name != "Human" else dummy_wrapper
        b_func = self.bots_dict.get(b_name) if b_name != "Human" else dummy_wrapper

        print(f"\n[PYTHON DEBUG] Preparing to launch...")
        print(f"[PYTHON DEBUG] White ({w_name}): {hex(w_func.address)}")
        print(f"[PYTHON DEBUG] Black ({b_name}): {hex(b_func.address)}")

        mode = 2
        if w_name == "Human": mode = 0
        elif b_name == "Human": mode = 1

        self.set_status(f"Running: {w_name} vs {b_name}…", FG_ACCENT)
        self.root.withdraw()
        try:
            self.run_cpp_engine(w_func, b_func, mode)
        except Exception as e:
            print(f"[ERROR] Engine Crash: {e}")
        finally:
            self.root.deiconify()
            self.set_status(f"{len(self.bots_dict)} bot(s) loaded  ·  Ready")

    def run_cpp_engine(self, white_cb, black_cb, mode):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        bindings_dir = os.path.join(curr_dir, "..", "bindings")
        lib_name = "libChessLib.dylib"
        lib_path = os.path.abspath(os.path.join(bindings_dir, lib_name))

        chess_lib = ctypes.CDLL(lib_path)
        chess_lib.startEngine.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p
        ]

        fen_bytes = self.fen_var.get().encode('utf-8')
        print("[PYTHON DEBUG] Calling C++ startEngine...")
        chess_lib.startEngine(white_cb.address, black_cb.address,
                              COMPETITION_DEPTH, mode, fen_bytes)

    # -----------------------------------------------------------
    # HEADLESS TOURNAMENT
    # -----------------------------------------------------------
    def launch_headless(self):
        w_name = self.white_var.get()
        b_name = self.black_var.get()

        if w_name == "Human" or b_name == "Human":
            self.set_status("✗  Headless requires Bot vs Bot", FG_RED)
            return

        if w_name == b_name:
            self.set_status("✗  Select two different bots", FG_RED)
            return

        # Resolve bot paths
        bot_a_name = w_name
        bot_b_name = b_name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        bots_dir = os.path.join(base_dir, "bots")
        bot_a_path = os.path.join(bots_dir, bot_a_name)
        bot_b_path = os.path.join(bots_dir, bot_b_name)

        if not os.path.isdir(bot_a_path) or not os.path.isdir(bot_b_path):
            self.set_status("✗  Bot directory not found", FG_RED)
            return

        # Build game list: 4 openings × 2 sides = 8 games
        games = []
        game_id = 0
        for opening_name, fen in TOURNAMENT_OPENINGS:
            games.append((game_id, bot_a_name, bot_a_path,
                          bot_b_name, bot_b_path, fen, opening_name))
            game_id += 1
            games.append((game_id, bot_b_name, bot_b_path,
                          bot_a_name, bot_a_path, fen, opening_name))
            game_id += 1

        num_games = len(games)
        self.set_status(f"Running {num_games} headless games…", FG_ACCENT)

        # Open results window
        results_win = ResultsWindow(self.root, bot_a_name, bot_b_name, num_games)

        # Launch all games as separate processes
        result_queue = multiprocessing.Queue()
        processes = []
        for (gid, w_n, w_p, b_n, b_p, fen, op_name) in games:
            p = multiprocessing.Process(
                target=_run_game_worker,
                args=(result_queue, gid, w_n, w_p, b_n, b_p, fen, op_name,
                      COMPETITION_DEPTH, MAX_MOVES_PER_GAME)
            )
            processes.append(p)
            p.start()

        # Poll for results without blocking the GUI
        self._poll_results(results_win, result_queue, processes,
                           num_games, bot_a_name, bot_b_name)

    def _poll_results(self, results_win, result_queue, processes,
                      total, bot_a, bot_b):
        """Non-blocking poll: checks the queue, updates the GUI, reschedules itself."""
        finished = 0
        scores = {bot_a: 0.0, bot_b: 0.0}

        def poll():
            nonlocal finished
            while not result_queue.empty():
                msg = result_queue.get_nowait()
                # msg: (game_id, opening, white, black, result_str, elapsed, w_pts, b_pts)
                if len(msg) == 8:
                    gid, opening, white, black, result_str, elapsed, w_pts, b_pts = msg
                    results_win.update_row(gid, opening, white, black, result_str, elapsed)
                    scores[white] = scores.get(white, 0) + w_pts
                    scores[black] = scores.get(black, 0) + b_pts
                elif len(msg) == 6:
                    gid, opening, white, black, result_str, detail = msg
                    results_win.update_row(gid, opening, white, black, result_str, detail)

                finished += 1
                results_win.set_summary(
                    f"{bot_a} {scores.get(bot_a, 0):.1f}  —  {scores.get(bot_b, 0):.1f} {bot_b}",
                    f"{finished}/{total} games complete"
                )

            if finished < total:
                self.root.after(200, poll)
            else:
                # All done — clean up processes
                for p in processes:
                    p.join(timeout=2)

                a_score = scores.get(bot_a, 0)
                b_score = scores.get(bot_b, 0)
                if a_score > b_score:
                    verdict = f"🏆 {bot_a} wins the match!"
                elif b_score > a_score:
                    verdict = f"🏆 {bot_b} wins the match!"
                else:
                    verdict = "Match drawn!"

                results_win.set_summary(
                    f"{bot_a} {a_score:.1f}  —  {b_score:.1f} {bot_b}",
                    verdict
                )
                self.set_status(f"{len(self.bots_dict)} bot(s) loaded  ·  Ready")

        poll()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bots_dir = os.path.join(base_dir, "bots")
    bots = {}
    if os.path.exists(bots_dir):
        for bot_name in os.listdir(bots_dir):
            bot_path = os.path.join(bots_dir, bot_name)
            if os.path.isdir(bot_path):
                if os.path.exists(os.path.join(bot_path, "evaluation.py")):
                    func = load_bot_safely(bot_name, bot_path)
                    if func: bots[bot_name] = func

    root = tk.Tk()
    app = LauncherApp(root, bots)
    root.mainloop()
