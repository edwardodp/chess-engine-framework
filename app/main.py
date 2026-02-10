import sys
import os
import glob
import ctypes
import importlib.util
import tkinter as tk
from tkinter import ttk
import numpy as np

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
COMPETITION_DEPTH = 4
LOADED_BOTS_CACHE = {}

# C callback type: int32_t (*)(const uint64_t*, const uint64_t*, uint32_t)
EVAL_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint32,
)

class CallbackWrapper:
    """Holds both the ctypes callback object (preventing GC) and its address."""
    def __init__(self, cb):
        self._cb = cb                                       # prevent garbage collection
        self.address = ctypes.cast(cb, ctypes.c_void_p).value

# -----------------------------------------------------------
# DUMMY EVAL
# -----------------------------------------------------------
def _dummy_eval(pieces_ptr, occupancy_ptr, side):
    return 0

dummy_wrapper = CallbackWrapper(EVAL_FUNC_TYPE(_dummy_eval))

# -----------------------------------------------------------
# BOT LOADING
# -----------------------------------------------------------
def load_bot_safely(bot_name, bot_path):
    print(f"Attempting to load: {bot_name}...")
    
    # Clean sys.modules to prevent pollution
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
                real_eval_func = eval_mod.evaluation_function
                
                def make_wrapper(fn):
                    """Create a ctypes callback that reliably closes over fn."""
                    def wrapper(pieces_ptr, occupancy_ptr, side):
                        # Convert raw C pointers to numpy arrays, cast to int64
                        # to match what Numba @njit bot functions expect
                        pieces = np.ctypeslib.as_array(pieces_ptr, shape=(12,)).astype(np.int64)
                        occupancy = np.ctypeslib.as_array(occupancy_ptr, shape=(3,)).astype(np.int64)
                        return int(fn(pieces, occupancy, np.int32(side)))
                    return CallbackWrapper(EVAL_FUNC_TYPE(wrapper))
                
                cb = make_wrapper(real_eval_func)
                LOADED_BOTS_CACHE[bot_path] = cb
                print(f"  -> Success. Address: {hex(cb.address)}")
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
BTN_BG     = "#89b4fa"
BTN_FG     = "#1e1e2e"
BTN_HOVER  = "#74c7ec"

CHESS_PIECES = "♔ ♕ ♖ ♗ ♘ ♙"

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

        w, h = 520, 560
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

        # --- Start Button ---
        self.btn = tk.Button(root, text="▶  START MATCH", font=("Helvetica", 13, "bold"),
                             bg=BTN_BG, fg=BTN_FG, activebackground=BTN_HOVER,
                             activeforeground=BTN_FG, relief="flat", cursor="hand2",
                             padx=20, pady=10, command=self.launch)
        self.btn.pack(pady=(24, 0))
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=BTN_HOVER))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=BTN_BG))

        # --- Status Bar ---
        self.status_var = tk.StringVar(value=f"{len(self.bots_dict)} bot(s) loaded  ·  Ready")
        self.status = tk.Label(root, textvariable=self.status_var, font=("Helvetica", 10),
                               fg=FG_DIM, bg=BG_DARK)
        self.status.pack(side=tk.BOTTOM, pady=12)

    def set_status(self, text, colour=FG_DIM):
        self.status_var.set(text)
        self.status.config(fg=colour)

    def launch(self):
        w_name = self.white_var.get()
        b_name = self.black_var.get()

        if w_name == "Human" and b_name == "Human":
            self.set_status("✗  Human vs Human not supported", FG_RED)
            return

        # 1. SELECT FUNCTIONS
        w_func = self.bots_dict.get(w_name) if w_name != "Human" else dummy_wrapper
        b_func = self.bots_dict.get(b_name) if b_name != "Human" else dummy_wrapper

        # 2. PRINT DEBUG INFO BEFORE SENDING
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
            # Always re-show the launcher when the C++ window closes
            self.root.deiconify()
            self.set_status(f"{len(self.bots_dict)} bot(s) loaded  ·  Ready")

    def run_cpp_engine(self, white_cb, black_cb, mode):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        bindings_dir = os.path.join(curr_dir, "..", "bindings")
        
        lib_name = "libChessLib.dylib" # Simplify for Mac, adjust if needed
        lib_path = os.path.abspath(os.path.join(bindings_dir, lib_name))
        
        chess_lib = ctypes.CDLL(lib_path)
        chess_lib.startEngine.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        
        fen_str = self.fen_var.get()
        fen_bytes = fen_str.encode('utf-8')
        
        # 3. CALL C++
        print("[PYTHON DEBUG] Calling C++ startEngine...")
        chess_lib.startEngine(white_cb.address, black_cb.address, COMPETITION_DEPTH, mode, fen_bytes)

if __name__ == "__main__":
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
