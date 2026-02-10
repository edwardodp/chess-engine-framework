import math
import os
import numpy as np
from numba import njit, int64, int32, uint32, uint64

#Sparse attention transformer evaluation function for chess positions, with a simple material + positional baseline.
#Temporal Difference took too long, trained on stockfish data. 

# NOTE (Fixed by marker): np.load("model_numba.npz") used a relative path which
# fails when the working directory isn't the bot's folder. Fixed to use absolute
# path relative to this file's location.
MODEL_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_numba.npz")



# Parameters are both for Sparse Chess-Geometric Transformer but also for Least
# Square regression baseline with typical chess features (material, doubled pawns, isolated pawns, 
# bishop pair, passed pawns, piece-square tables).
_z = np.load(MODEL_NPZ, allow_pickle=False)

NUM_LAYERS = int(_z["num_layers"])
NUM_HEADS = int(_z["num_heads"])
EMBED_DIM = int(_z["embed_dim"])
MAX_TOKENS = int(_z["max_tokens"])

PIECE_EMBED = _z["piece_embed"].astype(np.float32)
COLOR_EMBED = _z["color_embed"].astype(np.float32)
SQUARE_EMBED = _z["square_embed"].astype(np.float32)
SCORE_HEAD = _z["score_head"].astype(np.float32)

WQ = _z["wq"].astype(np.float32)
WK = _z["wk"].astype(np.float32)
WV = _z["wv"].astype(np.float32)
WO = _z["wo"].astype(np.float32)
FF1 = _z["ff1"].astype(np.float32)
FF2 = _z["ff2"].astype(np.float32)

BASELINE_DOUBLED = np.float32(_z["baseline_doubled"])
BASELINE_ISOLATED = np.float32(_z["baseline_isolated"])
BASELINE_BISHOP = np.float32(_z["baseline_bishop_pair"])
BASELINE_PASSED = _z["baseline_passed"].astype(np.float32)
BASELINE_PSTS = _z["baseline_piece_psts"].astype(np.float32).reshape(-1)

@njit
def _count_bits_u64(x):
    bb = np.uint64(x)
    cnt = 0
    while bb != 0:
        bb = bb & (bb - np.uint64(1))
        cnt += 1
    return cnt


@njit
def _count_tokens_position(board_pieces):
    n = 0
    for i in range(12):
        n += _count_bits_u64(board_pieces[i])
    return n


@njit
def _popcount_u64(x):
    bb = np.uint64(x)
    cnt = 0
    while bb != 0:
        bb = bb & (bb - np.uint64(1))
        cnt += 1
    return cnt


@njit
def _file_of(sq):
    return sq % 8


@njit
def _rank_of(sq):
    return sq // 8


@njit
def _abs_delta(a, b):
    return abs(_file_of(a) - _file_of(b)), abs(_rank_of(a) - _rank_of(b))


@njit
def _attacks_xray(piece_type, color, from_sq, to_sq):
    if from_sq == to_sq:
        return False
    df, dr = _abs_delta(from_sq, to_sq)
    f_from = _file_of(from_sq)
    r_from = _rank_of(from_sq)
    f_to = _file_of(to_sq)
    r_to = _rank_of(to_sq)

    if piece_type == 0:
        if color == 0:
            return (r_to - r_from) == 1 and df == 1
        return (r_to - r_from) == -1 and df == 1
    if piece_type == 1:
        return (df == 1 and dr == 2) or (df == 2 and dr == 1)
    if piece_type == 2:
        return df == dr and df > 0
    if piece_type == 3:
        return (f_from == f_to) or (r_from == r_to)
    if piece_type == 4:
        return (df == dr and df > 0) or (f_from == f_to) or (r_from == r_to)
    return max(df, dr) == 1


@njit
def _encode_position_tokens(board_pieces, max_tokens):
    piece_ids = np.zeros((max_tokens,), dtype=np.int64)
    color_ids = np.zeros((max_tokens,), dtype=np.int64)
    square_ids = np.zeros((max_tokens,), dtype=np.int64)
    attn_allowed = np.zeros((max_tokens, max_tokens), dtype=np.bool_)
    n = 0

    for i in range(max_tokens):
        attn_allowed[i, i] = True

    for color in range(2):
        offset = 0 if color == 0 else 6
        for piece_type in range(6):
            bb = np.uint64(board_pieces[piece_type + offset])
            for sq in range(64):
                if ((bb >> np.uint64(sq)) & np.uint64(1)) == 0:
                    continue
                piece_ids[n] = piece_type
                color_ids[n] = color
                square_ids[n] = sq
                n += 1

    for i in range(n):
        p_i = int(piece_ids[i]); c_i = int(color_ids[i]); s_i = int(square_ids[i])
        for j in range(i + 1, n):
            p_j = int(piece_ids[j]); c_j = int(color_ids[j]); s_j = int(square_ids[j])
            if _attacks_xray(p_i, c_i, s_i, s_j) or _attacks_xray(p_j, c_j, s_j, s_i):
                attn_allowed[i, j] = True
                attn_allowed[j, i] = True

    return piece_ids, color_ids, square_ids, attn_allowed, n


@njit
def _linear(x, w):
    t = x.shape[0]
    in_dim = x.shape[1]
    out_dim = w.shape[0]
    y = np.zeros((t, out_dim), dtype=np.float32)
    for i in range(t):
        for o in range(out_dim):
            s = 0.0
            for k in range(in_dim):
                s += x[i, k] * w[o, k]
            y[i, o] = s
    return y


@njit
def _layer_norm_no_affine(x, eps):
    t = x.shape[0]
    d = x.shape[1]
    y = np.empty((t, d), dtype=np.float32)
    for i in range(t):
        mean = 0.0
        for k in range(d):
            mean += x[i, k]
        mean /= d
        var = 0.0
        for k in range(d):
            u = x[i, k] - mean
            var += u * u
        var /= d
        inv = 1.0 / math.sqrt(var + eps)
        for k in range(d):
            y[i, k] = (x[i, k] - mean) * inv
    return y


@njit
def _gelu_inplace(x):
    inv_sqrt2 = 0.7071067811865475
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = x[i, j]
            x[i, j] = 0.5 * v * (1.0 + math.erf(v * inv_sqrt2))


@njit
def _self_attention(x_norm, attn_allowed, wq, wk, wv, wo, num_heads):
    q = _linear(x_norm, wq)
    k = _linear(x_norm, wk)
    v = _linear(x_norm, wv)

    t = x_norm.shape[0]
    d = x_norm.shape[1]
    head_dim = d // num_heads
    scale = 1.0 / math.sqrt(float(head_dim))
    out = np.zeros((t, d), dtype=np.float32)
    scores = np.empty((t,), dtype=np.float32)
    probs = np.empty((t,), dtype=np.float32)

    for h in range(num_heads):
        off = h * head_dim
        for i in range(t):
            max_score = -1e30
            for j in range(t):
                if not attn_allowed[i, j]:
                    s = -1e9
                else:
                    s = 0.0
                    for u in range(head_dim):
                        s += q[i, off + u] * k[j, off + u]
                    s *= scale
                scores[j] = s
                if s > max_score:
                    max_score = s
            denom = 0.0
            for j in range(t):
                e = math.exp(scores[j] - max_score)
                probs[j] = e
                denom += e
            inv_denom = 1.0 / denom
            for u in range(head_dim):
                s = 0.0
                for j in range(t):
                    s += (probs[j] * inv_denom) * v[j, off + u]
                out[i, off + u] = s

    return _linear(out, wo)


@njit
def _forward_position(piece_ids, color_ids, square_ids, attn_allowed, n_tokens):
    x = np.zeros((n_tokens, EMBED_DIM), dtype=np.float32)
    for i in range(n_tokens):
        p = piece_ids[i]; c = color_ids[i]; s = square_ids[i]
        for d in range(EMBED_DIM):
            x[i, d] = PIECE_EMBED[p, d] + COLOR_EMBED[c, d] + SQUARE_EMBED[s, d]

    for l in range(NUM_LAYERS):
        y = _self_attention(_layer_norm_no_affine(x, 1e-5), attn_allowed[:n_tokens, :n_tokens],
                            WQ[l], WK[l], WV[l], WO[l], NUM_HEADS)
        x = x + y
        z = _linear(_layer_norm_no_affine(x, 1e-5), FF1[l])
        _gelu_inplace(z)
        z = _linear(z, FF2[l])
        x = x + z

    corr = 0.0
    for i in range(n_tokens):
        s = 0.0
        for d in range(EMBED_DIM):
            s += x[i, d] * SCORE_HEAD[d]
        corr += s if color_ids[i] == 0 else -s
    return corr


@njit
def _file_pawn_counts(pawn_bb):
    FILE_MASKS = np.array(
        [0x0101010101010101 << f for f in range(8)], dtype=np.uint64
    )
    counts = np.zeros(8, dtype=np.int32)
    bb = np.uint64(pawn_bb)
    for f in range(8):
        counts[f] = _popcount_u64(bb & FILE_MASKS[f])
    return counts


@njit
def _is_passed_pawn(sq, own_is_white, opp_pawns):
    file_idx = sq % 8
    opp = np.uint64(opp_pawns)
    for opp_sq in range(64):
        if ((opp >> np.uint64(opp_sq)) & np.uint64(1)) == 0:
            continue
        opp_file = opp_sq % 8
        if abs(opp_file - file_idx) > 1:
            continue
        if own_is_white:
            if opp_sq > sq:
                return False
        else:
            if opp_sq < sq:
                return False
    return True


@njit
def _baseline_feature_vector(board_pieces):
    PIECE_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
    score_fixed = 0.0
    features = np.zeros(11 + 6 * 64, dtype=np.float32)

    white_bishops = np.uint64(board_pieces[2])
    black_bishops = np.uint64(board_pieces[8])
    white_pawns = np.uint64(board_pieces[0])
    black_pawns = np.uint64(board_pieces[6])

    for pt in range(6):
        white_bb = np.uint64(board_pieces[pt])
        black_bb = np.uint64(board_pieces[pt + 6])

        score_fixed += PIECE_VALUES[pt] * (_popcount_u64(white_bb) - _popcount_u64(black_bb))

        for sq in range(64):
            if ((white_bb >> np.uint64(sq)) & np.uint64(1)) != 0:
                features[11 + pt * 64 + sq] += 1.0
            if ((black_bb >> np.uint64(sq)) & np.uint64(1)) != 0:
                features[11 + pt * 64 + (sq ^ 56)] -= 1.0

    doubled = 0
    isolated = 0
    passed = np.zeros(8, dtype=np.int32)
    wc = _file_pawn_counts(white_pawns)
    bc = _file_pawn_counts(black_pawns)

    for f in range(8):
        w = int(wc[f]); b = int(bc[f])
        if w > 1: doubled -= (w - 1)
        if b > 1: doubled += (b - 1)

        wl = int(wc[f - 1]) if f > 0 else 0
        wr = int(wc[f + 1]) if f < 7 else 0
        if w > 0 and wl == 0 and wr == 0: isolated -= w

        bl = int(bc[f - 1]) if f > 0 else 0
        br = int(bc[f + 1]) if f < 7 else 0
        if b > 0 and bl == 0 and br == 0: isolated += b

    for sq in range(64):
        if ((white_pawns >> np.uint64(sq)) & np.uint64(1)) != 0:
            if _is_passed_pawn(sq, True, black_pawns):
                passed[sq // 8] += 1
        if ((black_pawns >> np.uint64(sq)) & np.uint64(1)) != 0:
            if _is_passed_pawn(sq, False, white_pawns):
                passed[7 - (sq // 8)] -= 1

    features[0] = float(doubled)
    features[1] = float(isolated)
    features[2] = float((1 if _popcount_u64(white_bishops) >= 2 else 0) -
                        (1 if _popcount_u64(black_bishops) >= 2 else 0))
    for i in range(8):
        features[3 + i] = passed[i]

    return score_fixed, features


@njit
def _baseline_white_score(board_pieces, board_occupancy):
    score_fixed, feat = _baseline_feature_vector(board_pieces)
    s = score_fixed
    s += BASELINE_DOUBLED * feat[0]
    s += BASELINE_ISOLATED * feat[1]
    s += BASELINE_BISHOP * feat[2]

    for i in range(8):
        s += BASELINE_PASSED[i] * feat[3 + i]
    for i in range(BASELINE_PSTS.shape[0]):
        s += BASELINE_PSTS[i] * feat[11 + i]
    return s


@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    # Neural-network correction disabled for speed.
    # needed = _count_tokens_position(board_pieces)
    # cap = MAX_TOKENS if MAX_TOKENS > needed else needed
    # piece_ids, color_ids, square_ids, attn_allowed, n_tokens = _encode_position_tokens(board_pieces, cap)
    # corr_white = 0.0
    # if n_tokens > 0:
    #     corr_white = _forward_position(piece_ids, color_ids, square_ids, attn_allowed, n_tokens)

    base_white = _baseline_white_score(board_pieces, board_occupancy)
    white_score = base_white

    if side_to_move == 1:
        white_score = -white_score

    return np.int32(white_score)
