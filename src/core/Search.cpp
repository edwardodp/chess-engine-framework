#include "Search.hpp"
#include "MoveGen.hpp"
#include "BoardState.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp" 
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace Search {

    // --- MVV-LVA LOOKUP ---
    const int mvv_lva[6][6] = {
        {105, 205, 305, 405, 505, 605}, // Victim P
        {104, 204, 304, 404, 504, 604}, // Victim N
        {103, 203, 303, 403, 503, 603}, // Victim B
        {102, 202, 302, 402, 502, 602}, // Victim R
        {101, 201, 301, 401, 501, 601}, // Victim Q
        {100, 200, 300, 400, 500, 600}  // Victim K
    };

    // --- KILLER MOVES ---
    // Two killer slots per ply. Killers are quiet moves that caused beta cutoffs.
    static constexpr int MAX_PLY = 128;
    static Move killers[MAX_PLY][2];

    // --- HISTORY HEURISTIC ---
    // history[side][from_sq][to_sq] — incremented when a quiet move causes a cutoff
    static int history[2][64][64];

    static void clear_heuristics() {
        std::memset(killers, 0, sizeof(killers));
        std::memset(history, 0, sizeof(history));
    }

    static void store_killer(const Move& m, int ply) {
        if (ply >= MAX_PLY) return;
        // Don't store duplicates
        if (killers[ply][0].raw() == m.raw()) return;
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = m;
    }

    static void update_history(const Move& m, Colour side, int depth) {
        int s = (side == Colour::White) ? 0 : 1;
        int from = static_cast<int>(m.from());
        int to   = static_cast<int>(m.to());
        // Bonus proportional to depth^2 (deeper cutoffs are more valuable)
        history[s][from][to] += depth * depth;
        // Prevent overflow — cap and age
        if (history[s][from][to] > 400000) {
            for (auto& row : history[s])
                for (auto& v : row)
                    v >>= 1;
        }
    }

    int get_piece_type(const BoardState& board, Square sq) {
        for (int i = 0; i < 6; ++i) {
            if (BitUtil::get_bit(board.pieces[i], sq) || BitUtil::get_bit(board.pieces[i+6], sq)) {
                return i;
            }
        }
        return 0;
    }

    int score_move(const Move& m, const BoardState& board, int ply) {
        // 1. Captures: MVV-LVA (highest priority)
        if (m.is_capture()) {
            int attacker = get_piece_type(board, m.from());
            int victim = 0; 
            if (m.flag() != MoveFlag::EnPassant) {
                victim = get_piece_type(board, m.to());
            }
            return 10000 + mvv_lva[victim][attacker];
        }
        
        // 2. Promotions
        if (m.is_promotion()) {
            return 9000;
        }

        // 3. Killer moves (quiet moves that caused cutoffs at this ply)
        if (ply < MAX_PLY) {
            if (m.raw() == killers[ply][0].raw()) return 8000;
            if (m.raw() == killers[ply][1].raw()) return 7000;
        }

        // 4. History heuristic (quiet move ordering)
        int side = (board.to_move == Colour::White) ? 0 : 1;
        return history[side][static_cast<int>(m.from())][static_cast<int>(m.to())];
    }

    // --- Quiescence Search ---
    static constexpr int QS_MAX_DEPTH = 8;
    static constexpr int DELTA_MARGIN  = 900;

    int32_t quiescence(BoardState& board, int32_t alpha, int32_t beta, EvalCallback eval, uint32_t moves_played, int qs_depth) {
        int32_t stand_pat = eval(board.pieces.data(), board.occupancy.data(), (board.to_move == Colour::White ? 0 : 1));
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;

        if (qs_depth >= QS_MAX_DEPTH) return alpha;

        std::vector<Move> moves;
        moves.reserve(32); 
        MoveGen::generate_captures(board, moves);

        // Sort captures by MVV-LVA (ply doesn't matter for captures, pass 0)
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            return score_move(a, board, 0) > score_move(b, board, 0);
        });

        for (const auto& move : moves) {
            if (!move.is_promotion() && stand_pat + DELTA_MARGIN < alpha) {
                break;
            }

            board.make_move(move);
            
            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            Square king_sq = find_king(board, us);
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            int32_t score = -quiescence(board, -beta, -alpha, eval, moves_played + 1, qs_depth + 1);
            board.undo_move(move);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    // --- Main Alpha-Beta with PVS ---
    int32_t alpha_beta(BoardState& board, int depth, int32_t alpha, int32_t beta, EvalCallback eval, int ply) {
        if (ply > 0 && board.is_draw()) {
            return 0;
        }

        if (depth == 0) {
            return quiescence(board, alpha, beta, eval, ply, 0);
        }

        std::vector<Move> moves;
        MoveGen::generate_moves(board, moves);

        // Sort moves: captures (MVV-LVA) > promotions > killers > history
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            return score_move(a, board, ply) > score_move(b, board, ply);
        });

        int legal_moves = 0;
        Colour us_before_move = board.to_move;

        for (const auto& move : moves) {
            board.make_move(move);

            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White; 
            Square king_sq = find_king(board, us);
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            int32_t score;
            if (legal_moves == 0) {
                // First legal move (expected best) — search with full window
                score = -alpha_beta(board, depth - 1, -beta, -alpha, eval, ply + 1);
            } else {
                // PVS: search with null window first
                score = -alpha_beta(board, depth - 1, -alpha - 1, -alpha, eval, ply + 1);
                // If it beats alpha but not beta, re-search with full window
                if (score > alpha && score < beta) {
                    score = -alpha_beta(board, depth - 1, -beta, -alpha, eval, ply + 1);
                }
            }

            board.undo_move(move);
            legal_moves++;

            if (score >= beta) {
                // Beta cutoff — update killer and history for quiet moves
                if (!move.is_capture() && !move.is_promotion()) {
                    store_killer(move, ply);
                    update_history(move, us_before_move, depth);
                }
                return beta;
            }
            if (score > alpha) {
                alpha = score;
            }
        }

        if (legal_moves == 0) {
            Colour us = board.to_move;
            Square king_sq = find_king(board, us);
            bool in_check = Attacks::is_square_attacked(king_sq, (us == Colour::White ? Colour::Black : Colour::White), board.pieces.data(), board.occupancy[2]);
            if (in_check) return -100000 + ply; 
            return 0;
        }

        return alpha;
    }

    // --- Iterative Deepening with PVS at root ---
    Move iterative_deepening(BoardState& board, const SearchParams& params, SearchStats& stats) {
        Move best_move;
        
        stats.depth_reached = 0;
        stats.score = 0;

        // Clear killer and history tables at the start of each search
        clear_heuristics();

        for (int d = 1; d <= params.depth; ++d) {
            int32_t alpha = -200000;
            int32_t beta = 200000;
            
            std::vector<Move> moves;
            MoveGen::generate_moves(board, moves);
            
            // Sort moves — at root, also boost the previous iteration's best move
            uint16_t prev_best_raw = best_move.raw();
            std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
                // Previous best move gets highest priority
                int sa = (a.raw() == prev_best_raw) ? 100000 : score_move(a, board, 0);
                int sb = (b.raw() == prev_best_raw) ? 100000 : score_move(b, board, 0);
                return sa > sb;
            });

            Move current_best_move;
            int32_t best_score = -200000;
            int legal_moves = 0;
            Colour us_before_move = board.to_move;

            for (const auto& move : moves) {
                board.make_move(move);
                
                Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
                Square king_sq = find_king(board, us);
                if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                    board.undo_move(move);
                    continue;
                }

                int32_t score;
                if (legal_moves == 0) {
                    score = -alpha_beta(board, d - 1, -beta, -alpha, params.evalFunc, 1);
                } else {
                    // PVS at root
                    score = -alpha_beta(board, d - 1, -alpha - 1, -alpha, params.evalFunc, 1);
                    if (score > alpha && score < beta) {
                        score = -alpha_beta(board, d - 1, -beta, -alpha, params.evalFunc, 1);
                    }
                }

                board.undo_move(move);
                legal_moves++;

                if (score > best_score) {
                    best_score = score;
                    current_best_move = move;
                }
                if (score > alpha) {
                    alpha = score;
                }
            }

            if (current_best_move.raw() != 0) {
                best_move = current_best_move;
                
                stats.depth_reached = d;
                stats.score = best_score;
                stats.best_move_raw = best_move.raw();
            }
        }
        return best_move;
    }

    Square find_king(const BoardState& board, Colour side) {
        int idx = (side == Colour::White) ? 5 : 11;
        if (board.pieces[idx] == 0) return Square::None; 
        return static_cast<Square>(BitUtil::lsb(board.pieces[idx]));
    }
}
