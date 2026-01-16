#include "Search.hpp"
#include "MoveGen.hpp"
#include "BoardState.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp" 
#include <vector>
#include <algorithm>
#include <iostream>

namespace Search {

    // --- MVV-LVA LOOKUP ---
    // Victim (Rows): P, N, B, R, Q, K
    // Attacker (Cols): P, N, B, R, Q, K
    // We want PxQ to be highest, QxP to be lowest.
    const int mvv_lva[6][6] = {
        {105, 205, 305, 405, 505, 605}, // Victim P, Attacker P, N, B, R, Q, K
        {104, 204, 304, 404, 504, 604}, // Victim N
        {103, 203, 303, 403, 503, 603}, // Victim B
        {102, 202, 302, 402, 502, 602}, // Victim R
        {101, 201, 301, 401, 501, 601}, // Victim Q
        {100, 200, 300, 400, 500, 600}  // Victim K
    };

    int get_piece_type(const BoardState& board, Square sq) {
        // Returns 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K
        for (int i = 0; i < 6; ++i) {
            if (BitUtil::get_bit(board.pieces[i], sq) || BitUtil::get_bit(board.pieces[i+6], sq)) {
                return i;
            }
        }
        return 0;
    }

    int score_move(const Move& m, const BoardState& board) {
        if (m.is_capture()) {
            int attacker = get_piece_type(board, m.from());
            
            int victim = 0; 
            if (m.flag() != MoveFlag::EnPassant) {
                victim = get_piece_type(board, m.to());
            }

            // Look up MVV-LVA score (Victim dominant)
            // Add huge offset (10000) so captures are always checked before quiet moves
            return 10000 + mvv_lva[victim][attacker];
        }
        
        if (m.is_promotion()) {
            return 9000;
        }

        return 0;
    }

    // --- Quiescence Search ---
    int32_t quiescence(BoardState& board, int32_t alpha, int32_t beta, EvalCallback eval, uint32_t moves_played) {
        // 1. Stand Pat
        int32_t stand_pat = eval(board.pieces.data(), board.occupancy.data(), (board.to_move == Colour::White ? 0 : 1));
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;

        // 2. Generate Captures
        std::vector<Move> moves;
        moves.reserve(32); 
        MoveGen::generate_captures(board, moves);

        // 3. Fast Sort (MVV-LVA)
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            return score_move(a, board) > score_move(b, board);
        });

        for (const auto& move : moves) {
            board.make_move(move);
            
            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            Square king_sq = find_king(board, us);
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            int32_t score = -quiescence(board, -beta, -alpha, eval, moves_played + 1);
            board.undo_move(move);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    // --- Main Search ---
    int32_t alpha_beta(BoardState& board, int depth, int32_t alpha, int32_t beta, EvalCallback eval, uint32_t moves_played) {
        if (depth == 0) {
            return quiescence(board, alpha, beta, eval, moves_played);
        }

        std::vector<Move> moves;
        MoveGen::generate_moves(board, moves);

        // Sort moves using MVV-LVA
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            return score_move(a, board) > score_move(b, board);
        });

        int legal_moves = 0;

        for (const auto& move : moves) {
            board.make_move(move);

            // Illegal Check Filter
            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White; 
            Square king_sq = find_king(board, us);
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            legal_moves++;
            int32_t score = -alpha_beta(board, depth - 1, -beta, -alpha, eval, moves_played + 1);
            board.undo_move(move);

            if (score >= beta) return beta;
            if (score > alpha) {
                alpha = score;
            }
        }

        if (legal_moves == 0) {
            Colour us = board.to_move;
            Square king_sq = find_king(board, us);
            bool in_check = Attacks::is_square_attacked(king_sq, (us == Colour::White ? Colour::Black : Colour::White), board.pieces.data(), board.occupancy[2]);
            if (in_check) return -100000 + moves_played; 
            return 0; // Stalemate
        }

        return alpha;
    }

    Move iterative_deepening(BoardState& board, const SearchParams& params, SearchStats& stats) {
        Move best_move;
        
        // Reset Stats
        stats.depth_reached = 0;
        stats.score = 0;

        for (int d = 1; d <= params.depth; ++d) {
            int32_t alpha = -200000;
            int32_t beta = 200000;
            
            std::vector<Move> moves;
            MoveGen::generate_moves(board, moves);
            
            // Sort moves
            std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
                return score_move(a, board) > score_move(b, board);
            });

            Move current_best_move;
            int32_t best_score = -200000;

            for (const auto& move : moves) {
                board.make_move(move);
                
                Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
                Square king_sq = find_king(board, us);
                if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                    board.undo_move(move);
                    continue;
                }

                int32_t score = -alpha_beta(board, d - 1, -beta, -alpha, params.evalFunc, 1);
                board.undo_move(move);

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
                
                // std::cout << "Info: Depth " << d << " Score: " << best_score << std::endl;
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
