#include "Search.hpp"
#include "MoveGen.hpp"
#include "Attacks.hpp"
#include <vector>

namespace Search {

    const int INF = 1000000;
    const int MATE_SCORE = 100000;

    Square find_king(const BoardState& board, Colour side) {
        int king_idx = (side == Colour::White) ? 5 : 11;
        return static_cast<Square>(BitUtil::lsb(board.pieces[king_idx]));
    }

    int negamax(BoardState& board, int depth, int alpha, int beta, EvalCallback evalFunc) {
        if (depth == 0) {
            return evalFunc(board.pieces.data(), board.occupancy.data(), static_cast<uint32_t>(board.full_move_number));
        }

        std::vector<Move> moves;
        moves.reserve(256);
        MoveGen::generate_moves(board, moves);

        int legal_moves = 0;
        int best_score = -INF;

        for (const auto& move : moves) {
            board.make_move(move);

            Colour enemy = board.to_move;
            Square king_sq = find_king(board, (enemy == Colour::White) ? Colour::Black : Colour::White);
            
            if (Attacks::is_square_attacked(king_sq, enemy, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            legal_moves++;
            int score = -negamax(board, depth - 1, -beta, -alpha, evalFunc);
            board.undo_move(move);

            if (score > best_score) best_score = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break; 
        }

        if (legal_moves == 0) {
            Square my_king = find_king(board, board.to_move);
            Colour enemy = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            
            if (Attacks::is_square_attacked(my_king, enemy, board.pieces.data(), board.occupancy[2])) {
                return -MATE_SCORE + depth; 
            } else {
                return 0; // Stalemate
            }
        }

        return best_score;
    }

    Move iterative_deepening(BoardState& board, const SearchParams& params) {
        Move best_move;
        int alpha = -INF;
        int beta = INF;
        
        std::vector<Move> moves;
        MoveGen::generate_moves(board, moves);

        int best_score = -INF;

        for (const auto& move : moves) {
            board.make_move(move);

            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            Square king_sq = find_king(board, us);
            
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            int score = -negamax(board, params.depth - 1, -beta, -alpha, params.evalFunc);
            board.undo_move(move);

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            if (score > alpha) alpha = score;
        }

        return best_move;
    }
}
