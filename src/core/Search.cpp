#include "Search.hpp"
#include "MoveGen.hpp"
#include "Attacks.hpp"

#include <iostream>
#include <vector>
#include <algorithm>


namespace Search {

    // Constants for infinity
    const int INF = 1000000;
    const int MATE_SCORE = 100000;

    // Helper to find the King (for mate detection)
    Square find_king(const BoardState& board, Colour side) {
        int king_idx = (side == Colour::White) ? 5 : 11;
        return static_cast<Square>(BitUtil::lsb(board.pieces[king_idx]));
    }

    // THE ALPHA-BETA ALGORITHM
    int negamax(BoardState& board, int depth, int alpha, int beta, EvalCallback evalFunc) {
        // 1. Base Case: Leaf Node
        if (depth == 0) {
            // Call Python!
            // Note: We need to ensure we pass the correct pointers.
            // board.pieces is std::array, so .data() works.
            return evalFunc(
                board.pieces.data(), 
                board.occupancy.data(), 
                static_cast<uint32_t>(board.full_move_number)
            );
        }

        // 2. Move Generation
        std::vector<Move> moves;
        moves.reserve(256);
        MoveGen::generate_moves(board, moves);

        int legal_moves = 0;
        int best_score = -INF;

        // 3. Iterate Moves
        for (const auto& move : moves) {
            board.make_move(move);

            // Filter Illegal Moves (King Safety)
            Colour enemy = board.to_move; // The side that just moved is now the 'enemy' of the new turn
            Square king_sq = find_king(board, (enemy == Colour::White) ? Colour::Black : Colour::White);
            
            if (Attacks::is_square_attacked(king_sq, enemy, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            legal_moves++;

            // Recursion: Flip Alpha/Beta and negate score
            int score = -negamax(board, depth - 1, -beta, -alpha, evalFunc);
            
            board.undo_move(move);

            // 4. Alpha-Beta Pruning
            if (score > best_score) {
                best_score = score;
            }
            if (score > alpha) {
                alpha = score;
            }
            if (alpha >= beta) {
                break; // Beta Cutoff (Too good, opponent won't allow this)
            }
        }

        // 5. Mate / Stalemate Detection
        if (legal_moves == 0) {
            Square my_king = find_king(board, board.to_move);
            Colour enemy = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            
            if (Attacks::is_square_attacked(my_king, enemy, board.pieces.data(), board.occupancy[2])) {
                return -MATE_SCORE + depth; // Checkmate (prefer faster mates)
            } else {
                return 0; // Stalemate
            }
        }

        return best_score;
    }

    // MAIN ENTRY POINT
    Move iterative_deepening(BoardState& board, const SearchParams& params) {
        Move best_move;
        int alpha = -INF;
        int beta = INF;
        
        // Simple fixed-depth search for now (Depth 1 to params.depth)
        // For the competition, you might just run straight to params.depth
        
        std::vector<Move> moves;
        MoveGen::generate_moves(board, moves);

        int best_score = -INF;

        for (const auto& move : moves) {
            board.make_move(move);

            // Illegal move check
            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
            Square king_sq = find_king(board, us);
            if (Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
                board.undo_move(move);
                continue;
            }

            // Call Negamax
            int score = -negamax(board, params.depth - 1, -beta, -alpha, params.evalFunc);
            
            board.undo_move(move);

            std::cout << "Move: " << move.from() << "->" << move.to() << " Score: " << score << std::endl;

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            if (score > alpha) alpha = score;
        }

        return best_move;
    }
}
