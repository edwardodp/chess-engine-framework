#include "Interface.hpp"
#include "Evaluation.hpp"
#include "MoveGen.hpp"
#include "Attacks.hpp"
#include "Zobrist.hpp"
#include "BitUtil.hpp"
#include <iostream>
#include <string>
#include <cstdint>
#include <atomic>
#include <vector>

static Search::EvalCallback global_white_eval = nullptr;
static Search::EvalCallback global_black_eval = nullptr;

std::atomic<int> g_current_searcher{0};

int cpp_dispatcher(const uint64_t* pieces, const uint64_t* occupancy, int side) {
    int searcher = g_current_searcher.load(std::memory_order_relaxed);
    Search::EvalCallback fn = (searcher == 0) ? global_white_eval : global_black_eval;
    if (fn) return fn(pieces, occupancy, side);
    return 0;
}

extern "C" {
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    
    void startEngine(Search::EvalCallback whiteFunc, Search::EvalCallback blackFunc, int depth, int human_side, const char* fen) {
        global_white_eval = whiteFunc;
        global_black_eval = blackFunc;

        std::string fen_str = (fen != nullptr) ? std::string(fen) : "startpos";
        
        GUI::Launch((Search::EvalCallback)cpp_dispatcher, depth, human_side, fen_str);
    }

    // Headless game: returns 0=draw, 1=white win, 2=black win, -1=exceeded max moves
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    int runHeadlessGame(Search::EvalCallback whiteFunc, Search::EvalCallback blackFunc,
                        int depth, const char* fen, int max_moves) {
        global_white_eval = whiteFunc;
        global_black_eval = blackFunc;

        Attacks::init();
        Zobrist::init();

        BoardState board;
        std::string fen_str = (fen != nullptr) ? std::string(fen) : "startpos";

        if (fen_str.empty() || fen_str == "startpos") {
            board.pieces[0]  = 0x000000000000FF00ULL; board.pieces[6]  = 0x00FF000000000000ULL;
            board.pieces[1]  = 0x0000000000000042ULL; board.pieces[7]  = 0x4200000000000000ULL;
            board.pieces[2]  = 0x0000000000000024ULL; board.pieces[8]  = 0x2400000000000000ULL;
            board.pieces[3]  = 0x0000000000000081ULL; board.pieces[9]  = 0x8100000000000000ULL;
            board.pieces[4]  = 0x0000000000000008ULL; board.pieces[10] = 0x0800000000000000ULL;
            board.pieces[5]  = 0x0000000000000010ULL; board.pieces[11] = 0x1000000000000000ULL;
            board.occupancy[0] = 0ULL; board.occupancy[1] = 0ULL;
            for (int i = 0;  i < 6;  ++i) board.occupancy[0] |= board.pieces[i];
            for (int i = 6;  i < 12; ++i) board.occupancy[1] |= board.pieces[i];
            board.occupancy[2] = board.occupancy[0] | board.occupancy[1];
            board.to_move = Colour::White;
            board.castle_rights = 0b1111;
            board.full_move_number = 1;
            board.refresh_hash();
        } else {
            board.load_fen(fen_str);
        }

        for (int move_num = 0; move_num < max_moves; ++move_num) {
            if (board.is_draw()) return 0;

            // Check for any legal move
            std::vector<Move> all_moves;
            MoveGen::generate_moves(board, all_moves);

            bool has_legal = false;
            for (const auto& m : all_moves) {
                board.make_move(m);
                Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
                Square k = Search::find_king(board, us);
                bool illegal = Attacks::is_square_attacked(
                    k, board.to_move, board.pieces.data(), board.occupancy[2]);
                board.undo_move(m);
                if (!illegal) { has_legal = true; break; }
            }

            if (!has_legal) {
                Colour us   = board.to_move;
                Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
                Square k    = Search::find_king(board, us);
                if (Attacks::is_square_attacked(k, them, board.pieces.data(), board.occupancy[2]))
                    return (us == Colour::White) ? 2 : 1;  // Checkmate
                return 0;  // Stalemate
            }

            // Run search
            g_current_searcher.store(
                (board.to_move == Colour::White) ? 0 : 1,
                std::memory_order_relaxed);

            Search::SearchParams params;
            params.depth    = depth;
            params.evalFunc = (Search::EvalCallback)cpp_dispatcher;

            Search::SearchStats stats;
            Move best = Search::iterative_deepening(board, params, stats);
            if (best.raw() == 0) return 0;

            board.make_move(best);
        }

        return -1;  // Exceeded max moves
    }
}

#ifndef BUILD_AS_LIBRARY
int main() {
    std::cout << "Starting Standalone Engine (Human vs Bot)" << std::endl;
    GUI::Launch(Evaluation::evaluate, 5, 0, "startpos"); 
    return 0;
}
#endif
