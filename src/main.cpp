#include "BoardState.hpp"
#include "Types.hpp"
#include "MoveGen.hpp"
#include "Attacks.hpp"
#include "Search.hpp"

#include <iostream>
#include <chrono>
#include <vector>


Square find_king(const BoardState& board, Colour side) {
    int king_idx = (side == Colour::White) ? 5 : 11;
    Bitboard b = board.pieces[king_idx];
    return static_cast<Square>(BitUtil::lsb(b));
}

uint64_t perft(BoardState& board, int depth) {
    if (depth == 0) return 1ULL;

    std::vector<Move> moves;
    moves.reserve(256);
    MoveGen::generate_moves(board, moves);

    uint64_t nodes = 0;

    for (const auto& move : moves) {
        // 1. Make Move
        board.make_move(move);
        
        // 2. CHECK LEGALITY
        // We just moved. It is now the *enemy's* turn in board.to_move.
        // So we need to check if *our* King (the side that just moved) is attacked by the *enemy* (side to move).
        
        Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
        Square king_sq = find_king(board, us);
        
        // If our King is attacked by the side currently to move (the enemy), the move was illegal.
        if (!Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2])) {
            nodes += perft(board, depth - 1);
        }
        
        // 3. Undo
        board.undo_move(move);
    }
    return nodes;
}

void run_debug_perft(int depth) {
    BoardState board; 
    
    board.pieces[0] = 0x000000000000FF00ULL; board.pieces[6] = 0x00FF000000000000ULL; // Pawns
    board.pieces[1] = 0x0000000000000042ULL; board.pieces[7] = 0x4200000000000000ULL; // Knights
    board.pieces[2] = 0x0000000000000024ULL; board.pieces[8] = 0x2400000000000000ULL; // Bishops
    board.pieces[3] = 0x0000000000000081ULL; board.pieces[9] = 0x8100000000000000ULL; // Rooks
    board.pieces[4] = 0x0000000000000008ULL; board.pieces[10] = 0x0800000000000000ULL; // Queen
    board.pieces[5] = 0x0000000000000010ULL; board.pieces[11] = 0x1000000000000000ULL; // King

    board.occupancy[0] = 0ULL; board.occupancy[1] = 0ULL;
    for (int i = 0; i < 6; ++i) board.occupancy[0] |= board.pieces[i];
    for (int i = 6; i < 12; ++i) board.occupancy[1] |= board.pieces[i];
    board.occupancy[2] = board.occupancy[0] | board.occupancy[1];

    board.to_move = Colour::White;
    board.castle_rights = 0b1111; 
    board.full_move_number = 1;

    std::cout << "Starting Perft (Depth " << depth << ")..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t nodes = perft(board, depth);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Nodes: " << nodes << " | Time: " << elapsed.count() << "s" << std::endl;
}


// Param 1: Pointer to 12 bitboards (Pieces)
// Param 2: Pointer to 3 bitboards (Occupancy: White, Black, All)
// Param 3: Current Move Number (32-bit int)
using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

extern "C" {
    void startEngine(EvalCallback evalFunc) {
        Attacks::init();
        BoardState board; 
        // ... (Your board init code) ...

        std::cout << "--- ENGINE STARTED ---" << std::endl;

        // Run Search at Depth 3
        Search::SearchParams params;
        params.depth = 3;
        params.evalFunc = evalFunc;

        Move best = Search::iterative_deepening(board, params);
        
        std::cout << "Best Move Found: " << static_cast<int>(best.from()) 
                  << " -> " << static_cast<int>(best.to()) << std::endl;
    }
}

// --- STANDALONE C++ ENTRY POINT (For testing without Python) ---
int main() {
    Attacks::init();
    std::cout << "Testing Bishop Attack at D4 (Empty Board)..." << std::endl;
    // Square D4 is index 27
    uint64_t attacks = Attacks::get_bishop_attacks(27, 0ULL); 
    Attacks::print_bitboard(attacks);

    std::cout << "Testing Rook Attack at D4 (Empty Board)..." << std::endl;
    uint64_t r_attacks = Attacks::get_rook_attacks(27, 0ULL);
    Attacks::print_bitboard(r_attacks);

    std::cout << "Running Standalone Chess Engine Test..." << std::endl;
    
    // Depth 1: Should be 20 nodes
    run_debug_perft(1);

    // Depth 2: Should be 400 nodes
    run_debug_perft(2);

    // Depth 3: Should be 8902 nodes
    run_debug_perft(3);

    // Depth 4
    run_debug_perft(4);

    // Depth 5
    run_debug_perft(5);

    // Depth 6
    // run_debug_perft(6);
    
    return 0;
}
