#include "BoardState.hpp"
#include "Types.hpp"
#include "Search.hpp"
#include "Attacks.hpp"
#include <iostream>

// Callback signature matches Python's ctypes definition
using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

extern "C" {
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    void startEngine(EvalCallback evalFunc) {
        // 1. Initialize Static Tables
        Attacks::init();

        // 2. Setup Board (Standard Starting Position)
        BoardState board;
        
        // White Pieces
        board.pieces[0] = 0x000000000000FF00ULL; // Pawns
        board.pieces[1] = 0x0000000000000042ULL; // Knights
        board.pieces[2] = 0x0000000000000024ULL; // Bishops
        board.pieces[3] = 0x0000000000000081ULL; // Rooks
        board.pieces[4] = 0x0000000000000008ULL; // Queen
        board.pieces[5] = 0x0000000000000010ULL; // King

        // Black Pieces
        board.pieces[6] = 0x00FF000000000000ULL; // Pawns
        board.pieces[7] = 0x4200000000000000ULL; // Knights
        board.pieces[8] = 0x2400000000000000ULL; // Bishops
        board.pieces[9] = 0x8100000000000000ULL; // Rooks
        board.pieces[10] = 0x0800000000000000ULL; // Queen
        board.pieces[11] = 0x1000000000000000ULL; // King
        
        // Initialize Occupancy Bitboards
        board.occupancy[0] = 0ULL; // White
        board.occupancy[1] = 0ULL; // Black
        for (int i = 0; i < 6; ++i) board.occupancy[0] |= board.pieces[i];
        for (int i = 6; i < 12; ++i) board.occupancy[1] |= board.pieces[i];
        board.occupancy[2] = board.occupancy[0] | board.occupancy[1]; // Both

        // Game State
        board.to_move = Colour::White;
        board.castle_rights = 0b1111; // KQkq
        board.full_move_number = 1;

        // 3. Configure Search
        Search::SearchParams params;
        params.depth = 6;
        params.evalFunc = evalFunc;

        // 4. Execute Search
        Move best = Search::iterative_deepening(board, params);
        
        // 5. Output Result
        // We cast to int to print raw indices (0-63)
        std::cout << "Best Move: " << static_cast<int>(best.from()) 
                  << " -> " << static_cast<int>(best.to()) << std::endl;
    }
}

// --- STANDALONE TEST HARNESS ---
// This code is ONLY compiled when building the 'ChessTest' executable.
// It is excluded from the 'ChessLib' shared library used by Python.
#ifndef BUILD_AS_LIBRARY

// Simple evaluation for C++-only testing
int32_t dummyEval(const uint64_t* pieces, const uint64_t* occupancy, uint32_t moveCount) {
    int32_t score = 0;
    // P=100, N=300, B=320, R=500, Q=900, K=20000
    int values[] = {100, 300, 320, 500, 900, 20000}; 

    for (int i = 0; i < 6; ++i) {
        // Count White
        uint64_t w = pieces[i];
        while(w) { w &= (w-1); score += values[i]; }

        // Count Black
        uint64_t b = pieces[i+6];
        while(b) { b &= (b-1); score -= values[i]; }
    }
    return score;
}

int main() {
    std::cout << "Running Standalone Search Test..." << std::endl;
    startEngine(dummyEval);
    return 0;
}
#endif
