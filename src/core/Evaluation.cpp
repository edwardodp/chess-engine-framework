#include "Evaluation.hpp"
#include "BitUtil.hpp"
#include "BoardState.hpp"
#include "MoveGen.hpp"
#include "Attacks.hpp"
#include "Types.hpp"
#include "Search.hpp"
#include <vector>
#include <algorithm>

namespace Evaluation {

    // --- 1. PEICE-SQUARE TABLES (PeSTO) ---
    // These tables give a bonus (or penalty) for every square on the board.
    // We have two values for every square: Middle Game (MG) and End Game (EG).
    // The engine blends these based on how many pieces are left.

    // Bonus tables are defined for WHITE. Black mirrors them.
    // Format: { MG, EG } for A1...H8
    
    // Pawn: Encourages pushing center pawns, punishing backward pawns
    const int val_pawn[64][2] = {
        { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0},
        {98, 0}, {134, 0}, {61, 0}, {95, 0}, {68, 0}, {126, 0}, {34, 0}, {-11, 0},
        {-6, 0}, { 7, 0}, {26, 0}, {31, 0}, {65, 0}, {56, 0}, {25, 0}, {-20, 0},
        {-14, 0}, {13, 0}, { 6, 0}, {21, 0}, {23, 0}, {12, 0}, {17, 0}, {-23, 0},
        {-27, 0}, {-2, 0}, {-5, 0}, {12, 0}, {17, 0}, { 6, 0}, {10, 0}, {-25, 0},
        {-26, 0}, {-4, 0}, {-4, 0}, {-10, 0}, { 3, 0}, { 3, 0}, {33, 0}, {-12, 0},
        { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, // Rank 7 (handled by search usually)
        { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}, { 0, 0}
    };

    // Knight: Loves the center, hates corners
    const int val_knight[64][2] = {
        {-167, -58}, {-89, -38}, {-34, -13}, {-49, -28}, { 61, -31}, {-38, -27}, {-126, -63}, {-210, -80},
        {-86, -25}, {-63, -6}, {-12, 12}, {-33, 2}, { 45, 25}, { 52, 28}, {-24, 0}, {-109, -27},
        {-18, -26}, {-4, 3}, { 21, 44}, {-28, 43}, { 8, 45}, { 7, 35}, { 32, 24}, {-53, -10},
        {-3, -20}, { 5, 29}, { 18, 56}, { 69, 56}, { 42, 68}, { 70, 50}, { 58, 36}, {-13, -11},
        {-14, -20}, { 5, 11}, { 6, 31}, { 39, 44}, { 52, 59}, { 65, 48}, { 34, 40}, {-30, -18},
        {-29, -15}, {-53, 20}, {-12, 33}, { 3, 33}, { 30, 48}, { 42, 36}, { 6, 26}, {-45, -8},
        {-53, -15}, {-29, 6}, {-12, 14}, { -8, 23}, { -1, 30}, { 32, 22}, {-10, 11}, {-85, -16},
        {-105, -34}, {-21, -26}, {-58, -7}, {-33, 6}, { 2, 8}, {-28, -2}, {-48, -29}, {-141, -29}
    };

    // Bishop: Loves long diagonals
    const int val_bishop[64][2] = {
        {-29, -6}, { 4, -19}, {-82, -18}, {-37, -6}, {-25, 9}, {-42, -5}, { 7, -24}, {-5, -14},
        {-26, -3}, { 16, -2}, {-62, -3}, {-1, -1}, {-10, 5}, {-24, 6}, { 12, -7}, { 1, -11},
        {-11, -7}, {-8, -2}, {-23, 6}, { 39, 14}, { -9, 21}, { 31, 19}, { 24, -2}, {-10, -5},
        { 22, -2}, { -3, 6}, { 13, 11}, { 11, 23}, { 16, 29}, { 12, 20}, { 5, 5}, { 6, -6},
        { -4, -4}, { 4, 3}, { 19, 14}, { 50, 31}, { 23, 31}, { 16, 17}, { -1, 4}, { -8, -6},
        { -6, -4}, { -6, 2}, { 7, 2}, { 19, 11}, { 27, 24}, { 10, 17}, { -5, 3}, { -19, -4},
        { -15, -9}, { -1, -9}, { 24, 3}, { 10, 11}, { 29, 19}, { -1, 8}, { -21, -6}, { 16, -11},
        { -4, -14}, { 40, -13}, { -6, -17}, { -20, -5}, { 2, 4}, { -2, -6}, { 12, -21}, { -23, -19}
    };

    // Rook: Likes 7th rank, center files
    const int val_rook[64][2] = {
        { 32, 13}, { 42, 10}, { 32, 18}, { 51, 15}, { 63, 12}, { 9, 21}, { 31, 11}, { 43, 12},
        { 27, 10}, { 32, 10}, { 58, 16}, { 62, 15}, { 80, 12}, { 67, 20}, { 23, 15}, { 44, 12},
        { -5, 6}, { 19, 6}, { 26, 11}, { 36, 17}, { 17, 16}, { 45, 17}, { 61, 5}, { 16, 6},
        { -24, 7}, { -11, 5}, { 7, 12}, { 26, 14}, { 24, 15}, { 35, 12}, { -8, 5}, { -20, 5},
        { -36, 3}, { -26, 6}, { -12, 12}, { 1, 14}, { 9, 15}, { -7, 12}, { 6, 5}, { -23, 4},
        { -45, 2}, { -25, 4}, { -16, 12}, { -17, 13}, { 3, 14}, { 0, 11}, { -5, 4}, { -33, 3},
        { -44, 0}, { -16, 6}, { -20, 10}, { -9, 11}, { -1, 14}, { 11, 11}, { -6, 5}, { -71, 0},
        { -19, 0}, { -13, 2}, { 1, 7}, { 17, 9}, { 16, 8}, { 7, 11}, { -37, 7}, { -26, -7}
    };

    // Queen: Keeps her safe early, dominates late
    const int val_queen[64][2] = {
        { -28, -9}, { 0, -55}, { 29, -43}, { 12, -31}, { 59, -15}, { 44, -18}, { 43, -29}, { 45, -23},
        { -24, -5}, { -39, -29}, { -5, -31}, { 1, -26}, { -16, -11}, { 57, 11}, { -4, 0}, { -4, -13},
        { -13, -9}, { -17, -35}, { 7, -19}, { 8, -13}, { 29, 7}, { 56, 12}, { 47, 7}, { 57, 11},
        { -27, -5}, { -27, -21}, { -16, -9}, { -16, 14}, { -1, 11}, { 17, 15}, { -2, 7}, { 1, 0},
        { -9, -3}, { -26, -12}, { -9, 0}, { -10, 14}, { -2, 23}, { -4, 21}, { 3, 11}, { -3, -5},
        { -14, -6}, { 2, -14}, { -11, 1}, { -2, 10}, { -5, 25}, { 2, 18}, { 14, 2}, { 5, -8},
        { -35, -4}, { -8, -14}, { 11, 1}, { 2, 10}, { 8, 20}, { 15, 17}, { -3, 1}, { 1, -11},
        { -2, -13}, { -27, -15}, { -6, -8}, { -13, 0}, { -17, 3}, { -7, 6}, { -9, -9}, { -24, -26}
    };

    // King: Hides in corner (MG), Centers board (EG)
    const int val_king[64][2] = {
        {-65, -74}, { 23, -35}, { 16, -18}, {-15, -18}, {-56, -11}, {-34, 15}, { 2, 4}, { 13, -22},
        { 29, -12}, {-1, -12}, {-20, 8}, { -7, 18}, { -8, 24}, { -4, 24}, {-38, 2}, {-29, -21},
        { -9, 14}, { 24, 21}, { 2, 21}, { -16, 26}, { -20, 38}, { 6, 27}, { 22, 21}, { -22, 1},
        { -17, 11}, { -20, 20}, { -12, 23}, { -27, 36}, { -30, 48}, { -25, 48}, { -14, 22}, { -36, 12},
        { -49, 10}, { -1, 23}, { -27, 22}, { -39, 44}, { -46, 52}, { -44, 42}, { -33, 27}, { -51, 0},
        { -14, -6}, { -14, 21}, { -22, 15}, { -46, 32}, { -44, 45}, { -30, 36}, { -15, 15}, { -27, -18},
        { 1, -16}, { 7, 7}, { -8, 14}, { -64, 23}, { -43, 30}, { -16, 32}, { 9, 12}, { 8, -16},
        { -17, -26}, { -9, -15}, { 22, -2}, { 17, 6}, { 29, 15}, { 12, 16}, { -6, 2}, { -17, -35}
    };

    // Material Values (MG, EG)
    const int mat_vals[6][2] = {
        { 82, 94 },   // Pawn
        { 337, 281 }, // Knight
        { 365, 297 }, // Bishop
        { 477, 512 }, // Rook
        { 1025, 936 },// Queen
        { 0, 0 }      // King (Invaluable)
    };

    // Game Phase Weights (to calculate taper)
    // P=0, N=1, B=1, R=2, Q=4
    const int phase_weights[6] = { 0, 1, 1, 2, 4, 0 };

    // Helper: Reconstruct board for check detection
    BoardState reconstruct_board(const uint64_t* pieces, const uint64_t* occupancy, uint32_t moveCount) {
        BoardState board;
        for(int i=0; i<12; ++i) board.pieces[i] = pieces[i];
        for(int i=0; i<3; ++i) board.occupancy[i] = occupancy[i];
        board.to_move = (moveCount % 2 == 0) ? Colour::White : Colour::Black;
        return board;
    }

    int32_t evaluate(const uint64_t* pieces, const uint64_t* occupancy, uint32_t sideToMove) {
        int mg[2] = {0, 0}; // [0]=White, [1]=Black
        int eg[2] = {0, 0};
        int game_phase = 0;

        // Loop through all piece types (Pawn(0) to King(5))
        for (int p = 0; p < 6; ++p) {
            // WHITE PIECES
            uint64_t w = pieces[p];
            while (w) {
                int sq = static_cast<int>(BitUtil::pop_lsb(w));
                
                mg[0] += mat_vals[p][0];
                eg[0] += mat_vals[p][1];
                
                const int (*table)[2] = nullptr;
                switch(p) {
                    case 0: table = val_pawn; break;
                    case 1: table = val_knight; break;
                    case 2: table = val_bishop; break;
                    case 3: table = val_rook; break;
                    case 4: table = val_queen; break;
                    case 5: table = val_king; break;
                }
                
                mg[0] += table[sq][0];
                eg[0] += table[sq][1];
                game_phase += phase_weights[p];
            }

            // BLACK PIECES
            uint64_t b = pieces[p + 6];
            while (b) {
                int sq = static_cast<int>(BitUtil::pop_lsb(b));
                
                mg[1] += mat_vals[p][0];
                eg[1] += mat_vals[p][1];

                const int (*table)[2] = nullptr;
                switch(p) {
                    case 0: table = val_pawn; break;
                    case 1: table = val_knight; break;
                    case 2: table = val_bishop; break;
                    case 3: table = val_rook; break;
                    case 4: table = val_queen; break;
                    case 5: table = val_king; break;
                }

                // Mirror for Black
                int mirrored_sq = sq ^ 56; 

                mg[1] += table[mirrored_sq][0];
                eg[1] += table[mirrored_sq][1];
                game_phase += phase_weights[p];
            }
        }

        // --- TAPERING ---
        if (game_phase > 24) game_phase = 24;
        
        int mg_score = mg[0] - mg[1];
        int eg_score = eg[0] - eg[1];
        
        int32_t final_score = (mg_score * game_phase + eg_score * (24 - game_phase)) / 24;

        // IMPORTANT: Return score relative to the side to move?
        // Standard engine convention: Always return relative to "Side to move".
        // BUT your Search.cpp logic (NegaMax) might expect absolute "White - Black".
        // Let's look at your Search.cpp:
        // You use: int32_t score = -alpha_beta(...);
        // This implies NegaMax. NegaMax requires the evaluation to be "Score from MY perspective".
        
        // If sideToMove == 0 (White), return (White - Black).
        // If sideToMove == 1 (Black), return (Black - White) -> which is -(White - Black).
        
        if (sideToMove == 1) {
            return -final_score;
        }
        
        return final_score;
    }
}
