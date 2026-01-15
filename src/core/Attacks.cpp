#include "Attacks.hpp"
#include "BitUtil.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cstring>


namespace Attacks {

// 1. DEFINE THE GLOBALS (Declared extern in header)
std::array<Magic, 64> RookMagics;
std::array<Magic, 64> BishopMagics;
std::array<Bitboard, ROOK_TABLE_SIZE> RookTable;
std::array<Bitboard, BISHOP_TABLE_SIZE> BishopTable;

// 2. INTERNAL HELPERS (Used only for initialization)
namespace {
    // Random Number Generator
    uint64_t random_u64() {
        static uint64_t seed = 1070372;
        seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
        return seed * 2685821657736338717ULL;
    }

    uint64_t random_u64_sparse() {
        return random_u64() & random_u64() & random_u64();
    }

    Bitboard gen_rook_mask(int sq) {
        Bitboard rook = 0;
        int r = sq / 8, f = sq % 8;
        for (int tr = r + 1; tr < 7; ++tr) rook |= (1ULL << (tr * 8 + f));
        for (int tr = r - 1; tr > 0; --tr) rook |= (1ULL << (tr * 8 + f));
        for (int tf = f + 1; tf < 7; ++tf) rook |= (1ULL << (r * 8 + tf));
        for (int tf = f - 1; tf > 0; --tf) rook |= (1ULL << (r * 8 + tf));
        return rook;
    }

    Bitboard gen_bishop_mask(int sq) {
        Bitboard bishop = 0;
        int r = sq / 8, f = sq % 8;
        for (int tr = r + 1, tf = f + 1; tr < 7 && tf < 7; ++tr, ++tf) bishop |= (1ULL << (tr * 8 + tf));
        for (int tr = r + 1, tf = f - 1; tr < 7 && tf > 0; ++tr, --tf) bishop |= (1ULL << (tr * 8 + tf));
        for (int tr = r - 1, tf = f + 1; tr > 0 && tf < 7; --tr, ++tf) bishop |= (1ULL << (tr * 8 + tf));
        for (int tr = r - 1, tf = f - 1; tr > 0 && tf > 0; --tr, --tf) bishop |= (1ULL << (tr * 8 + tf));
        return bishop;
    }

    Bitboard slow_rook_attacks(int sq, Bitboard occ) {
        Bitboard attacks = 0;
        int r = sq / 8, f = sq % 8;
        for (int tr = r + 1; tr <= 7; ++tr) { attacks |= (1ULL << (tr * 8 + f)); if (occ & (1ULL << (tr * 8 + f))) break; }
        for (int tr = r - 1; tr >= 0; --tr) { attacks |= (1ULL << (tr * 8 + f)); if (occ & (1ULL << (tr * 8 + f))) break; }
        for (int tf = f + 1; tf <= 7; ++tf) { attacks |= (1ULL << (r * 8 + tf)); if (occ & (1ULL << (r * 8 + tf))) break; }
        for (int tf = f - 1; tf >= 0; --tf) { attacks |= (1ULL << (r * 8 + tf)); if (occ & (1ULL << (r * 8 + tf))) break; }
        return attacks;
    }

    Bitboard slow_bishop_attacks(int sq, Bitboard occ) {
        Bitboard attacks = 0;
        int r = sq / 8, f = sq % 8;
        for (int tr = r + 1, tf = f + 1; tr <= 7 && tf <= 7; ++tr, ++tf) { attacks |= (1ULL << (tr * 8 + tf)); if (occ & (1ULL << (tr * 8 + tf))) break; }
        for (int tr = r + 1, tf = f - 1; tr <= 7 && tf >= 0; ++tr, --tf) { attacks |= (1ULL << (tr * 8 + tf)); if (occ & (1ULL << (tr * 8 + tf))) break; }
        for (int tr = r - 1, tf = f + 1; tr >= 0 && tf <= 7; --tr, ++tf) { attacks |= (1ULL << (tr * 8 + tf)); if (occ & (1ULL << (tr * 8 + tf))) break; }
        for (int tr = r - 1, tf = f - 1; tr >= 0 && tf >= 0; --tr, --tf) { attacks |= (1ULL << (tr * 8 + tf)); if (occ & (1ULL << (tr * 8 + tf))) break; }
        return attacks;
    }

    Bitboard set_occupancy(int index, int bits_in_mask, Bitboard mask) {
        Bitboard occ = 0ULL;
        for (int i = 0; i < bits_in_mask; ++i) {
            int square = BitUtil::lsb(mask);
            mask &= mask - 1; 
            if (index & (1 << i)) occ |= (1ULL << square);
        }
        return occ;
    }
}

// 3. THE MAGIC MINER (Finds non-colliding magic numbers)
void find_magics(bool rook, std::array<Magic, 64>& magics, uint64_t* table_start) {
    uint32_t current_offset = 0;

    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = rook ? gen_rook_mask(sq) : gen_bishop_mask(sq);
        int bits = BitUtil::count_bits(mask);
        int permutations = 1 << bits;

        std::vector<Bitboard> occupancies(permutations);
        std::vector<Bitboard> attacks(permutations);

        for (int i = 0; i < permutations; ++i) {
            occupancies[i] = set_occupancy(i, bits, mask);
            attacks[i] = rook ? slow_rook_attacks(sq, occupancies[i]) 
                              : slow_bishop_attacks(sq, occupancies[i]);
        }

        // Try random numbers until we find one that maps all permutations uniquely
        for (int k = 0; k < 10000000; ++k) {
            Bitboard magic_candidate = random_u64_sparse();
            int shift = 64 - bits;
            bool collision = false;
            
            // Clear the relevant part of the table
            // Note: We access the table via the raw pointer passed in
            for(int i=0; i<permutations; ++i) {
               // We don't strictly need to clear if we check carefully, 
               // but for safety in this simple implementation:
               size_t idx = ((occupancies[i] & mask) * magic_candidate) >> shift;
               table_start[current_offset + idx] = 0ULL;
            }

            for (int i = 0; i < permutations; ++i) {
                size_t idx = ((occupancies[i] & mask) * magic_candidate) >> shift;
                
                if (table_start[current_offset + idx] != 0ULL && table_start[current_offset + idx] != attacks[i]) {
                    collision = true; 
                    break; 
                }
                table_start[current_offset + idx] = attacks[i];
            }

            if (!collision) {
                magics[sq] = { mask, magic_candidate, current_offset, shift };
                current_offset += permutations;
                break;
            }
        }
    }
}

// 4. PUBLIC INIT FUNCTION
void init() {
    static bool initialized = false;
    if (initialized) return;

    std::cout << "Initializing Magic Bitboards (Mining)..." << std::endl;
    find_magics(true, RookMagics, RookTable.data());
    find_magics(false, BishopMagics, BishopTable.data());
    std::cout << "Magics Initialized." << std::endl;
    
    initialized = true;
}

// 5. DEBUG PRINT
void print_bitboard(Bitboard b) {
    std::string board = "+---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            board += "| ";
            board += (b & (1ULL << (r * 8 + f))) ? "X " : ". ";
        }
        board += "| " + std::to_string(r + 1) + "\n+---+---+---+---+---+---+---+---+\n";
    }
    board += "  a   b   c   d   e   f   g   h\n";
    std::cout << board << std::endl;
}

bool is_square_attacked(Square sq, Colour attacker, const Bitboard pieces[], Bitboard all_occ) {
    // 1. Check Pawns
    // If we are checking if White attacks 'sq', we pretend there is a Black pawn at 'sq' 
    // and see if it captures any White pawns.
    // Logic: PawnAttacks[0] = White Attacks, PawnAttacks[1] = Black Attacks.
    // If attacker is White (0), we use Black's attack table (1) to look "backwards".
    int attacker_side = static_cast<int>(attacker);
    int our_side = attacker_side ^ 1; // 0->1, 1->0
    
    // Index of attacker's pawns in the pieces array
    // White Pawns = 0, Black Pawns = 6. 
    int pawn_idx = attacker_side * 6; 
    
    if (PawnAttacks[our_side][static_cast<int>(sq)] & pieces[pawn_idx]) return true;

    // 2. Check Knights
    int knight_idx = attacker_side * 6 + 1; // N=1
    if (KnightAttacks[static_cast<int>(sq)] & pieces[knight_idx]) return true;

    // 3. Check Kings
    int king_idx = attacker_side * 6 + 5; // K=5
    if (KingAttacks[static_cast<int>(sq)] & pieces[king_idx]) return true;

    // 4. Check Bishops/Queens (Diagonal)
    int bishop_idx = attacker_side * 6 + 2; // B=2
    int queen_idx  = attacker_side * 6 + 4; // Q=4
    Bitboard diag_attackers = pieces[bishop_idx] | pieces[queen_idx];
    
    if (diag_attackers) {
        // Only generate slider attacks if there are actual attackers to check against
        if (get_bishop_attacks(static_cast<int>(sq), all_occ) & diag_attackers) return true;
    }

    // 5. Check Rooks/Queens (Straight)
    int rook_idx = attacker_side * 6 + 3; // R=3
    Bitboard straight_attackers = pieces[rook_idx] | pieces[queen_idx];
    
    if (straight_attackers) {
        if (get_rook_attacks(static_cast<int>(sq), all_occ) & straight_attackers) return true;
    }

    return false;
}

} // namespace Attacks
