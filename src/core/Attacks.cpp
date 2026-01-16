#include "Attacks.hpp"
#include "BitUtil.hpp"
#include <vector>
#include <cassert>

namespace Attacks {

std::array<std::array<uint64_t, 64>, 2> PawnAttacks;
std::array<uint64_t, 64> KnightAttacks;
std::array<uint64_t, 64> KingAttacks;

std::array<Magic, 64> RookMagics;
std::array<Magic, 64> BishopMagics;
std::array<Bitboard, 102400> RookTable;
std::array<Bitboard, 5248>   BishopTable;

namespace {
    // Improved pseudo-RNG (Xorshift)
    uint64_t random_state = 1804289383;

    uint64_t get_random_u64() {
        uint64_t x = random_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        random_state = x;
        return x;
    }

    uint64_t get_random_u64_sparse() {
        return get_random_u64() & get_random_u64() & get_random_u64();
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

        while (true) {
            Bitboard magic_candidate = get_random_u64_sparse();
            if (BitUtil::count_bits((mask * magic_candidate) & 0xFF00000000000000ULL) < 6) continue;

            int shift = 64 - bits;
            bool collision = false;
            
            std::vector<int> used(permutations, -1);
            
            for (int i = 0; i < permutations; ++i) {
                size_t idx = ((occupancies[i] & mask) * magic_candidate) >> shift;
                
                // If this index was visited by a different attack set -> collision
                if (used[idx] != -1 && used[idx] != i && attacks[used[idx]] != attacks[i]) {
                    collision = true; 
                    break;
                }
                used[idx] = i;
            }

            if (!collision) {
                for (int i = 0; i < permutations; ++i) {
                    size_t idx = ((occupancies[i] & mask) * magic_candidate) >> shift;
                    table_start[current_offset + idx] = attacks[i];
                }

                magics[sq] = { mask, magic_candidate, current_offset, (uint32_t)shift };
                current_offset += permutations;
                break;
            }
        }
    }
}

void init() {
    static bool initialized = false;
    if (initialized) return;

    find_magics(true, RookMagics, RookTable.data());
    find_magics(false, BishopMagics, BishopTable.data());

    // Init Leapers (Pawns, Knights, Kings)
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard p = (1ULL << sq);
        // White Pawns
        if ((p << 7) & 0x7F7F7F7F7F7F7F7FULL) PawnAttacks[0][sq] |= (p << 7);
        if ((p << 9) & 0xFEFEFEFEFEFEFEFEULL) PawnAttacks[0][sq] |= (p << 9);
        // Black Pawns
        if ((p >> 7) & 0xFEFEFEFEFEFEFEFEULL) PawnAttacks[1][sq] |= (p >> 7);
        if ((p >> 9) & 0x7F7F7F7F7F7F7F7FULL) PawnAttacks[1][sq] |= (p >> 9);

        // Knights
        Bitboard k = (1ULL << sq);
        if ((k << 17) & 0xFEFEFEFEFEFEFEFEULL) KnightAttacks[sq] |= (k << 17);
        if ((k << 15) & 0x7F7F7F7F7F7F7F7FULL) KnightAttacks[sq] |= (k << 15);
        if ((k << 10) & 0xFCFCFCFCFCFCFCFCULL) KnightAttacks[sq] |= (k << 10);
        if ((k << 6)  & 0x3F3F3F3F3F3F3F3FULL) KnightAttacks[sq] |= (k << 6);
        if ((k >> 17) & 0x7F7F7F7F7F7F7F7FULL) KnightAttacks[sq] |= (k >> 17);
        if ((k >> 15) & 0xFEFEFEFEFEFEFEFEULL) KnightAttacks[sq] |= (k >> 15);
        if ((k >> 10) & 0x3F3F3F3F3F3F3F3FULL) KnightAttacks[sq] |= (k >> 10);
        if ((k >> 6)  & 0xFCFCFCFCFCFCFCFCULL) KnightAttacks[sq] |= (k >> 6);

        // Kings
        Bitboard kb = (1ULL << sq);
        Bitboard attacks = 0;
        if (kb << 8) attacks |= (kb << 8);
        if (kb >> 8) attacks |= (kb >> 8);
        if ((kb << 1) & 0xFEFEFEFEFEFEFEFEULL) attacks |= (kb << 1);
        if ((kb >> 1) & 0x7F7F7F7F7F7F7F7FULL) attacks |= (kb >> 1);
        if ((kb << 9) & 0xFEFEFEFEFEFEFEFEULL) attacks |= (kb << 9);
        if ((kb >> 9) & 0x7F7F7F7F7F7F7F7FULL) attacks |= (kb >> 9);
        if ((kb << 7) & 0x7F7F7F7F7F7F7F7FULL) attacks |= (kb << 7);
        if ((kb >> 7) & 0xFEFEFEFEFEFEFEFEULL) attacks |= (kb >> 7);
        KingAttacks[sq] = attacks;
    }

    initialized = true;
}

uint64_t get_rook_attacks(int sq, uint64_t occ) {
    const auto& m = RookMagics[sq];
    occ &= m.mask;
    occ *= m.magic;
    occ >>= m.shift;
    return RookTable[m.offset + occ];
}

uint64_t get_bishop_attacks(int sq, uint64_t occ) {
    const auto& m = BishopMagics[sq];
    occ &= m.mask;
    occ *= m.magic;
    occ >>= m.shift;
    return BishopTable[m.offset + occ];
}

bool is_square_attacked(Square sq, Colour attacker, const Bitboard pieces[], Bitboard all_occ) {
    int s = static_cast<int>(sq);
    int us = static_cast<int>(attacker);
    int them = us ^ 1; 

    if (PawnAttacks[them][s] & pieces[us * 6]) return true;
    if (KnightAttacks[s] & pieces[us * 6 + 1]) return true;
    if (KingAttacks[s]   & pieces[us * 6 + 5]) return true;

    Bitboard bishops = pieces[us * 6 + 2] | pieces[us * 6 + 4]; 
    if (bishops && (get_bishop_attacks(s, all_occ) & bishops)) return true;

    Bitboard rooks = pieces[us * 6 + 3] | pieces[us * 6 + 4];
    if (rooks && (get_rook_attacks(s, all_occ) & rooks)) return true;

    return false;
}

}
