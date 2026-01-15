#pragma once

#include "Types.hpp"
#include "BitUtil.hpp"

#include <array>


namespace Attacks {

inline constexpr size_t ROOK_TABLE_SIZE = 102400;
inline constexpr size_t BISHOP_TABLE_SIZE = 5248;

struct Magic {
    Bitboard mask;
    Bitboard magic;
    uint32_t offset;
    int shift;

    // Removed constexpr so it works with runtime values
    size_t get_index(Bitboard occ) const {
        return ((occ & mask) * magic) >> shift;
    }
};

namespace detail {
    static constexpr Bitboard FILE_A{0x0101010101010101ULL};
    static constexpr Bitboard FILE_B{FILE_A << 1};
    static constexpr Bitboard FILE_G{FILE_A << 6};
    static constexpr Bitboard FILE_H{FILE_A << 7};

    static constexpr Bitboard NOT_A{~FILE_A};
    static constexpr Bitboard NOT_AB{~(FILE_A | FILE_B)};
    static constexpr Bitboard NOT_H{~FILE_H};
    static constexpr Bitboard NOT_GH{~(FILE_G | FILE_H)};

    constexpr Bitboard gen_knight_mask(int sq) {
        Bitboard b{1ULL << sq};
        Bitboard knight{0};
        knight |= (b << 17) & NOT_A;  knight |= (b << 15) & NOT_H;
        knight |= (b >> 15) & NOT_A;  knight |= (b >> 17) & NOT_H;
        knight |= (b << 10) & NOT_AB; knight |= (b << 6)  & NOT_GH;
        knight |= (b >> 6)  & NOT_AB; knight |= (b >> 10) & NOT_GH;
        return knight;
    }

    constexpr Bitboard gen_king_mask(int sq) {
        Bitboard b{1ULL << sq};
        Bitboard king{0};
        king |= (b << 8) | (b >> 8);
        king |= ((b << 1) | (b << 9) | (b >> 7)) & NOT_A;
        king |= ((b >> 1) | (b >> 9) | (b << 7)) & NOT_H;
        return king;
    }

    constexpr std::array<Bitboard, 64> init_knights() {
        std::array<Bitboard, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = gen_knight_mask(i);
        return arr;
    }

    constexpr std::array<Bitboard, 64> init_kings() {
        std::array<Bitboard, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = gen_king_mask(i);
        return arr;
    }

    constexpr std::array<std::array<Bitboard, 64>, 2> init_pawns() {
        std::array<std::array<Bitboard, 64>, 2> arr{};
        for (int i = 0; i < 64; ++i) {
            Bitboard b{1ULL << i};
            arr[0][i] = ((b << 7) & NOT_H) | ((b << 9) & NOT_A); // White
            arr[1][i] = ((b >> 7) & NOT_A) | ((b >> 9) & NOT_H); // Black
        }
        return arr;
    }
} // namespace detail


// Global Tables
// Simple jump tables can remain constexpr
inline constexpr std::array<Bitboard, 64> KnightAttacks = detail::init_knights();
inline constexpr std::array<Bitboard, 64> KingAttacks = detail::init_kings();
inline constexpr std::array<std::array<Bitboard, 64>, 2> PawnAttacks = detail::init_pawns(); 

// Magic Tables are now EXTERN (Definitions moved to Attacks.cpp)
// This allows the Mining code to write to them at runtime.
extern std::array<Magic, 64> RookMagics;
extern std::array<Magic, 64> BishopMagics;
extern std::array<Bitboard, ROOK_TABLE_SIZE> RookTable;
extern std::array<Bitboard, BISHOP_TABLE_SIZE> BishopTable;


// Public API

void print_bitboard(Bitboard b);

// Initialization function (Must be called in main!)
void init();

bool is_square_attacked(Square sq, Colour attacker, 
                        const Bitboard pieces[],
                        Bitboard all_occ);

inline Bitboard get_rook_attacks(int sq, Bitboard occ) {
    const Magic& m = RookMagics[sq];
    return RookTable[m.offset + m.get_index(occ)];
}

inline Bitboard get_bishop_attacks(int sq, Bitboard occ) {
    const Magic& m = BishopMagics[sq];
    return BishopTable[m.offset + m.get_index(occ)];
}

inline Bitboard get_queen_attacks(int sq, Bitboard occ) {
    return get_rook_attacks(sq, occ) | get_bishop_attacks(sq, occ);
}

}
