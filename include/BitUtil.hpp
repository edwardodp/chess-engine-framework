#pragma once

#include "Types.hpp"

#include <bit>


namespace BitUtil {
    constexpr Bitboard from_square(Square sq) {
        return 1ULL << static_cast<int>(sq);
    }

    constexpr void set_bit(Bitboard& bb, Square sq) { bb |= from_square(sq); }
    constexpr void clear_bit(Bitboard& bb, Square sq) { bb &= ~from_square(sq); }
    constexpr bool get_bit(Bitboard bb, Square sq) { return (bb & from_square(sq)) != 0; }

    constexpr Square pop_lsb(Bitboard& bb) {
        int index{std::countr_zero(bb)};
        bb &= bb - 1;
        return static_cast<Square>(index);
    }

    constexpr int count_bits(Bitboard bb) {
        return std::popcount(bb);
    }
}
