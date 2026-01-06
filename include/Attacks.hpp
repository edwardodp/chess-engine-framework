#pragma once

#include "Types.hpp"

#include <array>


namespace Attacks {
    extern std::array<std::array<Bitboard, 64>, 2> PawnAttacks;
    extern std::array<Bitboard, 64> KnightAttacks;
    extern std::array<Bitboard, 64> KingAttacks;
    
    constexpr void init();
}
