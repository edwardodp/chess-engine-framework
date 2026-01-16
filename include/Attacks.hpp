#pragma once

#include "Types.hpp"
#include <array>
#include <cstdint>

namespace Attacks {

    // Precomputed Attack Tables
    extern std::array<std::array<uint64_t, 64>, 2> PawnAttacks;
    extern std::array<uint64_t, 64> KnightAttacks;
    extern std::array<uint64_t, 64> KingAttacks;

    // Magic Bitboards
    struct Magic {
        uint64_t mask;
        uint64_t magic;
        uint32_t offset;
        uint32_t shift;
    };

    void init();

    // Magic Lookups
    uint64_t get_rook_attacks(int sq, uint64_t occ);
    uint64_t get_bishop_attacks(int sq, uint64_t occ);
    inline uint64_t get_queen_attacks(int sq, uint64_t occ) {
        return get_rook_attacks(sq, occ) | get_bishop_attacks(sq, occ);
    }

    bool is_square_attacked(Square sq, Colour attacker, const uint64_t pieces[], uint64_t all_occ);
}
