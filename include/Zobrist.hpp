#pragma once
#include "Types.hpp"
#include <array>

namespace Zobrist {
    // [Piece Type (0-11)][Square (0-63)]
    extern std::array<std::array<uint64_t, 64>, 12> piece_keys;
    
    // [Square (0-63)] (File is usually enough, but square is safer/easier)
    extern std::array<uint64_t, 65> en_passant_keys;
    
    // [Castle Rights (0-15)]
    extern std::array<uint64_t, 16> castle_keys;
    
    extern uint64_t side_key;

    void init();
}
