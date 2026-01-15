#pragma once

#include "BoardState.hpp"
#include <cstdint>

namespace Search {

    using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

    struct SearchParams {
        int depth;
        EvalCallback evalFunc;
    };

    struct SearchStats {
        int depth_reached = 0;
        int32_t score = 0;
        int best_move_raw = 0;
    };

    Square find_king(const BoardState& board, Colour side);

    Move iterative_deepening(BoardState& board, const SearchParams& params, SearchStats& stats);
}
