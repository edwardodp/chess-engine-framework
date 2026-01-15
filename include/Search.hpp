#pragma once

#include "BoardState.hpp"

namespace Search {

    using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

    struct SearchParams {
        int depth;
        EvalCallback evalFunc;
    };

    Square find_king(const BoardState& board, Colour side);

    Move iterative_deepening(BoardState& board, const SearchParams& params);
}
