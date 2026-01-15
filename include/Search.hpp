#pragma once

#include "BoardState.hpp"

#include <functional>


namespace Search {

    // The callback type (matches your Main.cpp definition)
    using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

    struct SearchParams {
        int depth;
        EvalCallback evalFunc;
    };

    // Returns the best move found
    Move iterative_deepening(BoardState& board, const SearchParams& params);

}
