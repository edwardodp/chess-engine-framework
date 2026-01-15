#pragma once
#include <cstdint>

namespace Evaluation {
    int32_t evaluate(const uint64_t* pieces, const uint64_t* occupancy, uint32_t moveCount);
}
