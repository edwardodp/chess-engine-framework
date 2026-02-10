#pragma once
#include "Search.hpp"
#include "Types.hpp"
#include <string>

namespace GUI {
    void Launch(Search::EvalCallback evalFunc, int depth, int human_side_int, std::string start_fen);
}
