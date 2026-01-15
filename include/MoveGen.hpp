#pragma once
#include "BoardState.hpp"
#include <vector>

namespace MoveGen {
    void generate_moves(const BoardState& board, std::vector<Move>& move_list);
    
    void generate_captures(const BoardState& board, std::vector<Move>& move_list);
}
