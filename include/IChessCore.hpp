#pragma once

#include "Types.hpp"
#include "BoardState.hpp"

#include <vector>
#include <string_view>


class IChessCore {
public:
    virtual ~IChessCore() = default;
    virtual void make_move(Move move) = 0;
    virtual void unmake_move() = 0;
    virtual std::vector<Move> legal_moves() = 0;
    virtual GameState get_game_state() const = 0;
    virtual bool is_check(Colour side) const = 0;
    virtual const BoardState& get_board_state() const = 0;
    virtual void reset() = 0;
    virtual bool fen(std::string_view fen) = 0;
};
