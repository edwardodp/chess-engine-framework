#pragma once

#include "Types.hpp"

#include <array>
#include <vector>


struct UndoInfo {
    uint8_t castle_rights{0}; // Bitfield: 1=WhiteKing, 2=WhiteQueen, 4=BlackKing, 8=BlackQueen
    Square en_passant_sq{Square::None};
    PieceType captured_piece{PieceType::None};
    int halfmove_clock{0};
    uint64_t pos_hash{0}; // For 3-fold-repetition detection
};

struct BoardState {
    // Bitboards (Pieces): White(P,N,B,R,Q,K),Black(P,N,B,R,Q,K)
    std::array<Bitboard, 12> pieces{};
    // Bitboards (Occupancy): White,Black,All
    std::array<Bitboard, 3> occupancy{};

    Colour to_move{Colour::White};
    Square en_passant_sq{Square::None};
    uint8_t castle_rights{0};
    int halfmove_clock{0};
    int full_move_number{1};

    std::vector<UndoInfo> history;

    static constexpr size_t get_piece_index(Colour c, PieceType p) {
        return static_cast<size_t>(c) * 6 + static_cast<size_t>(p);
    }
};

enum class GameState { Ongoing, WhiteWin, BlackWin, Draw };
