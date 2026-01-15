#pragma once

#include <cstdint>
#include <array>

using Bitboard = uint64_t;

enum class Colour { White, Black };

enum class Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    None
};

enum class MoveFlag : uint8_t {
    Quiet              = 0b0000,
    DoublePawnPush     = 0b0001,
    KingCastle         = 0b0010,
    QueenCastle        = 0b0011,
    Capture            = 0b0100,
    EnPassant          = 0b0101,
    
    // Promotions (Quiet)
    KnightPromotion    = 0b1000,
    BishopPromotion    = 0b1001,
    RookPromotion      = 0b1010,
    QueenPromotion     = 0b1011,
    
    // Promotions (Capture)
    KnightPromoCapture = 0b1100, 
    BishopPromoCapture = 0b1101,
    RookPromoCapture   = 0b1110,
    QueenPromoCapture  = 0b1111 
};

class Move {
public:
    Move() : data(0) {}
    Move(Square from, Square to, MoveFlag flag) {
        data = static_cast<uint16_t>(from) | 
              (static_cast<uint16_t>(to) << 6) | 
              (static_cast<uint16_t>(flag) << 12);
    }

    Square from() const { return static_cast<Square>(data & 0x3F); }
    Square to() const { return static_cast<Square>((data >> 6) & 0x3F); }
    MoveFlag flag() const { return static_cast<MoveFlag>((data >> 12) & 0xF); }
    uint16_t raw() const { return data; }

    bool is_capture() const {
        int f = static_cast<int>(flag());
        return (f & 0b0100); 
    }

    bool is_promotion() const {
        int f = static_cast<int>(flag());
        return (f & 0b1000);
    }

    bool is_promo_knight() const {
        int f = static_cast<int>(flag());
        return (f & 0b1011) == 0b1000;
    }

    bool is_promo_bishop() const {
        int f = static_cast<int>(flag());
        return (f & 0b1011) == 0b1001;
    }

    bool is_promo_rook() const {
        int f = static_cast<int>(flag());
        return (f & 0b1011) == 0b1010;
    }

    bool is_promo_queen() const {
        int f = static_cast<int>(flag());
        return (f & 0b1011) == 0b1011; 
    }

private:
    uint16_t data;
};
