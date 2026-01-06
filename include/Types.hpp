#pragma once

#include <cstdint>


using Bitboard = std::uint64_t;

enum class Colour : uint8_t { White, Black, None };
enum class PieceType : uint8_t { Pawn, Knight, Bishop, Rook, Queen, King, None };

enum class Square : int {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2 = 8, B2, C2, D2, E2, F2, G2, H2,
    A3 = 16, B3, C3, D3, E3, F3, G3, H3,
    A4 = 24, B4, C4, D4, E4, F4, G4, H4,
    A5 = 32, B5, C5, D5, E5, F5, G5, H5,
    A6 = 40, B6, C6, D6, E6, F6, G6, H6,
    A7 = 48, B7, C7, D7, E7, F7, G7, H7,
    A8 = 56, B8, C8, D8, E8, F8, G8, H8,
    None = 64
};

enum class MoveFlag : uint16_t {
    Quiet = 0b0000,
    DoublePawnPush = 0b0001,
    KingCastle = 0b0010,
    QueenCastle = 0b0011,
    Capture = 0b0100,
    EnPassant = 0b0101,
    Promotion = 0b1000,
    KnightPromo = 0b1000,
    BishopPromo = 0b1001,
    RookPromo = 0b1010,
    QueenPromo = 0b1011,
    KnightPromoCapture = 0b1100,
    BishopPromoCapture = 0b1101,
    RookPromoCapture = 0b1110,
    QueenPromoCapture = 0b1111
};

struct Move {
    uint16_t data;

    static constexpr uint16_t FROM_MASK{0x3F}; // Bits 0-5: From square
    static constexpr uint16_t TO_MASK{0xFC0}; // Bits 6-11: To sqaure
    static constexpr uint16_t FLAG_MASK{0xF000}; // Bits 12-15: Special moves flag
    
    constexpr Move() : data(0) {}

    constexpr Move(Square from, Square to, MoveFlag flags = MoveFlag::Quiet)
        : data(static_cast<uint16_t>(
            static_cast<uint16_t>(from) |
            (static_cast<uint16_t>(to) << 6) |
            (static_cast<uint16_t>(flags) << 12))) {}

    [[nodiscard]] constexpr Square from() const {
        return static_cast<Square>(data & FROM_MASK);
    }

    [[nodiscard]] constexpr Square to() const {
        return static_cast<Square>((data & TO_MASK) >> 6);
    }

    [[nodiscard]] constexpr MoveFlag flags() const {
        return static_cast<MoveFlag>((data & FLAG_MASK) >> 12);
    }

    [[nodiscard]] constexpr bool is_capture() const {
        return static_cast<uint16_t>(flags()) 
            & static_cast<uint16_t>(MoveFlag::Capture);
    }

    [[nodiscard]] constexpr bool is_promotion() const {
        return static_cast<uint16_t>(flags())
            & static_cast<uint16_t>(MoveFlag::Promotion);
    }

    bool operator==(const Move& other) const = default;
};
