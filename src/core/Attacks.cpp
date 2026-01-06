#include "Attacks.hpp"
#include "Types.hpp"
#include "BitUtil.hpp"

#include <array>


namespace Attacks {

std::array<Bitboard, 64> KnightAttacks{};
std::array<Bitboard, 64> KingAttacks{};
std::array<std::array<Bitboard, 64>, 2> PawnAttacks{};

static constexpr Bitboard FILE_A{0x0101010101010101ULL};
static constexpr Bitboard FILE_B{FILE_A << 1};
static constexpr Bitboard FILE_G{FILE_A << 6};
static constexpr Bitboard FILE_H{FILE_A << 7};

static constexpr Bitboard NOT_A{~FILE_A};
static constexpr Bitboard NOT_AB{~(FILE_A | FILE_B)};
static constexpr Bitboard NOT_H{~FILE_H};
static constexpr Bitboard NOT_GH{~(FILE_G | FILE_H)};

constexpr void init() {
    for (int sq{0}; sq < 64; ++sq) {
        Bitboard b{1ULL << sq};

        // Knight Attacks
        Bitboard knight{0};
        
        knight |= (b << 17) & NOT_A; // Up 2, Right 1
        knight |= (b << 15) & NOT_H; // Up 2, Left 1
        knight |= (b >> 15) & NOT_A; // Down 2, Right 1
        knight |= (b >> 17) & NOT_H; // Down 2, Left 1
        knight |= (b << 10) & NOT_AB; // Up 1, Right 2
        knight |= (b << 6)  & NOT_GH; // Up 1, Left 2
        knight |= (b >> 6)  & NOT_AB; // Down 1, Right 2
        knight |= (b >> 10) & NOT_GH; // Down 1, Left 2
        
        KnightAttacks[static_cast<size_t>(sq)] = knight;

        // King Attacks
        Bitboard king{0};
        king |= (b << 8); // Up
        king |= (b >> 8); // Down
        king |= (b << 1) & NOT_A; // Right
        king |= (b >> 1) & NOT_H; // Left
        king |= (b << 9) & NOT_A; // Up-Right
        king |= (b << 7) & NOT_H; // Up-Left
        king |= (b >> 7) & NOT_A; // Down-Right
        king |= (b >> 9) & NOT_H; // Down-Left
        
        KingAttacks[static_cast<size_t>(sq)] = king;

        // Pawn Attacks (captures)
        PawnAttacks[0][static_cast<size_t>(sq)] = ((b << 7) & NOT_H) | ((b << 9) & NOT_A); // White
        PawnAttacks[1][static_cast<size_t>(sq)] = ((b >> 7) & NOT_A) | ((b >> 9) & NOT_H); // Black
    }
}

}
