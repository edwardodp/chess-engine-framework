#include "Zobrist.hpp"
#include <random>

namespace Zobrist {
    std::array<std::array<uint64_t, 64>, 12> piece_keys;
    std::array<uint64_t, 65> en_passant_keys;
    std::array<uint64_t, 16> castle_keys;
    uint64_t side_key;

    void init() {
        std::mt19937_64 rng(123456789ULL);
        std::uniform_int_distribution<uint64_t> dist;

        for (int p = 0; p < 12; ++p) {
            for (int sq = 0; sq < 64; ++sq) {
                piece_keys[p][sq] = dist(rng);
            }
        }

        for (int sq = 0; sq < 65; ++sq) {
            en_passant_keys[sq] = dist(rng);
        }

        for (int i = 0; i < 16; ++i) {
            castle_keys[i] = dist(rng);
        }

        side_key = dist(rng);
    }
}
