#include <gtest/gtest.h>
#include "BoardState.hpp"
#include "Search.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"
#include "MoveGen.hpp"

// --- TEST HELPERS ---

int32_t testEval(const uint64_t* pieces, const uint64_t* occupancy, uint32_t moveCount) {
    int32_t score = 0;
    int values[] = {100, 300, 320, 500, 900, 20000}; 
    for (int i = 0; i < 6; ++i) {
        score += BitUtil::count_bits(pieces[i]) * values[i];      
        score -= BitUtil::count_bits(pieces[i + 6]) * values[i];  
    }
    return score;
}

void clear_board(BoardState& b) {
    b.pieces.fill(0);
    b.occupancy.fill(0);
    b.castle_rights = 0;
    b.en_passant_sq = Square::None;
    b.half_move_clock = 0;
    b.full_move_number = 1;
}

void add_piece(BoardState& b, Square sq, int piece_type, Colour color) {
    int offset = (color == Colour::White) ? 0 : 6;
    int idx = offset + piece_type;
    BitUtil::set_bit(b.pieces[idx], sq);
    int occ_idx = (color == Colour::White) ? 0 : 1;
    BitUtil::set_bit(b.occupancy[occ_idx], sq);
    BitUtil::set_bit(b.occupancy[2], sq);
}

class Environment : public ::testing::Environment {
public:
    void SetUp() override {
        Attacks::init();
    }
};

// --- TEST CASES ---

TEST(MoveGenTest, SimplePawnPush) {
    BoardState board;
    clear_board(board);
    add_piece(board, Square::E2, 0, Colour::White); // Pawn
    add_piece(board, Square::E1, 5, Colour::White); // King
    add_piece(board, Square::E8, 5, Colour::Black); // King
    board.to_move = Colour::White;

    std::vector<Move> moves;
    MoveGen::generate_moves(board, moves);

    bool found_push = false;
    for(auto m : moves) {
        if(m.from() == Square::E2 && m.to() == Square::E4) found_push = true;
    }
    EXPECT_TRUE(found_push) << "Engine failed to generate Double Pawn Push e2->e4";
}

TEST(SearchTest, FindsMateInOne) {
    BoardState board;
    clear_board(board);

    add_piece(board, Square::H1, 3, Colour::White); // Rook
    add_piece(board, Square::A1, 5, Colour::White); // King

    add_piece(board, Square::A8, 5, Colour::Black); // King
    add_piece(board, Square::A7, 0, Colour::Black); // Pawn (Blocker)
    add_piece(board, Square::B7, 0, Colour::Black); // Pawn (Blocker)
    add_piece(board, Square::C7, 0, Colour::Black); // Pawn (Blocker)
    add_piece(board, Square::A6, 0, Colour::Black); // Pawn (Blocker)
    add_piece(board, Square::B6, 0, Colour::Black); // Pawn (Blocker)
    add_piece(board, Square::C6, 0, Colour::Black); // Pawn (Blocker)

    board.to_move = Colour::White;
    
    Search::SearchParams params;
    params.depth = 3; 
    params.evalFunc = testEval;

    Move best = Search::iterative_deepening(board, params);

    EXPECT_EQ(best.from(), Square::H1);
    EXPECT_EQ(best.to(), Square::H8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new Environment);
    return RUN_ALL_TESTS();
}
