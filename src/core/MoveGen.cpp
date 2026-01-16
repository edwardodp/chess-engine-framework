#include "MoveGen.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"

namespace MoveGen {

namespace {
    // Helper for standard moves
    void serialize_moves(Square from, Bitboard targets, std::vector<Move>& list, MoveFlag flag) {
        while (targets) {
            Square to = BitUtil::pop_lsb(targets);
            list.emplace_back(from, to, flag);
        }
    }

    // Helper to add ALL 4 promotion types
    void add_promotions(Square from, Square to, std::vector<Move>& list, bool is_capture) {
        if (is_capture) {
            list.emplace_back(from, to, MoveFlag::QueenPromoCapture);
            list.emplace_back(from, to, MoveFlag::RookPromoCapture);
            list.emplace_back(from, to, MoveFlag::BishopPromoCapture);
            list.emplace_back(from, to, MoveFlag::KnightPromoCapture);
        } else {
            list.emplace_back(from, to, MoveFlag::QueenPromotion);
            list.emplace_back(from, to, MoveFlag::RookPromotion);
            list.emplace_back(from, to, MoveFlag::BishopPromotion);
            list.emplace_back(from, to, MoveFlag::KnightPromotion);
        }
    }
}

void generate_moves(const BoardState& board, std::vector<Move>& move_list) {
    Colour us = board.to_move;
    Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
    Bitboard us_occ = board.occupancy[static_cast<int>(us)];
    Bitboard them_occ = board.occupancy[static_cast<int>(them)];
    Bitboard all_occ = board.occupancy[2];

    // --- Pawns ---
    Bitboard pawns = board.pieces[(us == Colour::White) ? 0 : 6];
    
    Bitboard promo_rank = (us == Colour::White) ? 0xFF00000000000000ULL : 0x00000000000000FFULL;

    Bitboard single_push = (us == Colour::White) ? (pawns << 8) : (pawns >> 8);
    single_push &= ~all_occ; // Must be empty

    Bitboard quiet_pushes = single_push & ~promo_rank;
    Bitboard promo_pushes = single_push & promo_rank;

    while (quiet_pushes) {
        Square to = BitUtil::pop_lsb(quiet_pushes);
        Square from = static_cast<Square>(static_cast<int>(to) + ((us == Colour::White) ? -8 : 8));
        move_list.emplace_back(from, to, MoveFlag::Quiet);
    }

    while (promo_pushes) {
        Square to = BitUtil::pop_lsb(promo_pushes);
        Square from = static_cast<Square>(static_cast<int>(to) + ((us == Colour::White) ? -8 : 8));
        add_promotions(from, to, move_list, false); 
    }

    // Can only happen if the single push land was valid AND we are on start rank
    Bitboard single_push_valid = single_push; 
    Bitboard double_push_target = (us == Colour::White) ? 0x00000000FF000000ULL : 0x000000FF00000000ULL; 
    
    Bitboard double_push = (us == Colour::White) ? (single_push << 8) : (single_push >> 8);
    
    // Mask to ensure we are on the correct rank
    Bitboard rank_4_5 = (us == Colour::White) ? 0x00000000FF000000ULL : 0x000000FF00000000ULL;
    double_push &= rank_4_5;
    double_push &= ~all_occ;

    while (double_push) {
        Square to = BitUtil::pop_lsb(double_push);
        Square from = static_cast<Square>(static_cast<int>(to) + ((us == Colour::White) ? -16 : 16));
        move_list.emplace_back(from, to, MoveFlag::DoublePawnPush);
    }

    // Captures
    Bitboard pawns_copy = pawns;
    while(pawns_copy) {
        Square from = BitUtil::pop_lsb(pawns_copy);
        Bitboard attacks = (us == Colour::White) ? Attacks::PawnAttacks[0][static_cast<int>(from)] 
                                                 : Attacks::PawnAttacks[1][static_cast<int>(from)];
        
        Bitboard valid_captures = attacks & them_occ;

        Bitboard normal_captures = valid_captures & ~promo_rank;
        Bitboard promo_captures  = valid_captures & promo_rank;

        serialize_moves(from, normal_captures, move_list, MoveFlag::Capture);

        while (promo_captures) {
            Square to = BitUtil::pop_lsb(promo_captures);
            add_promotions(from, to, move_list, true);
        }

        // En Passant
        if (board.en_passant_sq != Square::None) {
            if (attacks & (1ULL << static_cast<int>(board.en_passant_sq))) {
                move_list.emplace_back(from, board.en_passant_sq, MoveFlag::EnPassant);
            }
        }
    }

    // --- Knights ---
    Bitboard knights = board.pieces[(us == Colour::White) ? 1 : 7];
    while (knights) {
        Square from = BitUtil::pop_lsb(knights);
        Bitboard moves = Attacks::KnightAttacks[static_cast<int>(from)] & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
        serialize_moves(from, moves & ~them_occ, move_list, MoveFlag::Quiet);
    }

    // --- King ---
    Bitboard king = board.pieces[(us == Colour::White) ? 5 : 11];
    if (king) {
        Square from = static_cast<Square>(BitUtil::lsb(king));
        Bitboard moves = Attacks::KingAttacks[static_cast<int>(from)] & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
        serialize_moves(from, moves & ~them_occ, move_list, MoveFlag::Quiet);
    }

    // --- Sliders ---
    Bitboard bishops = board.pieces[(us == Colour::White) ? 2 : 8];
    while (bishops) {
        Square from = BitUtil::pop_lsb(bishops);
        Bitboard moves = Attacks::get_bishop_attacks(static_cast<int>(from), all_occ) & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
        serialize_moves(from, moves & ~them_occ, move_list, MoveFlag::Quiet);
    }

    Bitboard rooks = board.pieces[(us == Colour::White) ? 3 : 9];
    while (rooks) {
        Square from = BitUtil::pop_lsb(rooks);
        Bitboard moves = Attacks::get_rook_attacks(static_cast<int>(from), all_occ) & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
        serialize_moves(from, moves & ~them_occ, move_list, MoveFlag::Quiet);
    }

    Bitboard queens = board.pieces[(us == Colour::White) ? 4 : 10];
    while (queens) {
        Square from = BitUtil::pop_lsb(queens);
        Bitboard moves = Attacks::get_queen_attacks(static_cast<int>(from), all_occ) & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
        serialize_moves(from, moves & ~them_occ, move_list, MoveFlag::Quiet);
    }

    // --- Castling ---
    if (us == Colour::White) {
        if ((board.castle_rights & 1) && !BitUtil::get_bit(all_occ, Square::F1) && !BitUtil::get_bit(all_occ, Square::G1)) {
            if (!Attacks::is_square_attacked(Square::E1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::F1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::G1, Colour::Black, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E1, Square::G1, MoveFlag::KingCastle);
            }
        }
        if ((board.castle_rights & 2) && !BitUtil::get_bit(all_occ, Square::D1) && !BitUtil::get_bit(all_occ, Square::C1) && !BitUtil::get_bit(all_occ, Square::B1)) {
            if (!Attacks::is_square_attacked(Square::E1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::D1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::C1, Colour::Black, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E1, Square::C1, MoveFlag::QueenCastle);
            }
        }
    } else {
        if ((board.castle_rights & 4) && !BitUtil::get_bit(all_occ, Square::F8) && !BitUtil::get_bit(all_occ, Square::G8)) {
            if (!Attacks::is_square_attacked(Square::E8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::F8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::G8, Colour::White, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E8, Square::G8, MoveFlag::KingCastle);
            }
        }
        if ((board.castle_rights & 8) && !BitUtil::get_bit(all_occ, Square::D8) && !BitUtil::get_bit(all_occ, Square::C8) && !BitUtil::get_bit(all_occ, Square::B8)) {
            if (!Attacks::is_square_attacked(Square::E8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::D8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::C8, Colour::White, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E8, Square::C8, MoveFlag::QueenCastle);
            }
        }
    }
}

void generate_captures(const BoardState& board, std::vector<Move>& move_list) {
    Colour us = board.to_move;
    Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
    Bitboard us_occ = board.occupancy[static_cast<int>(us)];
    Bitboard them_occ = board.occupancy[static_cast<int>(them)];
    Bitboard all_occ = board.occupancy[2];

    // --- Pawns ---
    Bitboard pawns = board.pieces[(us == Colour::White) ? 0 : 6];
    Bitboard promo_rank = (us == Colour::White) ? 0xFF00000000000000ULL : 0x00000000000000FFULL;

    Bitboard single_push = (us == Colour::White) ? (pawns << 8) : (pawns >> 8);
    single_push &= ~all_occ; 
    Bitboard promo_pushes = single_push & promo_rank;

    while (promo_pushes) {
        Square to = BitUtil::pop_lsb(promo_pushes);
        Square from = static_cast<Square>(static_cast<int>(to) + ((us == Colour::White) ? -8 : 8));
        add_promotions(from, to, move_list, false);
    }

    Bitboard pawns_copy = pawns;
    while(pawns_copy) {
        Square from = BitUtil::pop_lsb(pawns_copy);
        Bitboard attacks = (us == Colour::White) ? Attacks::PawnAttacks[0][static_cast<int>(from)] 
                                                 : Attacks::PawnAttacks[1][static_cast<int>(from)];
        
        Bitboard valid_captures = attacks & them_occ;

        Bitboard normal_captures = valid_captures & ~promo_rank;
        Bitboard promo_captures  = valid_captures & promo_rank;

        serialize_moves(from, normal_captures, move_list, MoveFlag::Capture);

        while (promo_captures) {
            Square to = BitUtil::pop_lsb(promo_captures);
            add_promotions(from, to, move_list, true);
        }

        // En Passant
        if (board.en_passant_sq != Square::None) {
            if (attacks & (1ULL << static_cast<int>(board.en_passant_sq))) {
                move_list.emplace_back(from, board.en_passant_sq, MoveFlag::EnPassant);
            }
        }
    }

    // --- Knights ---
    Bitboard knights = board.pieces[(us == Colour::White) ? 1 : 7];
    while (knights) {
        Square from = BitUtil::pop_lsb(knights);
        Bitboard moves = Attacks::KnightAttacks[static_cast<int>(from)] & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
    }

    // --- King ---
    Bitboard king = board.pieces[(us == Colour::White) ? 5 : 11];
    if (king) {
        Square from = static_cast<Square>(BitUtil::lsb(king));
        Bitboard moves = Attacks::KingAttacks[static_cast<int>(from)] & ~us_occ;
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
    }

    // --- Sliders (Bishops, Rooks, Queens) ---
    Bitboard bishops = board.pieces[(us == Colour::White) ? 2 : 8];
    while (bishops) {
        Square from = BitUtil::pop_lsb(bishops);
        Bitboard moves = Attacks::get_bishop_attacks(static_cast<int>(from), all_occ);
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
    }

    Bitboard rooks = board.pieces[(us == Colour::White) ? 3 : 9];
    while (rooks) {
        Square from = BitUtil::pop_lsb(rooks);
        Bitboard moves = Attacks::get_rook_attacks(static_cast<int>(from), all_occ);
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
    }

    Bitboard queens = board.pieces[(us == Colour::White) ? 4 : 10];
    while (queens) {
        Square from = BitUtil::pop_lsb(queens);
        Bitboard moves = Attacks::get_queen_attacks(static_cast<int>(from), all_occ);
        serialize_moves(from, moves & them_occ, move_list, MoveFlag::Capture);
    }
}

}
