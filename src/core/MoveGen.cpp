#include "Attacks.hpp"
#include "MoveGen.hpp"
#include "BitUtil.hpp"

#include <vector>


namespace MoveGen {

namespace {
    void serialize_moves(Square from, Bitboard targets, std::vector<Move>& list, MoveFlag flag = MoveFlag::Quiet) {
        while (targets) {
            Square to = BitUtil::pop_lsb(targets);
            list.emplace_back(from, to, flag);
        }
    }
}

void generate_moves(const BoardState& board, std::vector<Move>& move_list) {
    const Colour us = board.to_move;
    const Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
    
    const Bitboard us_occ = board.occupancy[static_cast<int>(us)];
    const Bitboard them_occ = board.occupancy[static_cast<int>(them)];
    const Bitboard all_occ = board.occupancy[2];

    size_t pawn_idx = BoardState::get_piece_index(us, PieceType::Pawn);
    Bitboard pawns = board.pieces[pawn_idx];

    if (us == Colour::White) {
        Bitboard single_push = (pawns << 8) & ~all_occ;
        Bitboard double_push = ((single_push & 0xFF0000) << 8) & ~all_occ;

        Bitboard capture_left = (pawns << 7) & Attacks::detail::NOT_H & them_occ;
        Bitboard capture_right = (pawns << 9) & Attacks::detail::NOT_A & them_occ;

        while (single_push) {
            Square to = BitUtil::pop_lsb(single_push);
            // Note: Simplification - Promotion handling omitted for brevity, add check for Rank 8 here
            move_list.emplace_back(static_cast<Square>(static_cast<int>(to) - 8), to, MoveFlag::Quiet);
        }
        while (double_push) {
            Square to = BitUtil::pop_lsb(double_push);
            move_list.emplace_back(static_cast<Square>(static_cast<int>(to) - 16), to, MoveFlag::DoublePawnPush);
        }
    } 
    else { // Black Logic (Shift Down)
        Bitboard single_push = (pawns >> 8) & ~all_occ;
        Bitboard double_push = ((single_push & 0xFF0000000000) >> 8) & ~all_occ;
        Bitboard capture_left = (pawns >> 9) & Attacks::detail::NOT_H & them_occ;
        Bitboard capture_right = (pawns >> 7) & Attacks::detail::NOT_A & them_occ;

        while (single_push) {
            Square to = BitUtil::pop_lsb(single_push);
            move_list.emplace_back(static_cast<Square>(static_cast<int>(to) + 8), to, MoveFlag::Quiet);
        }
        while (double_push) {
            Square to = BitUtil::pop_lsb(double_push);
            move_list.emplace_back(static_cast<Square>(static_cast<int>(to) + 16), to, MoveFlag::DoublePawnPush);
        }
    }

    Bitboard pawns_copy = pawns;
    while(pawns_copy) {
        Square from = BitUtil::pop_lsb(pawns_copy);
        
        // 1. Basic Attacks (Diagonals)
        Bitboard attacks = (us == Colour::White) 
            ? Attacks::PawnAttacks[0][static_cast<int>(from)] 
            : Attacks::PawnAttacks[1][static_cast<int>(from)];
        
        // 2. Standard Captures (Must hit enemy occupancy)
        Bitboard regular_captures = attacks & them_occ;
        serialize_moves(from, regular_captures, move_list, MoveFlag::Capture);

        // 3. En Passant Capture (Target must be the specific EP square)
        if (board.en_passant_sq != Square::None) {
            Bitboard ep_mask = (1ULL << static_cast<int>(board.en_passant_sq));
            // If this pawn attacks the En Passant square, add the move
            if (attacks & ep_mask) {
                move_list.emplace_back(from, board.en_passant_sq, MoveFlag::EnPassant);
            }
        }
    }


    for (int pt = 1; pt <= 5; ++pt) { 
        PieceType type = static_cast<PieceType>(pt);
        size_t idx = BoardState::get_piece_index(us, type);
        Bitboard pieces = board.pieces[idx];

        while (pieces) {
            Square from = BitUtil::pop_lsb(pieces);
            Bitboard attacks = 0;

            switch (type) {
                case PieceType::Knight: 
                    attacks = Attacks::KnightAttacks[static_cast<int>(from)]; 
                    break;
                case PieceType::Bishop: 
                    attacks = Attacks::get_bishop_attacks(static_cast<int>(from), all_occ); 
                    break;
                case PieceType::Rook:   
                    attacks = Attacks::get_rook_attacks(static_cast<int>(from), all_occ); 
                    break;
                case PieceType::Queen:  
                    attacks = Attacks::get_queen_attacks(static_cast<int>(from), all_occ); 
                    break;
                case PieceType::King:   
                    attacks = Attacks::KingAttacks[static_cast<int>(from)]; 
                    break;
                default: break;
            }

            // Mask out our own pieces (can't capture self)
            attacks &= ~us_occ;

            Bitboard captures = attacks & them_occ;
            Bitboard quiets = attacks & ~them_occ;

            serialize_moves(from, captures, move_list, MoveFlag::Capture);
            serialize_moves(from, quiets, move_list, MoveFlag::Quiet);
        }
    }

    if (us == Colour::White) {
        // White King Side (E1 -> G1)
        if ((board.castle_rights & 1) &&                                    // Has Right
            !BitUtil::get_bit(all_occ, Square::F1) &&                       // F1 Empty
            !BitUtil::get_bit(all_occ, Square::G1)) {                       // G1 Empty
            
            // Safety Check: E1, F1, G1 must not be attacked
            if (!Attacks::is_square_attacked(Square::E1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::F1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::G1, Colour::Black, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E1, Square::G1, MoveFlag::KingCastle);
            }
        }

        // White Queen Side (E1 -> C1)
        if ((board.castle_rights & 2) &&                                    // Has Right
            !BitUtil::get_bit(all_occ, Square::D1) &&                       // D1 Empty
            !BitUtil::get_bit(all_occ, Square::C1) &&                       // C1 Empty
            !BitUtil::get_bit(all_occ, Square::B1)) {                       // B1 Empty (But safety NOT checked!)
            
            // Safety Check: E1, D1, C1 must not be attacked. 
            // NOTE: We do NOT check B1 for attacks!
            if (!Attacks::is_square_attacked(Square::E1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::D1, Colour::Black, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::C1, Colour::Black, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E1, Square::C1, MoveFlag::QueenCastle);
            }
        }
    } else {
        // Black King Side (E8 -> G8)
        if ((board.castle_rights & 4) &&
            !BitUtil::get_bit(all_occ, Square::F8) &&
            !BitUtil::get_bit(all_occ, Square::G8)) {
            
            if (!Attacks::is_square_attacked(Square::E8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::F8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::G8, Colour::White, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E8, Square::G8, MoveFlag::KingCastle);
            }
        }

        // Black Queen Side (E8 -> C8)
        if ((board.castle_rights & 8) &&
            !BitUtil::get_bit(all_occ, Square::D8) &&
            !BitUtil::get_bit(all_occ, Square::C8) &&
            !BitUtil::get_bit(all_occ, Square::B8)) {
            
            // NOTE: Do NOT check B8 for attacks!
            if (!Attacks::is_square_attacked(Square::E8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::D8, Colour::White, board.pieces.data(), all_occ) &&
                !Attacks::is_square_attacked(Square::C8, Colour::White, board.pieces.data(), all_occ)) {
                move_list.emplace_back(Square::E8, Square::C8, MoveFlag::QueenCastle);
            }
        }
    }
}

}
