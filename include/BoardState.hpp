#pragma once

#include "Types.hpp"
#include "BitUtil.hpp"
#include <vector>
#include <array>

struct BoardState {
    std::array<uint64_t, 12> pieces;
    std::array<uint64_t, 3> occupancy;
    
    Colour to_move;
    Square en_passant_sq;
    uint8_t castle_rights;
    uint16_t half_move_clock; 
    uint16_t full_move_number;

    struct History {
        Move move;
        uint8_t castle_rights;
        Square en_passant_sq;
        uint16_t half_move_clock;
        uint64_t captured_piece; 
    };
    std::vector<History> history;

    BoardState() {
        pieces.fill(0);
        occupancy.fill(0);
        to_move = Colour::White;
        en_passant_sq = Square::None;
        castle_rights = 0;
        half_move_clock = 0;
        full_move_number = 1;
        history.reserve(256);
    }

    void make_move(Move move) {
        Square from = move.from();
        Square to = move.to();
        MoveFlag flag = move.flag();
        
        int us = static_cast<int>(to_move);
        int them = us ^ 1;

        History h;
        h.move = move;
        h.castle_rights = castle_rights;
        h.en_passant_sq = en_passant_sq;
        h.half_move_clock = half_move_clock;
        h.captured_piece = 0; 

        // Update Occupancy (Remove From)
        BitUtil::clear_bit(occupancy[us], from);
        BitUtil::clear_bit(occupancy[2], from);
        
        // Identify moving piece
        int piece_idx = -1;
        for (int i = us * 6; i < us * 6 + 6; ++i) {
            if (BitUtil::get_bit(pieces[i], from)) {
                piece_idx = i;
                BitUtil::clear_bit(pieces[i], from);
                break;
            }
        }

        // Handle Captures
        if (BitUtil::get_bit(occupancy[them], to)) {
            for (int i = them * 6; i < them * 6 + 6; ++i) {
                if (BitUtil::get_bit(pieces[i], to)) {
                    BitUtil::clear_bit(pieces[i], to);
                    h.captured_piece = (1ULL << i); 
                    break;
                }
            }
            BitUtil::clear_bit(occupancy[them], to);
            BitUtil::clear_bit(occupancy[2], to);
            half_move_clock = 0;
        }

        // En Passant Capture
        if (flag == MoveFlag::EnPassant) {
            Square cap_sq = static_cast<Square>(static_cast<int>(to) + (us == 0 ? -8 : 8));
            BitUtil::clear_bit(pieces[them * 6], cap_sq);
            BitUtil::clear_bit(occupancy[them], cap_sq);
            BitUtil::clear_bit(occupancy[2], cap_sq);
            half_move_clock = 0;
        }

        // Update Occupancy (Place To)
        BitUtil::set_bit(occupancy[us], to);
        BitUtil::set_bit(occupancy[2], to);
        BitUtil::set_bit(pieces[piece_idx], to);

        // Castling Moves
        if (flag == MoveFlag::KingCastle) {
            Square r_from = (us == 0) ? Square::H1 : Square::H8;
            Square r_to = (us == 0) ? Square::F1 : Square::F8;
            BitUtil::clear_bit(pieces[us * 6 + 3], r_from);
            BitUtil::clear_bit(occupancy[us], r_from);
            BitUtil::clear_bit(occupancy[2], r_from);
            BitUtil::set_bit(pieces[us * 6 + 3], r_to);
            BitUtil::set_bit(occupancy[us], r_to);
            BitUtil::set_bit(occupancy[2], r_to);
        } else if (flag == MoveFlag::QueenCastle) {
            Square r_from = (us == 0) ? Square::A1 : Square::A8;
            Square r_to = (us == 0) ? Square::D1 : Square::D8;
            BitUtil::clear_bit(pieces[us * 6 + 3], r_from);
            BitUtil::clear_bit(occupancy[us], r_from);
            BitUtil::clear_bit(occupancy[2], r_from);
            BitUtil::set_bit(pieces[us * 6 + 3], r_to);
            BitUtil::set_bit(occupancy[us], r_to);
            BitUtil::set_bit(occupancy[2], r_to);
        }

        // Promotions
        if (flag >= MoveFlag::KnightPromotion && flag <= MoveFlag::QueenPromoCapture) {
            BitUtil::clear_bit(pieces[piece_idx], to); // Remove Pawn
            int promo_offset = 0;
            if (flag == MoveFlag::KnightPromotion || flag == MoveFlag::KnightPromoCapture) promo_offset = 1;
            else if (flag == MoveFlag::BishopPromotion || flag == MoveFlag::BishopPromoCapture) promo_offset = 2;
            else if (flag == MoveFlag::RookPromotion || flag == MoveFlag::RookPromoCapture) promo_offset = 3;
            else promo_offset = 4; // Queen
            BitUtil::set_bit(pieces[us * 6 + promo_offset], to);
        }

        // Update Castle Rights
        // Simple mask approach: if king/rook moves from/to source spots, strip rights
        // (A robust engine has a lookup table for this, but manual checks work for now)
        if (from == Square::E1 || to == Square::E1) castle_rights &= ~3;
        if (from == Square::E8 || to == Square::E8) castle_rights &= ~12;
        if (from == Square::H1 || to == Square::H1) castle_rights &= ~1;
        if (from == Square::A1 || to == Square::A1) castle_rights &= ~2;
        if (from == Square::H8 || to == Square::H8) castle_rights &= ~4;
        if (from == Square::A8 || to == Square::A8) castle_rights &= ~8;

        // En Passant State
        en_passant_sq = Square::None;
        if (flag == MoveFlag::DoublePawnPush) {
            en_passant_sq = static_cast<Square>(static_cast<int>(from) + (us == 0 ? 8 : -8));
        }

        history.push_back(h);
        to_move = (to_move == Colour::White) ? Colour::Black : Colour::White;
        if (to_move == Colour::White) full_move_number++;
    }

    void undo_move(Move move) {
        if (history.empty()) return;
        History h = history.back();
        history.pop_back();

        if (to_move == Colour::White) full_move_number--;
        to_move = (to_move == Colour::White) ? Colour::Black : Colour::White;
        
        castle_rights = h.castle_rights;
        en_passant_sq = h.en_passant_sq;
        half_move_clock = h.half_move_clock;

        Square from = move.from();
        Square to = move.to();
        MoveFlag flag = move.flag();
        int us = static_cast<int>(to_move);
        int them = us ^ 1;

        // 1. Move the piece back
        int piece_idx = -1;
        // Determine what piece is currently on 'to' (handling promotion cases)
        for (int i = us * 6; i < us * 6 + 6; ++i) {
            if (BitUtil::get_bit(pieces[i], to)) {
                piece_idx = i;
                BitUtil::clear_bit(pieces[i], to);
                break;
            }
        }
        
        // If promotion, we just removed the Queen/Rook/etc. Now put the pawn back at 'from'.
        if (flag >= MoveFlag::KnightPromotion) {
            BitUtil::set_bit(pieces[us * 6], from);
        } else {
            BitUtil::set_bit(pieces[piece_idx], from);
        }
        
        BitUtil::clear_bit(occupancy[us], to);
        BitUtil::set_bit(occupancy[us], from);

        // 2. Restore captured piece
        if (h.captured_piece) {
            int cap_idx = BitUtil::lsb(h.captured_piece); // This is the piece TYPE index (0-11)
            // Wait, h.captured_piece stores (1ULL << index), so lsb gives index
            BitUtil::set_bit(pieces[cap_idx], to);
            BitUtil::set_bit(occupancy[them], to);
        }

        // 3. En Passant Special Case
        if (flag == MoveFlag::EnPassant) {
            Square cap_sq = static_cast<Square>(static_cast<int>(to) + (us == 0 ? -8 : 8));
            BitUtil::set_bit(pieces[them * 6], cap_sq);
            BitUtil::set_bit(occupancy[them], cap_sq);
        }

        // 4. Castling Unmake
        if (flag == MoveFlag::KingCastle) {
            Square r_from = (us == 0) ? Square::H1 : Square::H8;
            Square r_to = (us == 0) ? Square::F1 : Square::F8;
            BitUtil::clear_bit(pieces[us * 6 + 3], r_to);
            BitUtil::clear_bit(occupancy[us], r_to);
            BitUtil::set_bit(pieces[us * 6 + 3], r_from);
            BitUtil::set_bit(occupancy[us], r_from);
        } else if (flag == MoveFlag::QueenCastle) {
            Square r_from = (us == 0) ? Square::A1 : Square::A8;
            Square r_to = (us == 0) ? Square::D1 : Square::D8;
            BitUtil::clear_bit(pieces[us * 6 + 3], r_to);
            BitUtil::clear_bit(occupancy[us], r_to);
            BitUtil::set_bit(pieces[us * 6 + 3], r_from);
            BitUtil::set_bit(occupancy[us], r_from);
        }

        // Rebuild All Occupancy
        occupancy[2] = occupancy[0] | occupancy[1];
    }
};
