#pragma once

#include "Types.hpp"
#include "BitUtil.hpp"
#include "Zobrist.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>

struct BoardState {
    std::array<uint64_t, 12> pieces;
    std::array<uint64_t, 3> occupancy;
    
    Colour to_move;
    Square en_passant_sq;
    uint8_t castle_rights;
    uint16_t half_move_clock; 
    uint16_t full_move_number;
    uint64_t key;

    struct History {
        Move move;
        uint8_t castle_rights;
        Square en_passant_sq;
        uint16_t half_move_clock;
        uint64_t captured_piece; 
        uint64_t key; // Store hash history
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
        key = 0;
        history.reserve(256);
    }

    void refresh_hash() {
        key = 0;
        for (int p = 0; p < 12; ++p) {
            uint64_t bb = pieces[p];
            while (bb) {
                int sq = BitUtil::lsb(bb);
                BitUtil::clear_bit(bb, static_cast<Square>(sq));
                key ^= Zobrist::piece_keys[p][sq];
            }
        }
        key ^= Zobrist::castle_keys[castle_rights];
        if (en_passant_sq != Square::None) {
            key ^= Zobrist::en_passant_keys[static_cast<int>(en_passant_sq)];
        }
        if (to_move == Colour::Black) {
            key ^= Zobrist::side_key;
        }
    }

    void load_fen(std::string fen) {
        pieces.fill(0);
        occupancy.fill(0);
        history.clear();
        
        std::stringstream ss(fen);
        std::string placement, turn, castling, ep, half, full;
        ss >> placement >> turn >> castling >> ep >> half >> full;

        int rank = 7;
        int file = 0;
        
        for (char c : placement) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (isdigit(c)) {
                file += (c - '0');
            } else {
                int piece = -1;
                switch(c) {
                    case 'P': piece = 0; break;
                    case 'N': piece = 1; break;
                    case 'B': piece = 2; break;
                    case 'R': piece = 3; break;
                    case 'Q': piece = 4; break;
                    case 'K': piece = 5; break;
                    case 'p': piece = 6; break;
                    case 'n': piece = 7; break;
                    case 'b': piece = 8; break;
                    case 'r': piece = 9; break;
                    case 'q': piece = 10; break;
                    case 'k': piece = 11; break;
                }
                if (piece != -1) {
                    Square sq = static_cast<Square>(rank * 8 + file);
                    BitUtil::set_bit(pieces[piece], sq);
                    file++;
                }
            }
        }

        for (int i = 0; i < 6; ++i) occupancy[0] |= pieces[i];
        for (int i = 6; i < 12; ++i) occupancy[1] |= pieces[i];
        occupancy[2] = occupancy[0] | occupancy[1];

        to_move = (turn == "w") ? Colour::White : Colour::Black;

        castle_rights = 0;
        if (castling != "-") {
            if (castling.find('K') != std::string::npos) castle_rights |= 1;
            if (castling.find('Q') != std::string::npos) castle_rights |= 2;
            if (castling.find('k') != std::string::npos) castle_rights |= 4;
            if (castling.find('q') != std::string::npos) castle_rights |= 8;
        }

        en_passant_sq = Square::None;
        if (ep != "-") {
            int f = ep[0] - 'a';
            int r = ep[1] - '1';
            en_passant_sq = static_cast<Square>(r * 8 + f);
        }

        try {
            half_move_clock = std::stoi(half);
            full_move_number = std::stoi(full);
        } catch (...) {
            half_move_clock = 0;
            full_move_number = 1;
        }

        refresh_hash();
    }

    bool is_draw() const {
        if (half_move_clock >= 100) return true;

        int rep_count = 0;
        int limit = std::min((int)history.size(), (int)half_move_clock);
        
        for (int i = history.size() - 2; i >= (int)history.size() - limit; i -= 2) {
            if (history[i].key == key) {
                rep_count++;
                if (rep_count >= 2) return true;  // Threefold: current + 2 previous
            }
        }
        return false;
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
        h.key = key; // Save current hash

        // --- HASH UPDATE (REMOVE OLD STATE) ---
        if (en_passant_sq != Square::None) key ^= Zobrist::en_passant_keys[static_cast<int>(en_passant_sq)];
        key ^= Zobrist::castle_keys[castle_rights];
        
        // Update Logic
        half_move_clock++; // Increment clock by default

        // Identify moving piece
        int piece_idx = -1;
        for (int i = us * 6; i < us * 6 + 6; ++i) {
            if (BitUtil::get_bit(pieces[i], from)) {
                piece_idx = i;
                break;
            }
        }

        // Remove From (Hash & Bitboard)
        BitUtil::clear_bit(pieces[piece_idx], from);
        BitUtil::clear_bit(occupancy[us], from);
        BitUtil::clear_bit(occupancy[2], from);
        key ^= Zobrist::piece_keys[piece_idx][static_cast<int>(from)];

        // Pawn Move Reset
        if (piece_idx == 0 || piece_idx == 6) half_move_clock = 0;

        // Captures
        if (BitUtil::get_bit(occupancy[them], to)) {
            half_move_clock = 0; // Reset 50mr
            for (int i = them * 6; i < them * 6 + 6; ++i) {
                if (BitUtil::get_bit(pieces[i], to)) {
                    BitUtil::clear_bit(pieces[i], to);
                    h.captured_piece = (1ULL << i); 
                    key ^= Zobrist::piece_keys[i][static_cast<int>(to)]; // Hash out capture
                    break;
                }
            }
            BitUtil::clear_bit(occupancy[them], to);
            BitUtil::clear_bit(occupancy[2], to);
        }

        // En Passant Capture
        if (flag == MoveFlag::EnPassant) {
            Square cap_sq = static_cast<Square>(static_cast<int>(to) + (us == 0 ? -8 : 8));
            BitUtil::clear_bit(pieces[them * 6], cap_sq);
            BitUtil::clear_bit(occupancy[them], cap_sq);
            BitUtil::clear_bit(occupancy[2], cap_sq);
            key ^= Zobrist::piece_keys[them * 6][static_cast<int>(cap_sq)]; // Hash out EP capture
            half_move_clock = 0;
        }

        // Place Piece (Hash & Bitboard)
        int final_piece_idx = piece_idx;
        
        // Promotions
        if (flag >= MoveFlag::KnightPromotion) {
             // 0=P, 1=N, 2=B, 3=R, 4=Q
             int offset = 0;
             if (flag == MoveFlag::KnightPromotion || flag == MoveFlag::KnightPromoCapture) offset = 1;
             else if (flag == MoveFlag::BishopPromotion || flag == MoveFlag::BishopPromoCapture) offset = 2;
             else if (flag == MoveFlag::RookPromotion || flag == MoveFlag::RookPromoCapture) offset = 3;
             else offset = 4; // Queen
             final_piece_idx = us * 6 + offset;
        }

        BitUtil::set_bit(pieces[final_piece_idx], to);
        BitUtil::set_bit(occupancy[us], to);
        BitUtil::set_bit(occupancy[2], to);
        key ^= Zobrist::piece_keys[final_piece_idx][static_cast<int>(to)]; // Hash in new piece

        // Castling Physical Move
        if (flag == MoveFlag::KingCastle) {
            Square r_from = (us == 0) ? Square::H1 : Square::H8;
            Square r_to = (us == 0) ? Square::F1 : Square::F8;
            int r_idx = us * 6 + 3;
            // Move Rook
            BitUtil::clear_bit(pieces[r_idx], r_from);
            BitUtil::clear_bit(occupancy[us], r_from);
            BitUtil::clear_bit(occupancy[2], r_from);
            key ^= Zobrist::piece_keys[r_idx][static_cast<int>(r_from)];

            BitUtil::set_bit(pieces[r_idx], r_to);
            BitUtil::set_bit(occupancy[us], r_to);
            BitUtil::set_bit(occupancy[2], r_to);
            key ^= Zobrist::piece_keys[r_idx][static_cast<int>(r_to)];
        } 
        else if (flag == MoveFlag::QueenCastle) {
            Square r_from = (us == 0) ? Square::A1 : Square::A8;
            Square r_to = (us == 0) ? Square::D1 : Square::D8;
            int r_idx = us * 6 + 3;
            // Move Rook
            BitUtil::clear_bit(pieces[r_idx], r_from);
            BitUtil::clear_bit(occupancy[us], r_from);
            BitUtil::clear_bit(occupancy[2], r_from);
            key ^= Zobrist::piece_keys[r_idx][static_cast<int>(r_from)];

            BitUtil::set_bit(pieces[r_idx], r_to);
            BitUtil::set_bit(occupancy[us], r_to);
            BitUtil::set_bit(occupancy[2], r_to);
            key ^= Zobrist::piece_keys[r_idx][static_cast<int>(r_to)];
        }

        // Update Rights
        if (from == Square::E1 || to == Square::E1) castle_rights &= ~3;
        if (from == Square::E8 || to == Square::E8) castle_rights &= ~12;
        if (from == Square::H1 || to == Square::H1) castle_rights &= ~1;
        if (from == Square::A1 || to == Square::A1) castle_rights &= ~2;
        if (from == Square::H8 || to == Square::H8) castle_rights &= ~4;
        if (from == Square::A8 || to == Square::A8) castle_rights &= ~8;

        // Update En Passant
        en_passant_sq = Square::None;
        if (flag == MoveFlag::DoublePawnPush) {
            en_passant_sq = static_cast<Square>(static_cast<int>(from) + (us == 0 ? 8 : -8));
        }

        // --- HASH UPDATE (ADD NEW STATE) ---
        if (en_passant_sq != Square::None) key ^= Zobrist::en_passant_keys[static_cast<int>(en_passant_sq)];
        key ^= Zobrist::castle_keys[castle_rights];
        key ^= Zobrist::side_key; // Flip side

        to_move = (to_move == Colour::White) ? Colour::Black : Colour::White;
        if (to_move == Colour::White) full_move_number++;

        history.push_back(h);
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
        key = h.key; // Restore Hash directly!

        Square from = move.from();
        Square to = move.to();
        MoveFlag flag = move.flag();
        int us = static_cast<int>(to_move);
        int them = us ^ 1;

        // Move the piece back
        int piece_idx = -1;
        for (int i = us * 6; i < us * 6 + 6; ++i) {
            if (BitUtil::get_bit(pieces[i], to)) {
                piece_idx = i;
                BitUtil::clear_bit(pieces[i], to);
                break;
            }
        }
        
        if (flag >= MoveFlag::KnightPromotion) {
            BitUtil::set_bit(pieces[us * 6], from);
        } else {
            BitUtil::set_bit(pieces[piece_idx], from);
        }
        
        BitUtil::clear_bit(occupancy[us], to);
        BitUtil::set_bit(occupancy[us], from);

        if (h.captured_piece) {
            int cap_idx = BitUtil::lsb(h.captured_piece);
            BitUtil::set_bit(pieces[cap_idx], to);
            BitUtil::set_bit(occupancy[them], to);
        }

        if (flag == MoveFlag::EnPassant) {
            Square cap_sq = static_cast<Square>(static_cast<int>(to) + (us == 0 ? -8 : 8));
            BitUtil::set_bit(pieces[them * 6], cap_sq);
            BitUtil::set_bit(occupancy[them], cap_sq);
        }

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

        occupancy[2] = occupancy[0] | occupancy[1];
    }
};
