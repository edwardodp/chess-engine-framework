#include "BoardState.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"

// Helper to get the piece type at a specific square (for captures)
PieceType get_piece_at(const BoardState& board, Square sq, Colour side) {
    size_t offset = static_cast<size_t>(side) * 6;
    for (int i = 0; i < 6; ++i) {
        if (BitUtil::get_bit(board.pieces[offset + i], sq)) {
            return static_cast<PieceType>(i);
        }
    }
    return PieceType::None;
}

void BoardState::make_move(Move m) {
    UndoInfo undo;
    undo.castle_rights = this->castle_rights;
    undo.en_passant_sq = this->en_passant_sq;
    undo.halfmove_clock = this->halfmove_clock;
    undo.captured_piece = PieceType::None;
    // undo.pos_hash = this->zobrist_key; // TODO: Implement Zobrist later
    
    Square from = m.from();
    Square to = m.to();
    MoveFlag flag = m.flags();

    // Identify the moving piece
    Colour us = this->to_move;
    Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
    PieceType piece = get_piece_at(*this, from, us);
    
    if (m.is_capture()) {
        PieceType captured = PieceType::None;
        
        if (flag == MoveFlag::EnPassant) {
            Square cap_sq = (us == Colour::White) 
                ? static_cast<Square>(static_cast<int>(to) - 8) 
                : static_cast<Square>(static_cast<int>(to) + 8);
            
            captured = PieceType::Pawn;
            BitUtil::clear_bit(this->pieces[get_piece_index(them, PieceType::Pawn)], cap_sq);
        } else {
            captured = get_piece_at(*this, to, them);
            BitUtil::clear_bit(this->pieces[get_piece_index(them, captured)], to);
        }
        
        undo.captured_piece = captured;
        this->halfmove_clock = 0;
    } else {
        // Increment halfmove clock for non-capture, reset for Pawn moves
        if (piece == PieceType::Pawn) this->halfmove_clock = 0;
        else this->halfmove_clock++;
    }

    this->history.push_back(undo);

    size_t us_idx = get_piece_index(us, piece);
    
    BitUtil::clear_bit(this->pieces[us_idx], from);
    BitUtil::set_bit(this->pieces[us_idx], to);

    if (m.is_promotion()) {
        BitUtil::clear_bit(this->pieces[us_idx], to);
        
        PieceType promo_type = PieceType::None;
        switch (flag) {
            case MoveFlag::QueenPromo:        promo_type = PieceType::Queen; break;
            case MoveFlag::QueenPromoCapture: promo_type = PieceType::Queen; break;
            case MoveFlag::RookPromo:         promo_type = PieceType::Rook; break;
            case MoveFlag::RookPromoCapture:  promo_type = PieceType::Rook; break;
            case MoveFlag::BishopPromo:       promo_type = PieceType::Bishop; break;
            case MoveFlag::BishopPromoCapture:promo_type = PieceType::Bishop; break;
            case MoveFlag::KnightPromo:       promo_type = PieceType::Knight; break;
            case MoveFlag::KnightPromoCapture:promo_type = PieceType::Knight; break;
            default: break;
        }
        BitUtil::set_bit(this->pieces[get_piece_index(us, promo_type)], to);
    }

    if (flag == MoveFlag::KingCastle) {
        if (us == Colour::White) {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::H1);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::F1);
        } else {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::H8);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::F8);
        }
    } else if (flag == MoveFlag::QueenCastle) {
        if (us == Colour::White) {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::A1);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::D1);
        } else {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::A8);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::D8);
        }
    }

    if (flag == MoveFlag::DoublePawnPush) {
        this->en_passant_sq = (us == Colour::White) 
            ? static_cast<Square>(static_cast<int>(to) - 8) 
            : static_cast<Square>(static_cast<int>(to) + 8);
    } else {
        this->en_passant_sq = Square::None;
    }

    if (piece == PieceType::King) {
        if (us == Colour::White) this->castle_rights &= ~3; // Clear 1 and 2
        else this->castle_rights &= ~12; // Clear 4 and 8
    }

    // White Rooks
    if (from == Square::H1 || to == Square::H1) this->castle_rights &= ~1; // WK
    if (from == Square::A1 || to == Square::A1) this->castle_rights &= ~2; // WQ
    // Black Rooks
    if (from == Square::H8 || to == Square::H8) this->castle_rights &= ~4; // BK
    if (from == Square::A8 || to == Square::A8) this->castle_rights &= ~8; // BQ

    this->occupancy[0] = 0;
    this->occupancy[1] = 0;
    for (int i = 0; i < 6; ++i) this->occupancy[0] |= this->pieces[i];
    for (int i = 6; i < 12; ++i) this->occupancy[1] |= this->pieces[i];
    this->occupancy[2] = this->occupancy[0] | this->occupancy[1];

    this->to_move = them;
    if (us == Colour::Black) this->full_move_number++;
}

void BoardState::undo_move(Move m) {
    UndoInfo undo = this->history.back();
    this->history.pop_back();

    this->castle_rights = undo.castle_rights;
    this->en_passant_sq = undo.en_passant_sq;
    this->halfmove_clock = undo.halfmove_clock;
    if (this->to_move == Colour::White) this->full_move_number--;
    
    Colour us = (this->to_move == Colour::White) ? Colour::Black : Colour::White;
    this->to_move = us;

    Square from = m.from();
    Square to = m.to();
    MoveFlag flag = m.flags();

    PieceType moved_piece = get_piece_at(*this, to, us);
    
    if (m.is_promotion()) {
        size_t promo_idx = get_piece_index(us, moved_piece);
        BitUtil::clear_bit(this->pieces[promo_idx], to); // Remove promoted piece
        BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Pawn)], from); // Restore Pawn
    } else {
        size_t idx = get_piece_index(us, moved_piece);
        BitUtil::clear_bit(this->pieces[idx], to);
        BitUtil::set_bit(this->pieces[idx], from);
    }

    if (undo.captured_piece != PieceType::None) {
        Colour enemy = (us == Colour::White) ? Colour::Black : Colour::White;
        
        Square cap_sq = to;
        if (flag == MoveFlag::EnPassant) {
            cap_sq = (us == Colour::White) 
                ? static_cast<Square>(static_cast<int>(to) - 8) 
                : static_cast<Square>(static_cast<int>(to) + 8);
        }

        BitUtil::set_bit(this->pieces[get_piece_index(enemy, undo.captured_piece)], cap_sq);
    }

    if (flag == MoveFlag::KingCastle) {
        if (us == Colour::White) {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::F1);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::H1);
        } else {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::F8);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::H8);
        }
    } else if (flag == MoveFlag::QueenCastle) {
        if (us == Colour::White) {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::D1);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::A1);
        } else {
            BitUtil::clear_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::D8);
            BitUtil::set_bit(this->pieces[get_piece_index(us, PieceType::Rook)], Square::A8);
        }
    }

    this->occupancy[0] = 0;
    this->occupancy[1] = 0;
    for (int i = 0; i < 6; ++i) this->occupancy[0] |= this->pieces[i];
    for (int i = 6; i < 12; ++i) this->occupancy[1] |= this->pieces[i];
    this->occupancy[2] = this->occupancy[0] | this->occupancy[1];
}
