#include "Interface.hpp"
#include "imgui.h"
#include "imgui-SFML.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "BoardState.hpp"
#include "MoveGen.hpp"
#include "Search.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"

// --- VISUAL CONSTANTS ---
const int TILE_SIZE = 75;
const int BOARD_PADDING = 30;
const int PANEL_WIDTH = 300;
const int BOARD_PIXEL_SIZE = 8 * TILE_SIZE;
const int OFFSET_X = BOARD_PADDING;
const int OFFSET_Y = BOARD_PADDING;
const int WIN_WIDTH = BOARD_PIXEL_SIZE + (2 * BOARD_PADDING) + PANEL_WIDTH;
const int WIN_HEIGHT = BOARD_PIXEL_SIZE + (2 * BOARD_PADDING);

// --- ASSETS ---
struct Assets {
    sf::Font font;
    std::map<int, sf::Texture> textures;
    bool has_font = false;

    void load() {
        if (font.loadFromFile("assets/font.TTF")) has_font = true;
        
        auto load_piece = [&](int id, std::string filename) {
            sf::Texture tex;
            if (tex.loadFromFile("assets/" + filename)) {
                tex.setSmooth(true);
                textures[id] = tex;
            }
        };
        // WHITE
        load_piece(1, "Chess_plt45.png"); load_piece(2, "Chess_nlt45.png");
        load_piece(3, "Chess_blt45.png"); load_piece(4, "Chess_rlt45.png");
        load_piece(5, "Chess_qlt45.png"); load_piece(6, "Chess_klt45.png");
        // BLACK
        load_piece(-1, "Chess_pdt45.png"); load_piece(-2, "Chess_ndt45.png");
        load_piece(-3, "Chess_bdt45.png"); load_piece(-4, "Chess_rdt45.png");
        load_piece(-5, "Chess_qdt45.png"); load_piece(-6, "Chess_kdt45.png");
    }
};

// --- HELPERS ---
Square get_square_at(int mouse_x, int mouse_y) {
    int x = mouse_x - OFFSET_X;
    int y = mouse_y - OFFSET_Y;
    if (x < 0 || x >= BOARD_PIXEL_SIZE || y < 0 || y >= BOARD_PIXEL_SIZE) return Square::None;
    int file = x / TILE_SIZE;
    int rank = 7 - (y / TILE_SIZE); 
    return static_cast<Square>(rank * 8 + file);
}

int get_piece_at(const BoardState& b, Square sq) {
    int s = static_cast<int>(sq);
    for (int i = 0; i < 12; ++i) {
        if (BitUtil::get_bit(b.pieces[i], sq)) return (i < 6) ? (i + 1) : -(i - 5);
    }
    return 0; 
}

// --- MAIN LAUNCHER ---
namespace GUI {
    void Launch(Search::EvalCallback evalFunc, int depth) {
        sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Chess Engine");
        window.setFramerateLimit(30);
        ImGui::SFML::Init(window);

        // Init Engine Logic
        Attacks::init(); 
        BoardState board;
        // Standard Start
        board.pieces[0] = 0x000000000000FF00ULL; board.pieces[6] = 0x00FF000000000000ULL;
        board.pieces[1] = 0x0000000000000042ULL; board.pieces[7] = 0x4200000000000000ULL;
        board.pieces[2] = 0x0000000000000024ULL; board.pieces[8] = 0x2400000000000000ULL;
        board.pieces[3] = 0x0000000000000081ULL; board.pieces[9] = 0x8100000000000000ULL;
        board.pieces[4] = 0x0000000000000008ULL; board.pieces[10] = 0x0800000000000000ULL;
        board.pieces[5] = 0x0000000000000010ULL; board.pieces[11] = 0x1000000000000000ULL;
        board.occupancy[0] = 0ULL; board.occupancy[1] = 0ULL;
        for (int i = 0; i < 6; ++i) board.occupancy[0] |= board.pieces[i];
        for (int i = 6; i < 12; ++i) board.occupancy[1] |= board.pieces[i];
        board.occupancy[2] = board.occupancy[0] | board.occupancy[1];
        board.to_move = Colour::White;
        board.castle_rights = 0b1111;
        board.full_move_number = 1;

        Assets assets;
        assets.load();

        std::map<int, std::string> piece_symbols;
        piece_symbols[1] = "P"; piece_symbols[2] = "N"; piece_symbols[3] = "B";
        piece_symbols[4] = "R"; piece_symbols[5] = "Q"; piece_symbols[6] = "K";
        piece_symbols[-1] = "p"; piece_symbols[-2] = "n"; piece_symbols[-3] = "b";
        piece_symbols[-4] = "r"; piece_symbols[-5] = "q"; piece_symbols[-6] = "k";

        Square selected_sq = Square::None;
        std::vector<Move> valid_moves;
        sf::Clock deltaClock;

        // --- PROMOTION STATE ---
        bool is_promoting = false;
        Square promo_from = Square::None;
        Square promo_to = Square::None;

        // Helper to Draw
        auto render_board = [&]() {
            window.clear(sf::Color(30, 30, 30));

            // 1. Board
            sf::RectangleShape tile(sf::Vector2f(TILE_SIZE, TILE_SIZE));
            for (int r = 0; r < 8; ++r) {
                for (int f = 0; f < 8; ++f) {
                    bool is_light = ((r + f) % 2 != 0);
                    tile.setFillColor(is_light ? sf::Color(240, 217, 181) : sf::Color(181, 136, 99));
                    tile.setPosition(OFFSET_X + f * TILE_SIZE, OFFSET_Y + (7 - r) * TILE_SIZE); 
                    window.draw(tile);
                }
            }

            // 2. Highlights (Only if NOT promoting)
            if (!is_promoting && selected_sq != Square::None) {
                tile.setFillColor(sf::Color(255, 255, 0, 100)); 
                int sel_r = static_cast<int>(selected_sq) / 8;
                int sel_f = static_cast<int>(selected_sq) % 8;
                tile.setPosition(OFFSET_X + sel_f * TILE_SIZE, OFFSET_Y + (7 - sel_r) * TILE_SIZE);
                window.draw(tile);

                tile.setFillColor(sf::Color(100, 255, 100, 100)); 
                for (const auto& m : valid_moves) {
                    int to_r = static_cast<int>(m.to()) / 8;
                    int to_f = static_cast<int>(m.to()) % 8;
                    tile.setPosition(OFFSET_X + to_f * TILE_SIZE, OFFSET_Y + (7 - to_r) * TILE_SIZE);
                    window.draw(tile);
                }
            }

            // 3. Pieces
            for (int i = 0; i < 64; ++i) {
                // Don't draw the pawn if it's currently promoting (it's in limbo)
                if (is_promoting && static_cast<Square>(i) == promo_from) continue;

                int p = get_piece_at(board, static_cast<Square>(i));
                if (p != 0) {
                    int r = i / 8; int f = i % 8;
                    float x = OFFSET_X + f * TILE_SIZE + TILE_SIZE/2.0f;
                    float y = OFFSET_Y + (7 - r) * TILE_SIZE + TILE_SIZE/2.0f;

                    if (assets.textures.count(p)) {
                        sf::Sprite s(assets.textures[p]);
                        float scale = (TILE_SIZE * 0.85f) / s.getLocalBounds().width;
                        s.setScale(scale, scale);
                        s.setOrigin(s.getLocalBounds().width/2, s.getLocalBounds().height/2);
                        s.setPosition(x, y);
                        window.draw(s);
                    } else if (assets.has_font) {
                        sf::Text t; t.setFont(assets.font); t.setString(piece_symbols[p]);
                        t.setCharacterSize(TILE_SIZE*0.6f);
                        t.setFillColor(p>0?sf::Color::White:sf::Color::Black);
                        t.setOutlineThickness(2);
                        t.setOrigin(t.getLocalBounds().width/2, t.getLocalBounds().height/2);
                        t.setPosition(x - 10, y - 10);
                        window.draw(t);
                    }
                }
            }

            // 4. Promotion UI (Overlay)
            if (is_promoting) {
                // Dim background
                sf::RectangleShape overlay(sf::Vector2f(WIN_WIDTH, WIN_HEIGHT));
                overlay.setFillColor(sf::Color(0, 0, 0, 150));
                window.draw(overlay);

                // Draw Options (Queen, Rook, Bishop, Knight)
                // Center of the screen
                float cx = WIN_WIDTH / 2.0f;
                float cy = WIN_HEIGHT / 2.0f;
                int ids[] = {5, 4, 3, 2}; // White IDs
                if (board.to_move == Colour::Black) { ids[0]=-5; ids[1]=-4; ids[2]=-3; ids[3]=-2; }

                for(int i=0; i<4; ++i) {
                    if (assets.textures.count(ids[i])) {
                        sf::Sprite s(assets.textures[ids[i]]);
                        // Make them big
                        float scale = (TILE_SIZE * 1.2f) / s.getLocalBounds().width;
                        s.setScale(scale, scale);
                        s.setOrigin(s.getLocalBounds().width/2, s.getLocalBounds().height/2);
                        // Arrange in a row
                        s.setPosition(cx + (i - 1.5f) * (TILE_SIZE * 1.5f), cy);
                        window.draw(s);
                    }
                }
            }
        };

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                ImGui::SFML::ProcessEvent(window, event);
                if (event.type == sf::Event::Closed) window.close();

                if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                    if (ImGui::GetIO().WantCaptureMouse) continue;

                    // --- PROMOTION CLICK HANDLING ---
                    if (is_promoting) {
                        float cx = WIN_WIDTH / 2.0f;
                        float cy = WIN_HEIGHT / 2.0f;
                        float btn_size = TILE_SIZE * 1.5f;
                        int clicked_idx = -1;

                        // Check which piece was clicked
                        for(int i=0; i<4; ++i) {
                            float btn_x = cx + (i - 1.5f) * btn_size;
                            float mx = (float)event.mouseButton.x;
                            float my = (float)event.mouseButton.y;
                            
                            // Simple box check
                            if (mx > btn_x - btn_size/2 && mx < btn_x + btn_size/2 &&
                                my > cy - btn_size/2 && my < cy + btn_size/2) {
                                clicked_idx = i;
                                break;
                            }
                        }

                        if (clicked_idx != -1) {
                            // Find the matching move in valid_moves
                            for(const auto& m : valid_moves) {
                                if (m.to() == promo_to && m.from() == promo_from && m.is_promotion()) {
                                    bool match = false;
                                    // Map index 0->Q, 1->R, 2->B, 3->N
                                    if (clicked_idx == 0 && m.is_promo_queen()) match = true;
                                    if (clicked_idx == 1 && m.is_promo_rook()) match = true;
                                    if (clicked_idx == 2 && m.is_promo_bishop()) match = true;
                                    if (clicked_idx == 3 && m.is_promo_knight()) match = true;

                                    if (match) {
                                        board.make_move(m);
                                        is_promoting = false;
                                        selected_sq = Square::None;
                                        valid_moves.clear();
                                        
                                        // Update & Bot Move
                                        render_board(); window.display();
                                        
                                        Search::SearchParams params;
                                        params.depth = depth; params.evalFunc = evalFunc;
                                        Move botMove = Search::iterative_deepening(board, params);
                                        board.make_move(botMove);
                                        break;
                                    }
                                }
                            }
                        }
                    } 
                    // --- NORMAL CLICK HANDLING ---
                    else {
                        Square clicked = get_square_at(event.mouseButton.x, event.mouseButton.y);
                        if (clicked != Square::None) {
                            bool moved = false;
                            
                            if (selected_sq != Square::None) {
                                for (const auto& m : valid_moves) {
                                    if (m.to() == clicked) {
                                        // DETECT PROMOTION START
                                        if (m.is_promotion()) {
                                            is_promoting = true;
                                            promo_from = m.from();
                                            promo_to = m.to();
                                            moved = true; 
                                            // Break without making move yet!
                                            break; 
                                        }

                                        board.make_move(m);
                                        selected_sq = Square::None;
                                        valid_moves.clear();
                                        moved = true;
                                        
                                        render_board(); window.display();
                                        
                                        Search::SearchParams params;
                                        params.depth = depth; params.evalFunc = evalFunc;
                                        Move botMove = Search::iterative_deepening(board, params);
                                        board.make_move(botMove);
                                        break;
                                    }
                                }
                            }

                            if (!moved && !is_promoting) {
                                // Select Logic
                                int p = get_piece_at(board, clicked);
                                bool is_white = (p > 0);
                                if (p != 0 && ((board.to_move == Colour::White) == is_white)) {
                                    selected_sq = clicked;
                                    valid_moves.clear();
                                    std::vector<Move> all_moves;
                                    MoveGen::generate_moves(board, all_moves);
                                    for (const auto& m : all_moves) {
                                        if (m.from() == selected_sq) {
                                            board.make_move(m);
                                            Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White; 
                                            Square king_sq = Search::find_king(board, us);
                                            bool illegal = Attacks::is_square_attacked(king_sq, board.to_move, board.pieces.data(), board.occupancy[2]);
                                            board.undo_move(m);
                                            if (!illegal) valid_moves.push_back(m);
                                        }
                                    }
                                } else {
                                    selected_sq = Square::None;
                                    valid_moves.clear();
                                }
                            }
                        }
                    }
                }
            }

            ImGui::SFML::Update(window, deltaClock.restart());
            render_board();

            // UI
            ImGui::SetNextWindowPos(sf::Vector2f(WIN_WIDTH - PANEL_WIDTH, 0));
            ImGui::SetNextWindowSize(sf::Vector2f(PANEL_WIDTH, WIN_HEIGHT));
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration);
            if(is_promoting) ImGui::TextColored(ImVec4(1,0,0,1), "PROMOTION: Pick a piece!");
            ImGui::Text("Turn: %s", (board.to_move == Colour::White ? "White" : "Black"));
            if (ImGui::Button("Reset")) {
                board = BoardState(); 
                board.pieces[0] = 0x000000000000FF00ULL; board.pieces[6] = 0x00FF000000000000ULL;
                board.pieces[1] = 0x0000000000000042ULL; board.pieces[7] = 0x4200000000000000ULL;
                board.pieces[2] = 0x0000000000000024ULL; board.pieces[8] = 0x2400000000000000ULL;
                board.pieces[3] = 0x0000000000000081ULL; board.pieces[9] = 0x8100000000000000ULL;
                board.pieces[4] = 0x0000000000000008ULL; board.pieces[10] = 0x0800000000000000ULL;
                board.pieces[5] = 0x0000000000000010ULL; board.pieces[11] = 0x1000000000000000ULL;
                board.occupancy[0] = 0ULL; board.occupancy[1] = 0ULL;
                for (int i = 0; i < 6; ++i) board.occupancy[0] |= board.pieces[i];
                for (int i = 6; i < 12; ++i) board.occupancy[1] |= board.pieces[i];
                board.occupancy[2] = board.occupancy[0] | board.occupancy[1];
                board.to_move = Colour::White;
                board.castle_rights = 0b1111;
                board.full_move_number = 1;
                is_promoting = false;
            }
            ImGui::End();

            ImGui::SFML::Render(window);
            window.display();
        }
        ImGui::SFML::Shutdown();
    }
}
