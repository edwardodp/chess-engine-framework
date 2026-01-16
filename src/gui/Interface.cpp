#include "Interface.hpp"
#include "imgui.h"
#include "imgui-SFML.h"
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <map>

#include "BoardState.hpp"
#include "MoveGen.hpp"
#include "Search.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"

const int TILE_SIZE = 75;
const int BOARD_PADDING = 30;
const int PANEL_WIDTH = 300;
const int BOARD_PIXEL_SIZE = 8 * TILE_SIZE;
const int OFFSET_X = BOARD_PADDING;
const int OFFSET_Y = BOARD_PADDING;
const int WIN_WIDTH = BOARD_PIXEL_SIZE + (2 * BOARD_PADDING) + PANEL_WIDTH;
const int WIN_HEIGHT = BOARD_PIXEL_SIZE + (2 * BOARD_PADDING);

struct Assets {
    sf::Font font;
    std::map<int, sf::Texture> textures;
    bool has_font = false;
    void load() {
        if (font.loadFromFile("assets/font.TTF")) has_font = true;
        auto load_piece = [&](int id, std::string filename) {
            sf::Texture tex;
            if (tex.loadFromFile("assets/" + filename)) {
                tex.setSmooth(true); textures[id] = tex;
            }
        };
        load_piece(1, "Chess_plt45.png"); load_piece(2, "Chess_nlt45.png");
        load_piece(3, "Chess_blt45.png"); load_piece(4, "Chess_rlt45.png");
        load_piece(5, "Chess_qlt45.png"); load_piece(6, "Chess_klt45.png");
        load_piece(-1, "Chess_pdt45.png"); load_piece(-2, "Chess_ndt45.png");
        load_piece(-3, "Chess_bdt45.png"); load_piece(-4, "Chess_rdt45.png");
        load_piece(-5, "Chess_qdt45.png"); load_piece(-6, "Chess_kdt45.png");
    }
};

Square get_square_at(int mouse_x, int mouse_y, bool flipped) {
    int x = mouse_x - OFFSET_X;
    int y = mouse_y - OFFSET_Y;
    if (x < 0 || x >= BOARD_PIXEL_SIZE || y < 0 || y >= BOARD_PIXEL_SIZE) return Square::None;
    int col = x / TILE_SIZE; int row = y / TILE_SIZE;
    int file = flipped ? (7 - col) : col;
    int rank = flipped ? row : (7 - row);
    return static_cast<Square>(rank * 8 + file);
}

int get_piece_at(const BoardState& b, Square sq) {
    for (int i = 0; i < 12; ++i) if (BitUtil::get_bit(b.pieces[i], sq)) return (i < 6) ? (i + 1) : -(i - 5);
    return 0; 
}

namespace GUI {
    void Launch(Search::EvalCallback evalFunc, int depth) {
        sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Chess Engine");
        window.setFramerateLimit(30);
        ImGui::SFML::Init(window);

        Attacks::init(); 
        BoardState board;
        // Init Start Pos
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
        board.to_move = Colour::White; board.castle_rights = 0b1111; board.full_move_number = 1;

        Assets assets; assets.load();
        Square selected_sq = Square::None;
        std::vector<Move> valid_moves;
        sf::Clock deltaClock;

        bool is_promoting = false; Square promo_from = Square::None; Square promo_to = Square::None;
        bool view_flipped = false;
        Colour human_side = Colour::White;
        Search::SearchStats last_stats;

        bool game_over = false;
        std::string winner_text = "";

        auto check_game_over = [&](BoardState& b) {
            std::vector<Move> moves;
            MoveGen::generate_moves(b, moves);
            
            bool has_legal_move = false;
            for(const auto& m : moves) {
                b.make_move(m);
                Colour us = (b.to_move == Colour::White) ? Colour::Black : Colour::White;
                Square k = Search::find_king(b, us);
                if(!Attacks::is_square_attacked(k, b.to_move, b.pieces.data(), b.occupancy[2])) {
                    has_legal_move = true;
                }
                b.undo_move(m);
                if(has_legal_move) break;
            }

            if (!has_legal_move) {
                game_over = true;
                Colour us = b.to_move;
                Square k = Search::find_king(b, us);
                Colour them = (us == Colour::White) ? Colour::Black : Colour::White;
                bool in_check = Attacks::is_square_attacked(k, them, b.pieces.data(), b.occupancy[2]);
                
                if (in_check) {
                    winner_text = (us == Colour::White) ? "Black Wins!" : "White Wins!";
                } else {
                    winner_text = "Stalemate";
                }
            }
        };

        auto render_board = [&]() {
            window.clear(sf::Color(30, 30, 30));
            sf::RectangleShape tile(sf::Vector2f(TILE_SIZE, TILE_SIZE));
            for (int r = 0; r < 8; ++r) {
                for (int f = 0; f < 8; ++f) {
                    bool is_light = ((r + f) % 2 != 0);
                    tile.setFillColor(is_light ? sf::Color(240, 217, 181) : sf::Color(181, 136, 99));
                    float x = view_flipped ? OFFSET_X + (7 - f) * TILE_SIZE : OFFSET_X + f * TILE_SIZE;
                    float y = view_flipped ? OFFSET_Y + r * TILE_SIZE : OFFSET_Y + (7 - r) * TILE_SIZE;
                    tile.setPosition(x, y); window.draw(tile);
                }
            }
            auto draw_hl = [&](Square sq, sf::Color c) {
                int r = static_cast<int>(sq) / 8; int f = static_cast<int>(sq) % 8;
                float x = view_flipped ? OFFSET_X + (7 - f) * TILE_SIZE : OFFSET_X + f * TILE_SIZE;
                float y = view_flipped ? OFFSET_Y + r * TILE_SIZE : OFFSET_Y + (7 - r) * TILE_SIZE;
                tile.setPosition(x, y); tile.setFillColor(c); window.draw(tile);
            };
            if (!is_promoting && selected_sq != Square::None) {
                draw_hl(selected_sq, sf::Color(255, 255, 0, 100));
                for (const auto& m : valid_moves) draw_hl(m.to(), sf::Color(100, 255, 100, 100));
            }
            for (int i = 0; i < 64; ++i) {
                if (is_promoting && static_cast<Square>(i) == promo_from) continue;
                int p = get_piece_at(board, static_cast<Square>(i));
                if (p != 0) {
                    int r = i / 8; int f = i % 8;
                    float x = view_flipped ? OFFSET_X + (7 - f) * TILE_SIZE + TILE_SIZE/2.0f : OFFSET_X + f * TILE_SIZE + TILE_SIZE/2.0f;
                    float y = view_flipped ? OFFSET_Y + r * TILE_SIZE + TILE_SIZE/2.0f : OFFSET_Y + (7 - r) * TILE_SIZE + TILE_SIZE/2.0f;
                    if (assets.textures.count(p)) {
                        sf::Sprite s(assets.textures[p]);
                        float sc = (TILE_SIZE * 0.85f) / s.getLocalBounds().width;
                        s.setScale(sc, sc); s.setOrigin(s.getLocalBounds().width/2, s.getLocalBounds().height/2);
                        s.setPosition(x, y); window.draw(s);
                    }
                }
            }
            if (is_promoting) {
                sf::RectangleShape ov(sf::Vector2f(WIN_WIDTH, WIN_HEIGHT)); ov.setFillColor(sf::Color(0,0,0,150)); window.draw(ov);
                float cx = WIN_WIDTH/2.0f, cy = WIN_HEIGHT/2.0f;
                int ids[] = {5,4,3,2}; if(board.to_move == Colour::Black) { ids[0]=-5; ids[1]=-4; ids[2]=-3; ids[3]=-2; }
                for(int i=0; i<4; ++i) {
                    if(assets.textures.count(ids[i])) {
                        sf::Sprite s(assets.textures[ids[i]]);
                        float sc = (TILE_SIZE*1.2f)/s.getLocalBounds().width;
                        s.setScale(sc, sc); s.setOrigin(s.getLocalBounds().width/2, s.getLocalBounds().height/2);
                        s.setPosition(cx + (i-1.5f)*(TILE_SIZE*1.5f), cy); window.draw(s);
                    }
                }
            }
        };

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                ImGui::SFML::ProcessEvent(window, event);
                if (event.type == sf::Event::Closed) window.close();

                // HUMAN INPUT (Check Side & Game Over)
                if (!game_over && board.to_move == human_side) {
                    if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                        if (ImGui::GetIO().WantCaptureMouse) continue;
                        
                        if (is_promoting) {
                            // Promotion Click Logic
                            float cx = WIN_WIDTH/2.0f; float cy = WIN_HEIGHT/2.0f; float btn = TILE_SIZE*1.5f;
                            int idx = -1;
                            for(int i=0; i<4; ++i) {
                                float bx = cx + (i-1.5f)*btn;
                                if(event.mouseButton.x > bx-btn/2 && event.mouseButton.x < bx+btn/2 && 
                                   event.mouseButton.y > cy-btn/2 && event.mouseButton.y < cy+btn/2) { idx = i; break; }
                            }
                            if(idx != -1) {
                                for(const auto& m : valid_moves) {
                                    if(m.to() == promo_to && m.from() == promo_from && m.is_promotion()) {
                                        bool match = false;
                                        if(idx==0 && m.is_promo_queen()) match=true;
                                        if(idx==1 && m.is_promo_rook()) match=true;
                                        if(idx==2 && m.is_promo_bishop()) match=true;
                                        if(idx==3 && m.is_promo_knight()) match=true;
                                        if(match) { 
                                            board.make_move(m); 
                                            is_promoting=false; 
                                            selected_sq=Square::None; 
                                            valid_moves.clear(); 
                                            
                                            check_game_over(board);
                                            break; 
                                        }
                                    }
                                }
                            }
                        } else {
                            Square clicked = get_square_at(event.mouseButton.x, event.mouseButton.y, view_flipped);
                            if (clicked != Square::None) {
                                bool moved = false;
                                if (selected_sq != Square::None) {
                                    for (const auto& m : valid_moves) {
                                        if (m.to() == clicked) {
                                            if (m.is_promotion()) { is_promoting=true; promo_from=m.from(); promo_to=m.to(); moved=true; break; }
                                            
                                            board.make_move(m); 
                                            selected_sq = Square::None; valid_moves.clear(); moved = true;
                                            
                                            check_game_over(board);
                                            break;
                                        }
                                    }
                                }
                                if (!moved && !is_promoting) {
                                    // Selection Logic...
                                    int p = get_piece_at(board, clicked);
                                    if (p != 0 && ((board.to_move == Colour::White) == (p > 0))) {
                                        selected_sq = clicked; valid_moves.clear();
                                        std::vector<Move> all; MoveGen::generate_moves(board, all);
                                        for(const auto& m : all) {
                                            if(m.from() == selected_sq) {
                                                board.make_move(m);
                                                Colour us = (board.to_move == Colour::White) ? Colour::Black : Colour::White;
                                                Square k = Search::find_king(board, us);
                                                bool ill = Attacks::is_square_attacked(k, board.to_move, board.pieces.data(), board.occupancy[2]);
                                                board.undo_move(m);
                                                if(!ill) valid_moves.push_back(m);
                                            }
                                        }
                                    } else { selected_sq = Square::None; valid_moves.clear(); }
                                }
                            }
                        }
                    }
                }
            }

            // BOT LOGIC
            if (!game_over && board.to_move != human_side && !is_promoting) {
                render_board(); window.display();
                Search::SearchParams params; params.depth = depth; params.evalFunc = evalFunc;
                Move botMove = Search::iterative_deepening(board, params, last_stats);
                
                if(botMove.raw() != 0) {
                    board.make_move(botMove);
                    check_game_over(board);
                } else {
                    check_game_over(board);
                }
                
                selected_sq = Square::None; valid_moves.clear();
            }

            ImGui::SFML::Update(window, deltaClock.restart());
            render_board();

            // SIDEBAR
            ImGui::SetNextWindowPos(sf::Vector2f(WIN_WIDTH - PANEL_WIDTH, 0));
            ImGui::SetNextWindowSize(sf::Vector2f(PANEL_WIDTH, WIN_HEIGHT));
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration);
            
            // GAME OVER STATUS
            if (game_over) {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "GAME OVER");
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "%s", winner_text.c_str());
                ImGui::Separator();
            }

            ImGui::TextColored(ImVec4(1,1,0,1), "GAME STATUS");
            ImGui::Separator();
            ImGui::Text("Turn: %s", (board.to_move == Colour::White ? "White" : "Black"));
            ImGui::Text("Move #: %d", board.full_move_number);

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0,1,1,1), "SETTINGS");
            ImGui::Separator();
            if (ImGui::RadioButton("Play as White", human_side == Colour::White)) { human_side = Colour::White; view_flipped = false; }
            if (ImGui::RadioButton("Play as Black", human_side == Colour::Black)) { human_side = Colour::Black; view_flipped = true; }
            ImGui::Checkbox("Flip Board", &view_flipped);

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0,1,0,1), "ENGINE STATS");
            ImGui::Separator();
            if (last_stats.depth_reached > 0) {
                ImGui::Text("Depth: %d", last_stats.depth_reached);
                float sc = last_stats.score / 100.0f;
                if(last_stats.score > 90000) ImGui::Text("Eval: Mate (Win)");
                else if(last_stats.score < -90000) ImGui::Text("Eval: Mate (Loss)");
                else ImGui::Text("Eval: %.2f", sc);
            } else { ImGui::Text("Waiting..."); }

            ImGui::Spacing(); ImGui::Separator();
            if (ImGui::Button("Reset Game", ImVec2(100, 30))) {
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
                board.to_move = Colour::White; board.castle_rights = 0b1111; board.full_move_number = 1;
                is_promoting = false; last_stats = Search::SearchStats();
                game_over = false; winner_text = "";
            }
            ImGui::End();
            ImGui::SFML::Render(window);
            window.display();
        }
        ImGui::SFML::Shutdown();
    }
}
