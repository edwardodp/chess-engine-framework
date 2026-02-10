#include "Interface.hpp"
#include "imgui.h"
#include "imgui-SFML.h"
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <atomic>
#include <iostream>

#include "BoardState.hpp"
#include "MoveGen.hpp"
#include "Search.hpp"
#include "Attacks.hpp"
#include "BitUtil.hpp"
#include "Zobrist.hpp"

extern std::atomic<int> g_current_searcher;

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
    void Launch(Search::EvalCallback evalFunc, int depth, int human_side_int, std::string start_fen) {
        sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Chess Engine");
        window.setFramerateLimit(60);
        ImGui::SFML::Init(window);

        Attacks::init(); 
        Zobrist::init();
        BoardState board;
        
        // --- LOAD FEN OR DEFAULT ---
        if (start_fen.empty() || start_fen == "startpos") {
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
            board.refresh_hash();
        } else {
            board.load_fen(start_fen);
        }

        Assets assets; assets.load();
        Square selected_sq = Square::None;
        std::vector<Move> valid_moves;
        sf::Clock deltaClock;

        bool is_promoting = false; Square promo_from = Square::None; Square promo_to = Square::None;
        bool view_flipped = false;
        
        Colour human_side = (human_side_int == 0) ? Colour::White : Colour::Black;
        bool bot_vs_bot = (human_side_int == 2);

        Search::SearchStats last_stats;

        bool game_over = false;
        std::string winner_text = "";

        // --- THREADING STATE ---
        std::thread bot_thread;
        std::atomic<bool> is_thinking(false); // Flag to check if thread is busy
        Move bot_move_result;                 // Where the thread stores the best move
        Search::SearchStats bot_stats_result; // Where the thread stores stats

        auto check_game_over = [&](BoardState& b) {
            if (b.is_draw()) {
                game_over = true;
                winner_text = "Draw (Repetition/50MR)";
                return;
            }

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
            // Draw Background
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
            // Draw Highlights
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
            // Draw Pieces
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
            // Draw Promotion UI
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
                if (event.type == sf::Event::Closed) {
                    // Safety: If thread is running, detach it before closing to avoid abort()
                    if (is_thinking) bot_thread.detach(); 
                    window.close();
                }

                // HUMAN INPUT (Only if not thinking & not game over)
                bool is_human_turn = !(bot_vs_bot && board.to_move == human_side);

                if (is_human_turn && !game_over && !is_thinking) {
                    if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                        if (ImGui::GetIO().WantCaptureMouse) continue;
                        
                        // ... (Human click logic) ...
                        if (is_promoting) {
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
                                            board.make_move(m); is_promoting=false; selected_sq=Square::None; valid_moves.clear(); 
                                            check_game_over(board); break; 
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
                                            board.make_move(m); selected_sq = Square::None; valid_moves.clear(); moved = true;
                                            check_game_over(board); break;
                                        }
                                    }
                                }
                                if (!moved && !is_promoting) {
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

            ImGui::SFML::Update(window, deltaClock.restart());

            // --- SIDEBAR UI ---
            ImGui::SetNextWindowPos(sf::Vector2f(WIN_WIDTH - PANEL_WIDTH, 0));
            ImGui::SetNextWindowSize(sf::Vector2f(PANEL_WIDTH, WIN_HEIGHT));
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration);
            
            if (game_over) {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "GAME OVER");
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "%s", winner_text.c_str());
                ImGui::Separator();
            }

            ImGui::TextColored(ImVec4(1,1,0,1), "GAME STATUS");
            ImGui::Separator();
            ImGui::Text("Turn: %s", (board.to_move == Colour::White ? "White" : "Black"));
            ImGui::Text("Move #: %d", board.full_move_number);
            ImGui::Text("Mode: %s", bot_vs_bot ? "Bot vs Bot" : "Human vs Bot");

            if (is_thinking) {
                ImGui::TextColored(ImVec4(0,1,1,1), "Status: THINKING...");
            } else {
                ImGui::Text("Status: Waiting");
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0,1,1,1), "SETTINGS");
            ImGui::Separator();
            ImGui::Checkbox("Flip Board", &view_flipped); 

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0,1,0,1), "ENGINE STATS");
            ImGui::Separator();
            
            if (last_stats.depth_reached > 0) {
                std::string who_searched = (board.to_move == Colour::White) ? "Black (Prev Move)" : "White (Prev Move)";
                ImGui::Text("Eval Source: %s", who_searched.c_str());
                ImGui::Text("Depth: %d", last_stats.depth_reached);
                float sc = last_stats.score / 100.0f;
                if(last_stats.score > 90000) ImGui::Text("Score: Mate (Win)");
                else if(last_stats.score < -90000) ImGui::Text("Score: Mate (Loss)");
                else ImGui::Text("Score: %.2f", sc);
            }

            ImGui::Spacing(); ImGui::Separator();
            
            // Only allow Reset if bot isn't busy (to prevent threading crashes)
            if (!is_thinking) {
                if (ImGui::Button("Reset Game", ImVec2(100, 30))) {
                    board = BoardState(); 
                    if (start_fen.empty() || start_fen == "startpos") {
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
                        board.refresh_hash();
                    } else {
                        board.load_fen(start_fen);
                    }
                    is_promoting = false; last_stats = Search::SearchStats();
                    game_over = false; winner_text = "";
                }
            } else {
                ImGui::BeginDisabled();
                ImGui::Button("Reset (Busy)", ImVec2(100, 30));
                ImGui::EndDisabled();
            }

            ImGui::End();

            // 3. BOT LOGIC (NON-BLOCKING)
            bool is_bot_turn = (bot_vs_bot || board.to_move != human_side);

            if (!game_over && is_bot_turn && !is_promoting && !is_thinking && !bot_thread.joinable()) {
                is_thinking = true;
                
                // Tell the dispatcher which bot's eval function to use for this search
                g_current_searcher.store(
                    (board.to_move == Colour::White) ? 0 : 1,
                    std::memory_order_relaxed
                );

                BoardState board_copy = board; 
                Search::SearchParams params;
                params.depth = depth;
                params.evalFunc = evalFunc;
                
                bot_thread = std::thread([board_copy, params, &bot_move_result, &bot_stats_result, &is_thinking]() mutable {
                    bot_move_result = Search::iterative_deepening(board_copy, params, bot_stats_result);
                    is_thinking = false;
                });
            }

            if (!is_thinking && bot_thread.joinable()) {
                bot_thread.join();
                
                if (bot_move_result.raw() != 0) {
                    board.make_move(bot_move_result);
                    check_game_over(board);
                    last_stats = bot_stats_result;
                } else {
                    check_game_over(board);
                }
            }

            // 4. RENDER (Runs every frame, regardless of bot status)
            window.clear(sf::Color(30, 30, 30));
            render_board();
            ImGui::SFML::Render(window);
            window.display();
        }
        
        if (bot_thread.joinable()) bot_thread.join();
        ImGui::SFML::Shutdown();
    }
}
