#include "Interface.hpp"
#include "Evaluation.hpp"
#include <iostream>
#include <string>
#include <cstdint>
#include <atomic>

static Search::EvalCallback global_white_eval = nullptr;
static Search::EvalCallback global_black_eval = nullptr;

std::atomic<int> g_current_searcher{0};

int cpp_dispatcher(const uint64_t* pieces, const uint64_t* occupancy, int side) {
    int searcher = g_current_searcher.load(std::memory_order_relaxed);
    Search::EvalCallback fn = (searcher == 0) ? global_white_eval : global_black_eval;
    if (fn) return fn(pieces, occupancy, side);
    return 0;
}

extern "C" {
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    
    void startEngine(Search::EvalCallback whiteFunc, Search::EvalCallback blackFunc, int depth, int human_side, const char* fen) {
        global_white_eval = whiteFunc;
        global_black_eval = blackFunc;

        std::string fen_str = (fen != nullptr) ? std::string(fen) : "startpos";
        
        GUI::Launch((Search::EvalCallback)cpp_dispatcher, depth, human_side, fen_str);
    }
}

#ifndef BUILD_AS_LIBRARY
int main() {
    std::cout << "Starting Standalone Engine (Human vs Bot)" << std::endl;
    GUI::Launch(Evaluation::evaluate, 5, 0, "startpos"); 
    return 0;
}
#endif
