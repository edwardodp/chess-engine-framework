#include "Interface.hpp"
#include "Evaluation.hpp"
#include <iostream>

using EvalCallback = int32_t(*)(const uint64_t*, const uint64_t*, uint32_t);

extern "C" {
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    // When Python calls this, it opens the Window!
    void startEngine(EvalCallback evalFunc, int depth) {
        std::cout << "Starting GUI from Shared Library..." << std::endl;
        GUI::Launch(evalFunc, depth);
    }
}

#ifndef BUILD_AS_LIBRARY
int main() {
    std::cout << "Starting GUI from Standalone Executable..." << std::endl;
    GUI::Launch(Evaluation::evaluate, 5);
    return 0;
}
#endif
