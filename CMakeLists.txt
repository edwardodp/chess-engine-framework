cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source Files
# Automatically picks up MoveGen.cpp, Search.cpp, Attacks.cpp
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
set(APP_SOURCES src/Main.cpp)

# --- 1. Python Shared Library (ChessLib) ---
add_library(ChessLib SHARED ${CORE_SOURCES} ${APP_SOURCES})
target_include_directories(ChessLib PUBLIC include)
target_compile_definitions(ChessLib PRIVATE BUILD_AS_LIBRARY)

if(APPLE)
    set_target_properties(ChessLib PROPERTIES SUFFIX ".so")
endif()

# --- 2. C++ Test Executable (ChessTest) ---
add_executable(ChessTest ${APP_SOURCES} ${CORE_SOURCES})
target_include_directories(ChessTest PUBLIC include)

# --- Compiler Options ---
foreach(TARGET ChessLib ChessTest)
    if(MSVC)
        target_compile_options(${TARGET} PUBLIC /constexpr:steps 1000000 /O2)
    else()
        target_compile_options(${TARGET} PUBLIC 
            -O3 -march=native
            $<$<CXX_COMPILER_ID:GNU>:-fconstexpr-ops-limit=100000000>
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fconstexpr-steps=100000000>
        )
    endif()
endforeach()
