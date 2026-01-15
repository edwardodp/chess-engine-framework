cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Fetch GoogleTest ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: Prevent GTest from overriding our compiler flags
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- 2. Core Sources ---
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
set(APP_SOURCES src/Main.cpp)

# --- 3. Main Library (ChessLib) ---
add_library(ChessLib SHARED ${CORE_SOURCES} ${APP_SOURCES})
target_include_directories(ChessLib PUBLIC include)
target_compile_definitions(ChessLib PRIVATE BUILD_AS_LIBRARY)

if(APPLE)
    set_target_properties(ChessLib PROPERTIES SUFFIX ".so")
endif()

# --- 4. Test Executable (UnitTests) ---
# We create a new executable just for testing
add_executable(UnitTests tests/ChessTests.cpp ${CORE_SOURCES})
target_include_directories(UnitTests PUBLIC include)
target_link_libraries(UnitTests GTest::gtest_main)

# --- 5. Compiler Options ---
foreach(TARGET ChessLib UnitTests)
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

enable_testing()
add_test(NAME ChessEngineTests COMMAND UnitTests)
