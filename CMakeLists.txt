cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
set(APP_SOURCES src/Main.cpp)

add_library(ChessCore_Lib STATIC ${CORE_SOURCES})
target_include_directories(ChessCore_Lib PUBLIC include)

# Increase constexpr evaluation limits for all major compilers
if(MSVC)
    target_compile_options(ChessCore_Lib PUBLIC /constexpr:steps 1000000)
else()
    target_compile_options(ChessCore_Lib PUBLIC 
        $<$<CXX_COMPILER_ID:GNU>:-fconstexpr-ops-limit=100000000>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fconstexpr-steps=100000000>
    )
endif()

add_executable(ChessApp ${APP_SOURCES})
target_link_libraries(ChessApp PRIVATE ChessCore_Lib)
