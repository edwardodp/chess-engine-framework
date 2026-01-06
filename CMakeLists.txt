cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
set(APP_SOURCES src/main.cpp)

add_executable(ChessApp ${APP_SOURCES} ${CORE_SOURCES})

add_library(ChessCore_Lib STATIC ${CORE_SOURCES})
target_include_directories(ChessCore_Lib PUBLIC include)

target_link_libraries(ChessApp PRIVATE ChessCore_Lib)
