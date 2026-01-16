cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(sfml GIT_REPOSITORY https://github.com/SFML/SFML.git GIT_TAG 2.6.1)
set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sfml)

FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.89.9)
FetchContent_MakeAvailable(imgui)

set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "Path to ImGui" FORCE)
set(IMGUI_SFML_FIND_SFML OFF CACHE BOOL "" FORCE)
FetchContent_Declare(imgui-sfml GIT_REPOSITORY https://github.com/SFML/imgui-sfml.git GIT_TAG v2.6)
FetchContent_MakeAvailable(imgui-sfml)

file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp" "src/gui/*.cpp")
set(MAIN_API_SOURCE src/Main.cpp)

add_library(ChessLib SHARED ${CORE_SOURCES} ${MAIN_API_SOURCE})
target_include_directories(ChessLib PUBLIC include)
target_compile_definitions(ChessLib PRIVATE BUILD_AS_LIBRARY)
target_link_libraries(ChessLib PRIVATE sfml-graphics sfml-system sfml-window ImGui-SFML::ImGui-SFML)

if(APPLE)
    set_target_properties(ChessLib PROPERTIES SUFFIX ".so")
endif()

add_executable(ChessExe ${MAIN_API_SOURCE} ${CORE_SOURCES})
target_include_directories(ChessExe PUBLIC include)
target_link_libraries(ChessExe PRIVATE sfml-graphics sfml-system sfml-window ImGui-SFML::ImGui-SFML)

foreach(TARGET ChessLib ChessExe)
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
