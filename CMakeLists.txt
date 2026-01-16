cmake_minimum_required(VERSION 3.15)
project(ChessEngine LANGUAGES CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SFML_STATIC_LIBRARIES ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML 2.6 COMPONENTS graphics window system CONFIG QUIET)

if(SFML_FOUND)
    message(STATUS "Building with VCPKG SFML")
    set(IMGUI_SFML_FIND_SFML ON CACHE BOOL "" FORCE)
else()
    message(STATUS "Building with FetchContent SFML")
    include(FetchContent)

    # A. SFML
    FetchContent_Declare(sfml GIT_REPOSITORY https://github.com/SFML/SFML.git GIT_TAG 2.6.1)
    set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
    set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
    
    set(SFML_OSX_USE_FRAMEWORKS OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(sfml)
    
    set(IMGUI_SFML_FIND_SFML OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.89.9)
FetchContent_MakeAvailable(imgui)

set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "Path to ImGui" FORCE)
FetchContent_Declare(imgui-sfml GIT_REPOSITORY https://github.com/SFML/imgui-sfml.git GIT_TAG v2.6)
FetchContent_MakeAvailable(imgui-sfml)

file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp" "src/gui/*.cpp")
set(MAIN_API_SOURCE src/Main.cpp)

add_library(ChessLib SHARED ${CORE_SOURCES} ${MAIN_API_SOURCE})
target_include_directories(ChessLib PUBLIC include)
target_compile_definitions(ChessLib PRIVATE BUILD_AS_LIBRARY)

if(SFML_FOUND)
    target_link_libraries(ChessLib PRIVATE SFML::Graphics SFML::Window SFML::System ImGui-SFML::ImGui-SFML)
else()
    target_link_libraries(ChessLib PRIVATE sfml-graphics sfml-system sfml-window ImGui-SFML::ImGui-SFML)
endif()

add_executable(ChessExe ${MAIN_API_SOURCE} ${CORE_SOURCES})
target_include_directories(ChessExe PUBLIC include)

if(SFML_FOUND)
    target_link_libraries(ChessExe PRIVATE SFML::Graphics SFML::Window SFML::System ImGui-SFML::ImGui-SFML)
else()
    target_link_libraries(ChessExe PRIVATE sfml-graphics sfml-system sfml-window ImGui-SFML::ImGui-SFML)
endif()

foreach(TARGET ChessLib ChessExe)
    if(MSVC)
        target_compile_options(${TARGET} PUBLIC /constexpr:steps 1000000 /O2)
    else()
        target_compile_options(${TARGET} PUBLIC -O3 -march=native)
    endif()
endforeach()
