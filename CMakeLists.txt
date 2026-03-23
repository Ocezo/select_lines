cmake_minimum_required(VERSION 3.16)

project(select_lines VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SELECT_LINES_ENABLE_PROFILING "Enable profiler-friendly flags" OFF)
option(SELECT_LINES_ENABLE_GPROF "Enable gprof instrumentation" OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(select_lines
    src/main.cpp
)

target_link_libraries(select_lines
    PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

if (SELECT_LINES_ENABLE_PROFILING)
    target_compile_options(select_lines PRIVATE -g -fno-omit-frame-pointer)
endif()

if (SELECT_LINES_ENABLE_GPROF)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(select_lines PRIVATE -pg)
        target_link_options(select_lines PRIVATE -pg)
    else()
        message(FATAL_ERROR "SELECT_LINES_ENABLE_GPROF requires GCC or Clang")
    endif()
endif()
