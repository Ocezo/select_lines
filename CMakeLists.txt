cmake_minimum_required(VERSION 3.16)

project(select_lines VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
