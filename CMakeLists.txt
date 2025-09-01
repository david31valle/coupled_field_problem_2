cmake_minimum_required(VERSION 3.26)
project(coupled_field_problem_2)

set(CMAKE_CXX_STANDARD 17)

set(SRCS
        main.cpp
        Eigen
        data.txt
        initialization
        preprocess
        node/node.hpp
        preprocess/mesh.cpp
        preprocess/mesh.hpp
        preprocess/mesh_1D.cpp
        preprocess/mesh_1D.hpp
        preprocess/mesh_2D.cpp
        preprocess/mesh_2D.hpp
        preprocess/mesh_3D.cpp
        preprocess/mesh_3D.hpp
        node/node.cpp
        element/element.cpp
        utils/utils.cpp
        initialization/Initialize.hpp
        initialization/Initialize.cpp
        problem/problem.hpp
        problem/problem.cpp
        problem/problem.cpp
        postprocess/postprocess.hpp
        postprocess/postprocess.cpp
)

add_executable(coupled_field_problem_2 ${SRCS})
target_include_directories(coupled_field_problem_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(${CMAKE_SOURCE_DIR}/config.ini
        ${CMAKE_CURRENT_BINARY_DIR}/config.ini COPYONLY)

# (Optional build time wins)
set(CMAKE_UNITY_BUILD ON)                           # big speedup on GCC/Clang

# Extra optimization flags for GCC/Clang
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")