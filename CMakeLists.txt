cmake_minimum_required(VERSION 3.26)
project(coupled_field_problem_2)

set(CMAKE_CXX_STANDARD 17)

add_executable(coupled_field_problem_2
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

)
