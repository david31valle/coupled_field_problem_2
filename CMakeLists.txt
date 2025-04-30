cmake_minimum_required(VERSION 3.26)
project(coupled_field_problem_2)

set(CMAKE_CXX_STANDARD 14)

add_executable(coupled_field_problem_2
        main.cpp
        Eigen
        data.txt
        test.txt
)
