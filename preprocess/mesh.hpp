
#ifndef PROGRAMMING_PROJECT_MESH_HPP
#define PROGRAMMING_PROJECT_MESH_HPP

#include "mesh_1D.hpp"
#include "mesh_2D.hpp"
#include "mesh_3D.hpp"
#include <iostream>

std::pair<Eigen::MatrixXd , Eigen::MatrixXd> generate_mesh(int domain_size, int partition, int element_order, int problem_dimension);


#endif //PROGRAMMING_PROJECT_MESH_HPP
