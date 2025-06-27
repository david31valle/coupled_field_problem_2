#pragma once

#include "../Eigen/Dense"
#include <utility>

std::tuple<Eigen::MatrixXd,  std::vector<Eigen::MatrixXd> >
generate_mesh(int PD, double domain_size,
               int partition, const std::vector<int>& element_orders,
               int problem_dimension);
