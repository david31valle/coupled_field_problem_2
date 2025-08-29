
#ifndef PROGRAMMING_PROJECT_INITIALIZE_HPP
#define PROGRAMMING_PROJECT_INITIALIZE_HPP


#include <iostream>
#include "../Eigen/Dense"
#include "../node/node.hpp"
#include "../element/element.hpp"


std::pair<std::vector<Node>, std::vector<element>>
Initialize(int PD,
           const Eigen::MatrixXd &nl,
           const Eigen::MatrixXd &el_1,
           const Eigen::MatrixXd &el_2,
           double domain_size,
           double C_initial,
           double C_perturb,
           const Eigen::VectorXd &Density,
           const Eigen::VectorXd &Velocity,
           const std::string &Initial_density,
           const std::vector<int> &element_order,
           const Eigen::Vector2i &field_dim,
           const std::vector<double> &parameters);

static std::vector<int> find_linear_indices_eq(const Eigen::MatrixXd& M, int value);



#endif //PROGRAMMING_PROJECT_INITIALIZE_HPP
