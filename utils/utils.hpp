//
// Created by David Valle on 28-Jan-25.
//

#ifndef PROGRAMMING_PROJECT_UTILS_HPP
#define PROGRAMMING_PROJECT_UTILS_HPP
#include "../Eigen/Dense"
#include <iostream>
#include "string"

void printMatrix(const std::vector<std::vector<double>>& matrix, const std::string&);
std::vector<std::vector<double>> compute_gp(int NGP, int PD);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> compute_N_xi_gp(int degree, const std::vector<std::vector<double>>& GP, int PD);


#endif //PROGRAMMING_PROJECT_UTILS_HPP
