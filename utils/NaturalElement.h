#pragma once
#include "../Eigen/Dense"



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> build(int NPE, int PD, int NGP, const Eigen::MatrixXd &GP);

