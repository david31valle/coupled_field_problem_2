#pragma once

#include "../Eigen/Dense"
#include <utility>

// returns:  first  = nodal coordinates  (MatrixXd: N×dim, dim=1,2 or 3)
//           second = element connectivity (MatrixXd: #elements × nodes-per-element)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
generate_mesh(int PD,
              double domain_size,
              int partition,
              int element_order,
              int problem_dimension);
