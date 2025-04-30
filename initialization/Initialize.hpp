
#ifndef PROGRAMMING_PROJECT_INITIALIZE_HPP
#define PROGRAMMING_PROJECT_INITIALIZE_HPP


#include <iostream>
#include "../Eigen/Dense"
#include "../node/node.hpp"
#include "../element/element.hpp"


std::pair<std::vector<node>, std::vector<element>> initialize(int problem_dimension,
                                                      Eigen::MatrixXd& node_list,
                                                      Eigen::MatrixXd& element_list,
                                                      int domain_size,
                                                      int element_order,
                                                      double lamda,
                                                      double mu
);



#endif //PROGRAMMING_PROJECT_INITIALIZE_HPP
