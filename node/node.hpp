
// example of usage: Eigen::VectorXd X_material_position(problem_dimension);
//    Eigen::VectorXd x_spatial_position(problem_dimension);
//    Eigen::VectorXd element_list(5); // Assuming 5 elements are associated
//
//    X_material_position(0)= 1.0;  // Example material position
//    x_spatial_position(0)=1.1;  // Example spatial position
//    element_list(0)=0;        // Example element list
//
//    // Create an instance of the `node` class
//    node myNode(2, problem_dimension, X_material_position, x_spatial_position, element_list);
//

#ifndef PROGRAMMING_PROJECT_NODE_HPP
#define PROGRAMMING_PROJECT_NODE_HPP
#include "../Eigen/Dense"
#include <iostream>
#include <vector>

class node {
public:
    //Constructor
    node(int node_number, int problem_dimension, Eigen::VectorXd X_material_position, Eigen::VectorXd x_spatial_configuration, std::vector<int> element_list);

    int problem_dimension;
    int node_number;
    Eigen::MatrixXd X_material_position;
    Eigen::MatrixXd x_spatial_position;
    std::vector<int> element_list;
    Eigen::VectorXd boundary_condition;
    Eigen::VectorXd boundary_condition_value;
    Eigen::VectorXd DOF;
    Eigen::VectorXd degree_constrained;
    Eigen::VectorXd global_index;


    Eigen::VectorXd gauss_point_BC;
    Eigen::VectorXd gauss_point_DOF;
    Eigen::VectorXd gauss_point_values;
    void printNodeData() const;

private:
    void initialization(int PD);

};


#endif //PROGRAMMING_PROJECT_NODE_HPP
