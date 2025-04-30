

#include "node.hpp"

node::node(int node_number, int problem_dimension,  Eigen::VectorXd X_material_position,
           Eigen::VectorXd x_spatial_position, std::vector<int> element_list) {
    this->node_number = node_number;
    this->problem_dimension = problem_dimension;

    // Resize member vectors and assign passed values
    this->X_material_position.resize(problem_dimension, problem_dimension);
    this->X_material_position = X_material_position;

    this->x_spatial_position.resize(problem_dimension, problem_dimension);
    this->x_spatial_position = x_spatial_position;

    this->element_list=element_list;

    initialization(problem_dimension);

}

void node::initialization(int PD) {
    boundary_condition.resize(PD);
    boundary_condition.setOnes();

    boundary_condition_value.resize(PD);
    boundary_condition_value.setZero();

    DOF.resize(PD);
    DOF.setZero();

    degree_constrained.resize(PD);
    degree_constrained.setZero();

    global_index.resize(PD);
    global_index.setZero();

    size_t size = PD * PD + PD * PD;

    gauss_point_BC.resize(size);
    gauss_point_BC.setOnes();

    gauss_point_DOF.resize(size);
    gauss_point_DOF.setZero();

    gauss_point_values.resize(size);
    gauss_point_values.setZero();
}

void node::printNodeData() const {
    std::cout << "Node Number: " << node_number << std::endl;
    std::cout << "Problem Dimension: " << problem_dimension << std::endl;

    std::cout << "Material Position (X):" << std::endl;
    std::cout << X_material_position << std::endl;

    std::cout << "Spatial Position (x):" << std::endl;
    std::cout << x_spatial_position << std::endl;

    std::cout << "Element List: ";
    for (const auto& elem : element_list) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    std::cout << "Boundary Conditions:" << std::endl;
    std::cout << boundary_condition.transpose() << std::endl;

    std::cout << "Boundary Condition Values:" << std::endl;
    std::cout << boundary_condition_value.transpose() << std::endl;

    std::cout << "Degrees of Freedom (DOF):" << std::endl;
    std::cout << DOF.transpose() << std::endl;

    std::cout << "Degree Constrained:" << std::endl;
    std::cout << degree_constrained.transpose() << std::endl;

    std::cout << "Global Indices:" << std::endl;
    std::cout << global_index.transpose() << std::endl;

    std::cout << "Element List: ";
    for (const auto& elem : element_list) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}


