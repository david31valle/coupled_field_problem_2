
#include "Initialize.hpp"


std::pair<std::vector<node>, std::vector<element>>
initialize(int problem_dimension, Eigen::MatrixXd &node_list, Eigen::MatrixXd &element_list, int domain_size,
           int element_order, double lamda, double mu) {
    double tol = 1e-6;
    int NGP = 0;
    int number_of_nodes=node_list.rows();
    int number_of_elements=element_list.rows();
    int nodes_per_elements=element_list.cols();

    switch (problem_dimension) {
        case 1:
            switch (element_order) {
                case 1: NGP = 2; break;
                case 2: NGP = 3; break;
                case 3: NGP = 4; break;
                case 4: NGP = 5; break;
                default: std::cerr << "Invalid element_order for PD=1\n"; break;
            }
            break;
        case 2:
            switch (element_order) {
                case 1: NGP = 4; break;
                case 2: NGP = 9; break;
                case 3: NGP = 16; break;
                case 4: NGP = 25; break;
                default: std::cerr << "Invalid element_order for PD=2\n"; break;
            }
            break;
        case 3:
            switch (element_order) {
                case 1: NGP = 8; break;
                case 2: NGP = 27; break;
                case 3: NGP = 64; break;
                case 4: NGP = 125; break;
                default: std::cerr << "Invalid element_order for PD=3\n"; break;
            }
            break;
        default:
            std::cerr << "Invalid PD value\n";
            break;
    }

    int rows = element_list.rows();
    int cols = element_list.cols();

    std::vector<node> Node_List;

    for (int i = 0; i < number_of_nodes; ++i) {
        // Find all occurrences of 'i+1' in 'element_list'
        std::vector<int> element_indices;
        for (int row = 0; row < rows ; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (element_list(row, col) == i+1) {
                    element_indices.push_back(col*rows+row);  // Store element index
                }
            }
        }
        Eigen::VectorXd X = node_list.row(i);
        Node_List.push_back(node(i, problem_dimension, X, X, element_indices));
    }

    std::vector<element> Element_List;

    for (int i = 0; i < number_of_elements; ++i) {
        Eigen::MatrixXd X = Eigen::MatrixXd::Zero(problem_dimension, nodes_per_elements);  // Initialize (PD x NPE) matrix

        Eigen::MatrixXd NdL = element_list.row(i);  // Extract node indices for element i

        // Iterate over nodes in the element
        for (int j = 0; j < nodes_per_elements; ++j) {
            X.col(j) = Node_List[NdL(j)].X_material_position;  // Assign node coordinates (assuming NL[j].X is an Eigen::VectorXd)
        }

        Eigen::MatrixXd x = X;  // Equivalent to 'x = X' in MATLAB

        // Create and store the element
        element test=element(i, problem_dimension, NdL, X, x, NGP, element_order, lamda, mu);
        Element_List.push_back(element(i, problem_dimension, NdL, X, x, NGP, element_order, lamda, mu));
        ;
    }

    std::pair<std::vector<node>, std::vector<element>> node_and_element_list;
    node_and_element_list.first=Node_List;
    node_and_element_list.second=Element_List;
    return node_and_element_list;
}
