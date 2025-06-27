#pragma once
#include "../Eigen/Dense"
#include <array>

class Node {
public:
    int PD;                    // Problem dimension
    int Nr;                    // Node number

    Eigen::VectorXd X;         // Material position
    Eigen::VectorXd x;         // Spatial position
    Eigen::VectorXd U;         // Unknowns (material config)
    Eigen::VectorXd u;         // Unknowns (spatial config)
    Eigen::VectorXd un;        // Previous unknowns (spatial config)

    Eigen::MatrixXd EIL_1;     // Element list 1
    Eigen::MatrixXd EIL_2;     // Element list 2

    Eigen::VectorXd BC;        // Boundary condition vector
    Eigen::VectorXd DOF;       // Degrees of freedom vector

    Eigen::VectorXd field;     // Indicates which fields the node carries (2-element vector)

    Eigen::VectorXd GP_BC;     // Gauss point boundary conditions
    Eigen::VectorXd GP_DOF;    // Gauss point DOFs
    Eigen::VectorXd GP_vals;   // Gauss point values

    // Constructor
    Node(int Nr, int PD, const Eigen::VectorXd& X_input,
         const Eigen::VectorXd& C, const Eigen::VectorXd& V,
         const Eigen::MatrixXd& EIL1, const Eigen::MatrixXd& EIL2,
         const Eigen::VectorXd& field_input, const Eigen::Vector2i& field_dim);
};
