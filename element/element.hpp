
#pragma once
#include <vector>
#include "../Eigen/Dense"
#include "../utils/utils.hpp"
#include <iostream>
#include <tuple>


// Extended element class for coupled scalar-vector field FEM problems
class element {
public:
    // === Element Info ===
    int Nr;         // Element number
    int PD;         // Problem dimension (2D/3D)

    Eigen::VectorXd NdL1;  // Node list for scalar field
    Eigen::VectorXd NdL2;  // Node list for vector field

    int NPE1;       // Nodes per element for scalar field
    int NPE2;       // Nodes per element for vector field
    int deg1;       // Polynomial degree for scalar field shape functions
    int deg2;       // Polynomial degree for vector field shape functions

    // === Geometry ===
    Eigen::MatrixXd X;    // Reference coordinates
    Eigen::MatrixXd x;    // Current configuration
    Eigen::MatrixXd xn;   // Previous configuration

    // === Scalar field (e.g., concentration) ===
    Eigen::VectorXd C;
    Eigen::VectorXd c;
    Eigen::VectorXd cn;

    // === Vector field (e.g., velocity) ===
    Eigen::MatrixXd V;
    Eigen::MatrixXd v;
    Eigen::MatrixXd vn;

    // === Gauss integration ===
    int NGP;
    Eigen::MatrixXd GP;

    // === Material or physical parameters ===
    std::vector<double> parameters;

    // === Constructor ===
    element(int Nr, int PD,
            const Eigen::VectorXd& NdL1,
            const Eigen::VectorXd& NdL2,
            const Eigen::MatrixXd& X,
            const Eigen::VectorXd& C,
            const Eigen::MatrixXd& V,
            int NGP,
            std::pair<int, int> element_order,
            const std::vector<double>& parameters);

    // === Residual and Tangent Assembly ===

    // Returns residual vectors for scalar and vector field
    std::pair<Eigen::VectorXd, Eigen::VectorXd> Residual(double dt);

    // Returns residuals and stiffness matrices for coupled system
    std::tuple<
            Eigen::VectorXd, Eigen::VectorXd,   // R1 (scalar), R2 (vector)
            Eigen::MatrixXd, Eigen::MatrixXd,   // K11, K12
            Eigen::MatrixXd, Eigen::MatrixXd    // K21, K22
    > RK(double dt);

    // Returns per-Gauss-point contributions for debugging or visualization
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> RK_GP(double dt, int NGP_val);

    // Print element info (for debugging)
    void printElementData() const;

private:
    // === Shape Functions and Jacobians (internal) ===
    Eigen::MatrixXd N1;      // Shape functions (scalar field)
    Eigen::MatrixXd N2;      // Shape functions (vector field)
    Eigen::MatrixXd GradN1;  // Gradients in spatial coords (scalar field)
    Eigen::MatrixXd GradN2;  // Gradients in spatial coords (vector field)
    Eigen::MatrixXd JJ;      // Jacobians per Gauss point

    // === Helper Methods ===
    Eigen::MatrixXd convertToEigenMatrix(const std::vector<std::vector<double>>& vec);

    // Computes Jacobian matrix at each Gauss point
    Eigen::MatrixXd compute_J(const Eigen::MatrixXd& X_e,
                              int NGP, int PD,
                              const std::vector<std::vector<double>>& GradN_xi_gp);

    // Computes physical-space gradients of shape functions
    Eigen::MatrixXd compute_GradN(const Eigen::MatrixXd& JJ,
                                  int NGP,
                                  int PD,
                                  const std::vector<std::vector<double>>& GradN_xi_gp);


    static Eigen::MatrixXd gauss_points;
    static Eigen::MatrixXd shape_functions_N;
    static Eigen::MatrixXd gradient_N_xi;
    static std::vector<std::vector<double>> gauss_points_vector;
    static std::vector<std::vector<double>> shape_functions_N_vector;
    static std::vector<std::vector<double>> gradient_N_xi_vector;

    void    compute_at_gp(const Eigen::VectorXd &c,
                  const Eigen::MatrixXd &v,
                  const Eigen::VectorXd &cn,
                  const Eigen::MatrixXd &vn,
                  const Eigen::MatrixXd &N1_gp,
                  const Eigen::MatrixXd &N2_gp,
                  const Eigen::MatrixXd &GradN1_gp,
                  const Eigen::MatrixXd &GradN2_gp,
                  Eigen::VectorXd &c_gp,
                  Eigen::MatrixXd &v_gp,
                  Eigen::VectorXd &cn_gp,
                  Eigen::MatrixXd &vn_gp,
                  Eigen::MatrixXd &Gradc_gp,
                  Eigen::MatrixXd &Gradv_gp,
                  Eigen::MatrixXd &Gradcn_gp,
                  Eigen::MatrixXd &Gradvn_gp);

    void assemble_tangent_matrices(double JxW,
                                   double c_val,
                                   const Eigen::VectorXd& v_val,
                                   const Eigen::VectorXd& N1_gp,
                                   const Eigen::VectorXd& N2_gp,
                                   const Eigen::MatrixXd& GradN1_gp,
                                   const Eigen::MatrixXd& GradN2_gp,
                                   const Eigen::MatrixXd& sig,
                                   const Eigen::MatrixXd& dsig_dc,
                                   Eigen::MatrixXd& K11,
                                   Eigen::MatrixXd& K12,
                                   Eigen::MatrixXd& K21,
                                   Eigen::MatrixXd& K22);



};