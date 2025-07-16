#include "element.hpp"
#include "../utils/utils.hpp"

// Define static members
//Eigen::MatrixXd element::gauss_points;
//Eigen::MatrixXd element::shape_functions_N;
//Eigen::MatrixXd element::gradient_N_xi;
//std::vector<std::vector<double>> element::gauss_points_vector;
//std::vector<std::vector<double>> element::shape_functions_N_vector;
//std::vector<std::vector<double>> element::gradient_N_xi_vector;


#include "element.hpp"
#include "../utils/utils.hpp"  // compute_gp, compute_N_xi_gp, printMatrix

element::element(int Nr, int PD,
                 const Eigen::VectorXd& NdL1,
                 const Eigen::VectorXd& NdL2,
                 const Eigen::MatrixXd& X,
                 const Eigen::VectorXd& C,
                 const Eigen::MatrixXd& V,
                 int NGP,
                 std::pair<int, int> element_order,
                 const std::vector<double>& parameters)
        : Nr(Nr), PD(PD), NdL1(NdL1), NdL2(NdL2), X(X), C(C), V(V),
          NGP(NGP), parameters(parameters) {

    // Assign degrees and nodes per element
    this->deg1 = element_order.first;
    this->deg2 = element_order.second;
    this->NPE1 = NdL1.size();
    this->NPE2 = NdL2.size();

    // Initialize geometry
    this->x  = X;
    this->xn = X;

    // Scalar field (concentration)
    this->c  = C;
    this->cn = C;

    // Vector field (velocity)
    this->v  = V;
    this->vn = V;

    // === Gauss points ===
    std::vector<std::vector<double>> GP_vec = compute_gp(this->NGP, this->PD);
    this->GP = Eigen::MatrixXd(GP_vec.size(), GP_vec[0].size());
    for (size_t i = 0; i < GP_vec.size(); ++i)
        for (size_t j = 0; j < GP_vec[0].size(); ++j)
            this->GP(i, j) = GP_vec[i][j];

    // === Shape functions and derivatives ===
    auto [N1_vec, GradN1_xi_vec] = compute_N_xi_gp(this->deg1, GP_vec, this->PD);
    auto [N2_vec, GradN2_xi_vec] = compute_N_xi_gp(this->deg2, GP_vec, this->PD);

    this->N1 = element::convertToEigenMatrix(N1_vec);
    this->N2 = element::convertToEigenMatrix(N2_vec);

    // === Jacobian and gradient shape functions in physical space ===
    this->JJ     = this->compute_J(this->X, this->NGP, this->PD, GradN2_xi_vec);
    this->GradN1 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN1_xi_vec);
    this->GradN2 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN2_xi_vec);
}


Eigen::MatrixXd element::compute_J(const Eigen::MatrixXd& X_e, int NGP, int PD,
                                   const std::vector<std::vector<double>>& GradN_xi_gp) {
    // Number of nodes per element = number of rows in GradN_xi
    int numNodes = static_cast<int>(GradN_xi_gp.size());

    // Initialize output: each Jacobian block is PD×PD, stacked horizontally for each Gauss point
    Eigen::MatrixXd J_gp = Eigen::MatrixXd::Zero(PD, NGP * PD);

    for (int gp = 0; gp < NGP; ++gp) {
        // Reconstruct GradN_xi matrix for this Gauss point
        Eigen::MatrixXd GradN_xi(numNodes, PD);
        for (int i = 0; i < numNodes; ++i) {
            for (int j = 0; j < PD; ++j) {
                GradN_xi(i, j) = GradN_xi_gp[i][gp * PD + j];
            }
        }

        // Compute Jacobian: J = X_e * GradN_xi
        Eigen::MatrixXd J = X_e * GradN_xi;

        // Store J into the corresponding block in J_gp
        J_gp.block(0, gp * PD, PD, PD) = J;
    }

    return J_gp;
}


Eigen::MatrixXd element::compute_GradN(const Eigen::MatrixXd &J_gp, int number_GP, int problem_dimension,
                                       const std::vector<std::vector<double>> &gradient_N_xi) {
    // Determine matrix dimensions
    int numNodes = gradient_N_xi.size(); // Number of nodes (rows in GradN_xi)
    int cols = number_GP * problem_dimension; // Total columns in GradN_X_gp

    // Initialize GradN_X_gp as a zero matrix
    Eigen::MatrixXd GradN_X_gp = Eigen::MatrixXd::Zero(numNodes, cols);

    for (int gp = 0; gp < number_GP; ++gp) {
        // Extract the corresponding gradient block
        Eigen::MatrixXd GradN_xi(numNodes, problem_dimension);
        for (int i = 0; i < numNodes; ++i) {
            for (int j = 0; j < problem_dimension; ++j) {
                GradN_xi(i, j) = gradient_N_xi[i][gp * problem_dimension + j];
            }
        }

        // Extract the Jacobian block
        Eigen::MatrixXd J = J_gp.block(0, gp * problem_dimension, problem_dimension, problem_dimension);

        // Compute the inverse transpose of J
        Eigen::MatrixXd JinvT = J.inverse().transpose();

        // Compute GradN_X_gp at this Gauss point
        Eigen::MatrixXd GradN_X = (JinvT * GradN_xi.transpose()).transpose();

        // Store result in GradN_X_gp
        GradN_X_gp.block(0, gp * problem_dimension, GradN_X.rows(), GradN_X.cols()) = GradN_X;
    }

    return GradN_X_gp;
}

//void element::initialize_static_data(int node_per_element, int number_gauss_point, int problem_dimension, int element_order) {
//    if (gauss_points.size() == 0) {
//
//        // Compute only once
//        gauss_points_vector = compute_gp(number_gauss_point, problem_dimension);
//        gauss_points = convertToEigenMatrix(gauss_points_vector);
//        //auto testing=build(node_per_element, problem_dimension, number_gauss_point, gauss_points);
//        //std::cout<< " number of GP" << number_gauss_point<<std::endl;
//        //std::cout <<" number of dp" << problem_dimension<<std::endl;
//        //std::cout <<" GP" << gauss_points<<std::endl;
//
//        auto shape_function = compute_N_xi_gp(element_order, gauss_points_vector, problem_dimension);
//        shape_functions_N_vector = shape_function.first;
//        gradient_N_xi_vector = shape_function.second;
//
//        std::cout << "shape_functions_N_vector (" << shape_functions_N_vector.size() << " x "
//                 << (shape_functions_N_vector.empty() ? 0 : shape_functions_N_vector[0].size()) << "):" << std::endl;
//
//        /*for (const auto& row : shape_functions_N_vector) {
//            for (double value : row) {
//                std::cout << value << "\t";  // Use tab for better spacing
//            }
//            std::cout << std::endl;  // New line for each row
//        }*/
//
//
//         gauss_points = convertToEigenMatrix(gauss_points_vector);
//         shape_functions_N = convertToEigenMatrix(shape_functions_N_vector);
//         gradient_N_xi = convertToEigenMatrix(gradient_N_xi_vector);
//        //std::cout <<"in the creation"<<std::endl;
//         //std::cout<< shape_functions_N<< std::endl;
//
//
//
//    }
//}

Eigen::MatrixXd element::convertToEigenMatrix(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) return Eigen::MatrixXd();  // Return an empty matrix if input is empty
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Eigen::MatrixXd mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = vec[i][j];  // Assign each element individually
        }
    }
    return mat;
}


void element::compute_at_gp(const Eigen::VectorXd& c,
                            const Eigen::MatrixXd& v,
                            const Eigen::VectorXd& cn,
                            const Eigen::MatrixXd& vn,
                            const Eigen::MatrixXd& N1_gp,
                            const Eigen::MatrixXd& N2_gp,
                            const Eigen::MatrixXd& GradN1_gp,
                            const Eigen::MatrixXd& GradN2_gp,
                            Eigen::VectorXd& c_gp,
                            Eigen::MatrixXd& v_gp,
                            Eigen::VectorXd& cn_gp,
                            Eigen::MatrixXd& vn_gp,
                            Eigen::MatrixXd& Gradc_gp,
                            Eigen::MatrixXd& Gradv_gp,
                            Eigen::MatrixXd& Gradcn_gp,
                            Eigen::MatrixXd& Gradvn_gp)
{
    // Scalar values at Gauss points
    c_gp  = c  * N1_gp;
    cn_gp = cn * N1_gp;

    // Vector values at Gauss points
    v_gp  = v  * N2_gp;
    vn_gp = vn * N2_gp;

    // Scalar gradients at Gauss points
    Gradc_gp  = c  * GradN1_gp;
    Gradcn_gp = cn * GradN1_gp;

    // Vector gradients at Gauss points
    Gradv_gp  = v  * GradN2_gp;
    Gradvn_gp = vn * GradN2_gp;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> element::Residual(double dt) {
    // === Unpack physical parameters ===
    double E  = parameters[0];
    double R  = parameters[1];
    double xi = parameters[2];

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);

    // === Field values and gradients at Gauss points ===
    Eigen::VectorXd c_gp, cn_gp;
    Eigen::MatrixXd v_gp, vn_gp;
    Eigen::MatrixXd Gradc_gp, Gradcn_gp;
    Eigen::MatrixXd Gradv_gp, Gradvn_gp;

    this->compute_at_gp(
            this->c, this->v, this->cn, this->vn,
            this->N1, this->N2, this->GradN1, this->GradN2,
            c_gp, v_gp, cn_gp, vn_gp,
            Gradc_gp, Gradv_gp, Gradcn_gp, Gradvn_gp
    );

    // === Initialize residuals ===
    Eigen::VectorXd R1 = Eigen::VectorXd::Zero(NPE1);         // scalar field
    Eigen::VectorXd R2 = Eigen::VectorXd::Zero(NPE2 * PD);    // vector field

    int NGP = GP.cols();
    Eigen::VectorXd wp = GP.row(GP.rows() - 1);

    for (int gp = 0; gp < NGP; ++gp) {
        Eigen::VectorXd N1 = N1.col(gp);
        Eigen::VectorXd N2 = N2.col(gp);

        Eigen::MatrixXd GradN1 = GradN1.block(0, gp * PD, NPE1, PD);
        Eigen::MatrixXd GradN2 = GradN2.block(0, gp * PD, NPE2, PD);
        Eigen::MatrixXd J = JJ.block(0, gp * PD, PD, PD);
        double JxW = J.determinant() * wp(gp);

        // Extract scalar values
        double c   = c_gp(gp);
        double cn  = cn_gp(gp);

        // Extract vector values
        Eigen::VectorXd v  = v_gp.col(gp);
        Eigen::VectorXd vn = vn_gp.col(gp);

        // Extract gradients (note: Gradc, Gradcn are column vectors)
        Eigen::VectorXd Gradc   = Gradc_gp.block(0, gp * PD, 1, PD).transpose();
        Eigen::VectorXd Gradcn  = Gradcn_gp.block(0, gp * PD, 1, PD).transpose();
        Eigen::MatrixXd Gradv   = Gradv_gp.block(0, gp * PD, PD, PD);
        Eigen::MatrixXd Gradvn  = Gradvn_gp.block(0, gp * PD, PD, PD);

        // Divergence of vector field
        double Divv = Gradv.trace();

        // Stress term depending on dimension
        Eigen::MatrixXd sig(PD, PD);
        if (PD == 1) {
            sig = -(E * 2 * R * c) / (1 - 2 * R * c) * II;
        } else if (PD == 2) {
            sig = -(E * M_PI * R * R * c) / (1 - M_PI * R * R * c) * II;
        } else if (PD == 3) {
            sig = -(E * (4.0 / 3.0) * M_PI * R * R * R * c) / (1 - (4.0 / 3.0) * M_PI * R * R * R * c) * II;
        } else {
            throw std::runtime_error("Unsupported PD dimension.");
        }

        // === Residuals ===
        double R1_1 = (c - cn) / dt;
        Eigen::VectorXd R1_2 = -c * v;

        Eigen::VectorXd R2_1 = xi * c * v;
        Eigen::MatrixXd R2_2 = sig;

        // === Assembly into R1 (scalar field) ===
        for (int I = 0; I < NPE1; ++I) {
            double r = R1_1 * N1(I) + R1_2.dot(GradN1.row(I));
            R1(I) += r * JxW;
        }

        // === Assembly into R2 (vector field) ===
        for (int I = 0; I < NPE2; ++I) {
            Eigen::VectorXd r = R2_1 * N2(I) + R2_2 * GradN2.row(I).transpose();
            R2.segment(I * PD, PD) += r * JxW;
        }
    }

    return {R1, R2};
}

void element::assemble_tangent_matrices(double JxW,
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
                                        Eigen::MatrixXd& K22) {
    for (int I = 0; I < NPE1; ++I) {
        for (int J = 0; J < NPE1; ++J) {
            double K11_1 = (1.0 / parameters[3]) * N1_gp(I) * N1_gp(J) * JxW;

            Eigen::VectorXd dR1_2_dc = -v_val;
            Eigen::MatrixXd dR1_2_dGradc = Eigen::MatrixXd::Zero(PD, PD);

            Eigen::VectorXd K11_3 = dR1_2_dc.transpose() * GradN1_gp.row(I).transpose() * N1_gp(J) * JxW;

            double K11_2 = 0.0;
            double K11_4 = 0.0;

            for (int i = 0; i < PD; ++i)
                for (int j = 0; j < PD; ++j)
                    K11_4 += dR1_2_dGradc(i, j) * GradN1_gp(I, i) * GradN1_gp(J, j) * JxW;

            double total_K = K11_1 + K11_2 + K11_3.sum() + K11_4;

            K11(I, J) += total_K;
        }

        for (int J = 0; J < NPE2; ++J) {
            Eigen::VectorXd dR1_1_dv = Eigen::VectorXd::Zero(PD);
            Eigen::MatrixXd dR1_1_dGradv = Eigen::MatrixXd::Zero(PD, PD);
            Eigen::MatrixXd dR1_2_dv = -c_val * Eigen::MatrixXd::Identity(PD, PD);
            Eigen::MatrixXd K12_4 = Eigen::MatrixXd::Zero(PD, 1);

            Eigen::VectorXd K12_1 = dR1_1_dv * N1_gp(I) * N2_gp(J) * JxW;
            Eigen::VectorXd K12_2 = dR1_1_dGradv * N1_gp(I) * GradN2_gp.row(J).transpose() * JxW;
            Eigen::VectorXd K12_3 = dR1_2_dv.transpose() * GradN1_gp.row(I).transpose() * N2_gp(J) * JxW;

            Eigen::VectorXd total_K = K12_1 + K12_2 + K12_3 + K12_4;

            for (int d = 0; d < PD; ++d)
                K12(I, J * PD + d) += total_K(d);
        }
    }

    for (int I = 0; I < NPE2; ++I) {
        for (int J = 0; J < NPE1; ++J) {
            Eigen::VectorXd dR2_1_dc = parameters[2] * v_val;
            Eigen::MatrixXd dR2_2_dc = dsig_dc;

            Eigen::VectorXd K21_1 = dR2_1_dc * N2_gp(I) * N1_gp(J) * JxW;
            Eigen::VectorXd K21_2 = Eigen::VectorXd::Zero(PD);
            Eigen::VectorXd K21_3 = dR2_2_dc * GradN2_gp.row(I).transpose() * N1_gp(J) * JxW;
            Eigen::VectorXd K21_4 = Eigen::VectorXd::Zero(PD);

            Eigen::VectorXd total_K = K21_1 + K21_2 + K21_3 + K21_4;
            for (int d = 0; d < PD; ++d)
                K21(I * PD + d, J) += total_K(d);
        }

        for (int J = 0; J < NPE2; ++J) {
            Eigen::MatrixXd K22_1 = parameters[2] * c_val * N2_gp(I) * N2_gp(J) * Eigen::MatrixXd::Identity(PD, PD) * JxW;
            Eigen::MatrixXd K22_2 = Eigen::MatrixXd::Zero(PD, PD);
            Eigen::MatrixXd K22_3 = Eigen::MatrixXd::Zero(PD, PD);
            Eigen::MatrixXd K22_4 = Eigen::MatrixXd::Zero(PD, PD);

            Eigen::MatrixXd total_K = K22_1 + K22_2 + K22_3 + K22_4;

            for (int d1 = 0; d1 < PD; ++d1)
                for (int d2 = 0; d2 < PD; ++d2)
                    K22(I * PD + d1, J * PD + d2) += total_K(d1, d2);
        }
    }
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
element::RK(double dt) {
    double E = parameters[0];
    double R = parameters[1];
    double xi = parameters[2];

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);

    Eigen::VectorXd c_gp, cn_gp;
    Eigen::MatrixXd v_gp, vn_gp;
    Eigen::MatrixXd Gradc_gp, Gradcn_gp;
    Eigen::MatrixXd Gradv_gp, Gradvn_gp;

    compute_at_gp(c, v, cn, vn, N1, N2, GradN1, GradN2,
                  c_gp, v_gp, cn_gp, vn_gp,
                  Gradc_gp, Gradv_gp, Gradcn_gp, Gradvn_gp);

    Eigen::VectorXd R1 = Eigen::VectorXd::Zero(NPE1);
    Eigen::VectorXd R2 = Eigen::VectorXd::Zero(NPE2 * PD);
    Eigen::MatrixXd K11 = Eigen::MatrixXd::Zero(NPE1, NPE1);
    Eigen::MatrixXd K12 = Eigen::MatrixXd::Zero(NPE1, NPE2 * PD);
    Eigen::MatrixXd K21 = Eigen::MatrixXd::Zero(NPE2 * PD, NPE1);
    Eigen::MatrixXd K22 = Eigen::MatrixXd::Zero(NPE2 * PD, NPE2 * PD);

    int NGP = GP.cols();
    Eigen::VectorXd wp = GP.row(GP.rows() - 1);

    for (int gp = 0; gp < NGP; ++gp) {
        Eigen::VectorXd N1_gp = N1.col(gp);
        Eigen::VectorXd N2_gp = N2.col(gp);
        Eigen::MatrixXd GradN1_gp = GradN1.block(0, gp * PD, NPE1, PD);
        Eigen::MatrixXd GradN2_gp = GradN2.block(0, gp * PD, NPE2, PD);
        Eigen::MatrixXd J = JJ.block(0, gp * PD, PD, PD);
        double JxW = J.determinant() * wp(gp);

        double c_val = c_gp(gp);
        double cn_val = cn_gp(gp);
        Eigen::VectorXd v_val = v_gp.col(gp);

        Eigen::VectorXd Gradc = Gradc_gp.block(0, gp * PD, 1, PD).transpose();
        Eigen::MatrixXd Gradv = Gradv_gp.block(0, gp * PD, PD, PD);

        Eigen::MatrixXd sig = Eigen::MatrixXd::Zero(PD, PD);
        Eigen::MatrixXd dsig_dc = Eigen::MatrixXd::Zero(PD, PD);

        if (PD == 1) {
            sig = -(E * 2 * R * c_val) / (1 - 2 * R * c_val) * II;
            dsig_dc = -(2 * E * R) / std::pow(1 - 2 * R * c_val, 2) * II;
        } else if (PD == 2) {
            sig = -(E * M_PI * R * R * c_val) / (1 - M_PI * R * R * c_val) * II;
            dsig_dc = -(E * M_PI * R * R) / std::pow(1 - M_PI * R * R * c_val, 2) * II;
        } else if (PD == 3) {
            sig = -(E * (4.0 / 3.0) * M_PI * R * R * R * c_val) / (1 - (4.0 / 3.0) * M_PI * R * R * R * c_val) * II;
            dsig_dc = -(E * (4.0 / 3.0) * M_PI * R * R * R) / std::pow(1 - (4.0 / 3.0) * M_PI * R * R * R * c_val, 2) * II;
        }

        double R1_1 = (c_val - cn_val) / dt;
        Eigen::VectorXd R1_2 = -c_val * v_val;
        Eigen::VectorXd R2_1 = xi * c_val * v_val;
        Eigen::MatrixXd R2_2 = sig;

        for (int I = 0; I < NPE1; ++I) {
            double r = R1_1 * N1_gp(I) + R1_2.dot(GradN1_gp.row(I));
            R1(I) += r * JxW;
        }

        for (int I = 0; I < NPE2; ++I) {
            Eigen::VectorXd r = R2_1 * N2_gp(I) + R2_2 * GradN2_gp.row(I).transpose();
            R2.segment(I * PD, PD) += r * JxW;
        }

        assemble_tangent_matrices(JxW, c_val, v_val, N1_gp, N2_gp, GradN1_gp, GradN2_gp, sig, dsig_dc, K11, K12, K21, K22);
    }

    return {R1, R2, K11, K12, K21, K22};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> element::RK_GP(double dt, int NGP_val) {
    double E  = parameters[0];
    double R  = parameters[1];
    double xi = parameters[2];

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd III = Eigen::MatrixXd::Identity(PD * PD, PD * PD);

    Eigen::VectorXd c_gp, cn_gp;
    Eigen::MatrixXd v_gp, vn_gp;
    Eigen::MatrixXd Gradc_gp, Gradcn_gp;
    Eigen::MatrixXd Gradv_gp, Gradvn_gp;

    compute_at_gp(c, v, cn, vn, N1, N2, GradN1, GradN2,
                  c_gp, v_gp, cn_gp, vn_gp,
                  Gradc_gp, Gradv_gp, Gradcn_gp, Gradvn_gp);

    Eigen::MatrixXd R_GP = Eigen::MatrixXd::Zero(NPE1, NGP_val);
    Eigen::MatrixXd K_GP = Eigen::MatrixXd::Zero(NPE1, NPE1);

    int NGP = GP.cols();
    Eigen::VectorXd wp = GP.row(GP.rows() - 1);

    for (int gp = 0; gp < NGP; ++gp) {
        Eigen::VectorXd N1_gp = N1.col(gp);
        Eigen::MatrixXd GradN1_gp = GradN1.block(0, gp * PD, NPE1, PD);
        Eigen::MatrixXd GradN2_gp = GradN2.block(0, gp * PD, NPE2, PD);
        Eigen::MatrixXd J = JJ.block(0, gp * PD, PD, PD);
        double JxW = J.determinant() * wp(gp);

        double c_val = c_gp(gp);
        Eigen::VectorXd v_val = v_gp.col(gp);
        double cn_val = cn_gp(gp);
        Eigen::VectorXd vn_val = vn_gp.col(gp);

        Eigen::VectorXd Gradc = Gradc_gp.block(0, gp * PD, 1, PD).transpose();
        Eigen::VectorXd Gradcn = Gradcn_gp.block(0, gp * PD, 1, PD).transpose();
        Eigen::MatrixXd Gradv = Gradv_gp.block(0, gp * PD, PD, PD);
        Eigen::MatrixXd Gradvn = Gradvn_gp.block(0, gp * PD, PD, PD);

        Eigen::MatrixXd sig = Eigen::MatrixXd::Zero(PD, PD);
        if (PD == 1) {
            sig = -(E * 2 * R * c_val) / (1 - 2 * R * c_val) * II;
        } else if (PD == 2) {
            sig = -(E * M_PI * R * R * c_val) / (1 - M_PI * R * R * c_val) * II;
        } else if (PD == 3) {
            sig = -(E * (4.0 / 3.0) * M_PI * R * R * R * c_val) / (1 - (4.0 / 3.0) * M_PI * R * R * R * c_val) * II;
        }

        Eigen::VectorXd R1 = Eigen::Map<Eigen::VectorXd>(sig.transpose().data(), PD * PD);

        for (int s = 0; s < R1.size(); ++s) {
            for (int I = 0; I < NPE1; ++I) {
                double R11 = R1(s) * N1_gp(I) * JxW;
                R_GP(I, s) += R11;
            }
        }

        for (int I = 0; I < NPE1; ++I) {
            for (int J = 0; J < NPE1; ++J) {
                double K11 = N1_gp(I) * N1_gp(J) * JxW;
                K_GP(I, J) += K11;
            }
        }
    }

    return {R_GP, K_GP};
}



