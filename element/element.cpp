#include "element.hpp"
#include "../utils/utils.hpp"

// Define static members
//Eigen::MatrixXd element::gauss_points;
//Eigen::MatrixXd element::shape_functions_N;
//Eigen::MatrixXd element::gradient_N_xi;
//std::vector<std::vector<double>> element::gauss_points_vector;
//std::vector<std::vector<double>> element::shape_functions_N_vector;
//std::vector<std::vector<double>> element::gradient_N_xi_vector;


class Timer {
public:
    Timer(const std::string& name = "Timer")
            : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " took " << duration << " ms\n";
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
};


#include "element.hpp"
#include "../utils/utils.hpp"  // compute_gp, compute_N_xi_gp, printMatrix

template <class Derived>
inline void printEigen(const std::string& name,
                       const Eigen::MatrixBase<Derived>& X) {
    static const Eigen::IOFormat fmt(
            Eigen::StreamPrecision,  // precision
            0,                       // no alignment commas
            " ",                     // coeff separator
            "\n",                    // row separator
            "", "",                  // row prefix/suffix
            "", ""                   // mat prefix/suffix
    );
    std::cout << name << "  (" << X.rows() << "x" << X.cols() << ")\n"
              << X.format(fmt) << "\n\n";
}


element::element(int Nr, int PD,
                 const Eigen::VectorXd& NdL1,
                 const Eigen::VectorXd& NdL2,
                 const Eigen::MatrixXd& X,
                 const Eigen::VectorXd& C,
                 const Eigen::MatrixXd& V,
                 int NGP,
                 std::pair<int,int> element_order,
                 const std::vector<double>& parameters)
        : Nr(Nr), PD(PD), NdL1(NdL1), NdL2(NdL2), X(X), C(C), V(V),
          NGP(NGP), parameters(parameters)
{
    this->deg1 = element_order.first;
    this->deg2 = element_order.second;
    this->NPE1 = NdL1.size();
    this->NPE2 = NdL2.size();

    this->x  = X; this->xn = X;
    this->c  = C; this->cn = C;
    this->v  = V; this->vn = V;

    // 1) Gauss points: reuse per (NGP, PD)
    this->GP = fem_cache::ShapeCache::gp_mat(this->NGP, this->PD);

    // 2) Shape functions in parent space: reuse per (degree, NGP, PD)
    const auto& N1_mat = fem_cache::ShapeCache::N_mat(this->deg1, this->NGP, this->PD);
    const auto& N2_mat = fem_cache::ShapeCache::N_mat(this->deg2, this->NGP, this->PD);
    this->N1 = N1_mat;
    this->N2 = N2_mat;

    // 3) Gradients in parent space come from the same cached pair
    const auto& GradN1_xi_vec = fem_cache::ShapeCache::N_and_Grad_vec(this->deg1, this->NGP, this->PD).second;
    const auto& GradN2_xi_vec = fem_cache::ShapeCache::N_and_Grad_vec(this->deg2, this->NGP, this->PD).second;

    // 4) Geometry-dependent quantities are per-element (must be recomputed)
    this->JJ     = this->compute_J(this->X, this->NGP, this->PD, GradN2_xi_vec);
    this->GradN1 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN1_xi_vec);
    this->GradN2 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN2_xi_vec);
}


//element::element(int Nr, int PD,
//                 const Eigen::VectorXd& NdL1,
//                 const Eigen::VectorXd& NdL2,
//                 const Eigen::MatrixXd& X,
//                 const Eigen::VectorXd& C,
//                 const Eigen::MatrixXd& V,
//                 int NGP,
//                 std::pair<int, int> element_order,
//                 const std::vector<double>& parameters)
//        : Nr(Nr), PD(PD), NdL1(NdL1), NdL2(NdL2), X(X), C(C), V(V),
//          NGP(NGP), parameters(parameters) {
//
//    // Assign degrees and nodes per element
//    this->deg1 = element_order.first;
//    this->deg2 = element_order.second;
//    this->NPE1 = NdL1.size();
//    this->NPE2 = NdL2.size();
//
//    // Initialize geometry
//    this->x  = X;
//    this->xn = X;
//
//    // Scalar field (concentration)
//    this->c  = C;
//    this->cn = C;
//
//    // Vector field (velocity)
//    this->v  = V;
//    this->vn = V;
//
//    // === Gauss points ===
//    std::vector<std::vector<double>> GP_vec = compute_gp(this->NGP, this->PD);
//    this->GP = Eigen::MatrixXd(GP_vec.size(), GP_vec[0].size());
//    for (size_t i = 0; i < GP_vec.size(); ++i)
//        for (size_t j = 0; j < GP_vec[0].size(); ++j)
//            this->GP(i, j) = GP_vec[i][j];
//
//    // === Shape functions and derivatives ===
//    auto [N1_vec, GradN1_xi_vec] = compute_N_xi_gp(this->deg1, GP_vec, this->PD);
//    auto [N2_vec, GradN2_xi_vec] = compute_N_xi_gp(this->deg2, GP_vec, this->PD);
//
//    this->N1 = element::convertToEigenMatrix(N1_vec);
//    this->N2 = element::convertToEigenMatrix(N2_vec);
//
//    // === Jacobian and gradient shape functions in physical space ===
//    this->JJ     = this->compute_J(this->X, this->NGP, this->PD, GradN2_xi_vec);
//    this->GradN1 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN1_xi_vec);
//    this->GradN2 = this->compute_GradN(this->JJ, this->NGP, this->PD, GradN2_xi_vec);
//}


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
    // Scalar values at Gauss points  (OK: (NGP x 1))
    c_gp  = N1_gp.transpose() * c;   // (NGP x 1)
    cn_gp = N1_gp.transpose() * cn;  // (NGP x 1)

// Vector values at Gauss points (already OK for shapes PD×NPE2 times NPE2×NGP = PD×NGP)
    v_gp  = v  * N2_gp;
    vn_gp = vn * N2_gp;

// Scalar gradients at Gauss points (you later index as 1×(PD*NGP), so make row)
    Gradc_gp  = c.transpose()  * GradN1_gp;  // (1 x PD*NGP)
    Gradcn_gp = cn.transpose() * GradN1_gp;  // (1 x PD*NGP)

// Vector gradients at Gauss points (already consistent: PD×NPE2 times NPE2×(PD*NGP) = PD×(PD*NGP))
    Gradv_gp  = v  * GradN2_gp;
    Gradvn_gp = vn * GradN2_gp;

}

std::pair<Eigen::VectorXd, Eigen::VectorXd> element::Residual(double dt) {
    // unpack parameters
    const double E  = parameters[0];
    const double R  = parameters[1];
    const double xi = parameters[2];

    const Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);

    // interpolate fields and gradients at GP
    Eigen::VectorXd c_gp, cn_gp;
    Eigen::MatrixXd v_gp, vn_gp;
    Eigen::MatrixXd Gradc_gp, Gradcn_gp;
    Eigen::MatrixXd Gradv_gp, Gradvn_gp;

    compute_at_gp(c, v, cn, vn, N1, N2, GradN1, GradN2,
                  c_gp, v_gp, cn_gp, vn_gp,
                  Gradc_gp, Gradv_gp, Gradcn_gp, Gradvn_gp);

    // residual vectors
    Eigen::VectorXd R1 = Eigen::VectorXd::Zero(NPE1);        // size NPE1×1
    Eigen::VectorXd R2 = Eigen::VectorXd::Zero(NPE2 * PD);   // size (NPE2*PD)×1

    // number of GP and weights (last row of GP)
    const int NGP = GP.cols();
    const Eigen::VectorXd wp = GP.row(GP.rows() - 1).transpose();

    for (int gp = 0; gp < NGP; ++gp) {
        // N1 = N1_gp(:,gp)   N2 = N2_gp(:,gp)
        const Eigen::VectorXd N1_gp = N1.col(gp);              // NPE1×1
        const Eigen::VectorXd N2_gp = N2.col(gp);              // NPE2×1

        // GradN1 = GradN1_gp(:, block)   GradN2 = GradN2_gp(:, block)
        const int col0 = gp * PD;
        const Eigen::MatrixXd GradN1_gp = GradN1.block(0, col0, NPE1, PD); // NPE1×PD
        const Eigen::MatrixXd GradN2_gp = GradN2.block(0, col0, NPE2, PD); // NPE2×PD

        // JxW = det(JJ(:,block)) * wp(gp)
        const Eigen::MatrixXd J = JJ.block(0, col0, PD, PD);
        const double JxW = J.determinant() * wp(gp);

        // values at this GP
        const double c_val = c_gp(gp);
        const double cn_val = cn_gp(gp);

        const Eigen::VectorXd v_val  = v_gp.col(gp);           // PD×1
        const Eigen::VectorXd vn_val = vn_gp.col(gp);          // PD×1

//        std::cout<<"Gradcn_gp"<<std::endl;
//        std::cout<<Gradc_gp<<std::endl;
        const Eigen::VectorXd Gradc  = Gradc_gp.block(0, col0, 1, PD).transpose();  // PD×1
        const Eigen::VectorXd Gradcn = Gradcn_gp.block(0, col0, 1, PD).transpose();// PD×1
//        std::cout<<"Gradvn_gp"<<std::endl;
//        std::cout<<Gradv_gp<<std::endl;
        const Eigen::MatrixXd Gradv  = Gradv_gp.block(0, col0, PD, PD);             // PD×PD
        const Eigen::MatrixXd Gradvn = Gradvn_gp.block(0, col0, PD, PD);            // PD×PD

        // Divv = trace(Gradv)
        const double Divv = Gradv.trace();

        // sig depending on PD
        Eigen::MatrixXd sig(PD, PD);
        if (PD == 1) {
            sig = -(E * 2.0 * R * c_val) / (1.0 - 2.0 * R * c_val) * II;
        } else if (PD == 2) {
            sig = -(E * M_PI * R * R * c_val) / (1.0 - M_PI * R * R * c_val) * II;
        } else if (PD == 3) {
            const double coeff = (4.0 / 3.0) * M_PI * R * R * R;
            sig = -(E * coeff * c_val) / (1.0 - coeff * c_val) * II;
        } else {
            throw std::runtime_error("Residual: unsupported PD");
        }

        // local pieces
        const double        R1_1 = (c_val - cn_val) / dt;
        const Eigen::VectorXd R1_2 = -c_val * v_val;

        const Eigen::VectorXd R2_1 = xi * c_val * v_val;
        const Eigen::MatrixXd R2_2 = sig;

        // assemble R1
        for (int I = 0; I < NPE1; ++I) {
            // R11 = ( R1_1 * N1(I) + R1_2' * GradN1(I,:)' ) * JxW
            const double scalar = R1_1 * N1_gp(I)
                                  + R1_2.dot(GradN1_gp.row(I).transpose());
            R1(I) += scalar * JxW;
        }

        // assemble R2
        for (int I = 0; I < NPE2; ++I) {
            // R22 = ( R2_1 * N2(I) + R2_2 * GradN2(I,:)' ) * JxW
            Eigen::VectorXd r = R2_1 * N2_gp(I)
                                + R2_2 * GradN2_gp.row(I).transpose();   // PD×1
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
    //Timer t("timer RK");
// Unpack parameters (E, R, xi)
    double E  = parameters[0];
    double R  = parameters[1];
    double xi = parameters[2];

// Identity
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);

// Values and grads at GP
    Eigen::VectorXd c_gp, cn_gp;
    Eigen::MatrixXd v_gp, vn_gp;
    Eigen::MatrixXd Gradc_gp, Gradcn_gp;
    Eigen::MatrixXd Gradv_gp, Gradvn_gp;

// Interpolate to GP (c, v, cn, vn; N1, N2; GradN1, GradN2)
    compute_at_gp(c, v, cn, vn, N1, N2, GradN1, GradN2,
                  c_gp, v_gp, cn_gp, vn_gp,
                  Gradc_gp, Gradv_gp, Gradcn_gp, Gradvn_gp);

// Allocate residuals and blocks
    Eigen::VectorXd R1 = Eigen::VectorXd::Zero(NPE1);       // c
    Eigen::VectorXd R2 = Eigen::VectorXd::Zero(NPE2 * PD);  // v
    Eigen::MatrixXd K11 = Eigen::MatrixXd::Zero(NPE1, NPE1);
    Eigen::MatrixXd K12 = Eigen::MatrixXd::Zero(NPE1, NPE2 * PD);
    Eigen::MatrixXd K21 = Eigen::MatrixXd::Zero(NPE2 * PD, NPE1);
    Eigen::MatrixXd K22 = Eigen::MatrixXd::Zero(NPE2 * PD, NPE2 * PD);

// Number of GP and weights
    int NGP = GP.cols();
    Eigen::VectorXd wp = GP.row(GP.rows() - 1);


//    for (int gp = 0; gp < NGP; ++gp) {
//        // column/block offset for this Gauss point
//        const int col0 = gp * PD;
//
//// N1 = N1_gp(:,gp);  N2 = N2_gp(:,gp);
//        Eigen::VectorXd N1 = this->N1.col(gp);            // size: NPE1 × 1
//        Eigen::VectorXd N2 = this->N2.col(gp);            // size: NPE2 × 1
//
//// GradN1 = GradN1_gp(:,(gp-1)*PD+1:(gp-1)*PD+PD);
//// GradN2 = GradN2_gp(:,(gp-1)*PD+1:(gp-1)*PD+PD);
//        Eigen::MatrixXd GradN1 = this->GradN1.block(0, col0, NPE1, PD); // NPE1 × PD
//        Eigen::MatrixXd GradN2 = this->GradN2.block(0, col0, NPE2, PD); // NPE2 × PD
//
//// JxW = det(JJ(:,(gp-1)*PD+1:(gp-1)*PD+PD)) * wp(gp);
//        Eigen::MatrixXd JJgp = this->JJ.block(0, col0, PD, PD); // PD × PD
//        double JxW = JJgp.determinant() * wp(gp);
//
//// c  = c_gp(:,gp);  v  = v_gp(:,gp);
//        double c  = c_gp(gp);
//        Eigen::VectorXd v  = v_gp.col(gp);               // PD × 1
//
//// cn = cn_gp(:,gp); vn = vn_gp(:,gp);
//        double cn = cn_gp(gp);
//        Eigen::VectorXd vn = vn_gp.col(gp);              // PD × 1
//
//// Gradc  = Gradc_gp(:,block)';   Gradv  = Gradv_gp(:,block);
//        Eigen::VectorXd Gradc  = Gradc_gp.block(0, col0, 1, PD).transpose(); // PD × 1
//        Eigen::MatrixXd Gradv  = Gradv_gp.block(0, col0, PD, PD);            // PD × PD
//
//// Gradcn = Gradcn_gp(:,block)';  Gradvn = Gradvn_gp(:,block);
//        Eigen::VectorXd Gradcn = Gradcn_gp.block(0, col0, 1, PD).transpose(); // PD × 1
//        Eigen::MatrixXd Gradvn = Gradvn_gp.block(0, col0, PD, PD);            // PD × PD
//
//// Divv = trace(Gradv);
//        double Divv = Gradv.trace();
//
//        // Constitutive stress tensor and its derivatives depending on problem dimension
//        Eigen::MatrixXd sig, dsig_dc;
//        Eigen::MatrixXd dsig_dv;                // 3rd-order in MATLAB, collapsed here
//        std::vector<Eigen::MatrixXd> dsig_dGradc; // 3rd-order zeros(PD,PD,PD)
//        std::vector<std::vector<Eigen::MatrixXd>> dsig_dGradv; // 4th-order zeros(PD,PD,PD,PD)
//
//        if (PD == 1) {
//            sig     = -(E * 2.0 * R * c) / (1.0 - 2.0 * R * c) * II;
//            dsig_dc = -(2.0 * E * R) / std::pow(1.0 - 2.0 * R * c, 2) * II;
//
//            dsig_dv     = Eigen::MatrixXd::Zero(PD, PD * PD);
//            dsig_dGradc = std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD));
//            dsig_dGradv = std::vector<std::vector<Eigen::MatrixXd>>(PD,
//                                                                    std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD)));
//        }
//        else if (PD == 2) {
//            sig     =  -(E * M_PI * R * R * c) / (1.0 - M_PI * R * R * c) * II;
//            dsig_dc = -(E * M_PI * R * R) / std::pow(1.0 - M_PI * R * R * c, 2) * II;
//
//            dsig_dv     = Eigen::MatrixXd::Zero(PD, PD * PD);
//            dsig_dGradc = std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD));
//            dsig_dGradv = std::vector<std::vector<Eigen::MatrixXd>>(PD,
//                                                                    std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD)));
//        }
//        else if (PD == 3) {
//            double coeff = (4.0 / 3.0) * M_PI * R * R * R;
//            sig     = -(E * coeff * c) / (1.0 - coeff * c) * II;
//            dsig_dc = -(E * coeff) / std::pow(1.0 - coeff * c, 2) * II;
//
//            dsig_dv     = Eigen::MatrixXd::Zero(PD, PD * PD);
//            dsig_dGradc = std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD));
//            dsig_dGradv = std::vector<std::vector<Eigen::MatrixXd>>(PD,
//                                                                    std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD)));
//        }
//        // Local residual pieces and their derivatives (sizes follow PD)
//        double R1_1 = (c - cn) / dt;
//        double dR1_1_dc = 1.0 / dt;
//        Eigen::VectorXd dR1_1_dv      = Eigen::VectorXd::Zero(PD);     // PD×1
//        Eigen::VectorXd dR1_1_dGradc  = Eigen::VectorXd::Zero(PD);     // PD×1
//        Eigen::MatrixXd dR1_1_dGradv  = Eigen::MatrixXd::Zero(PD, PD); // PD×PD
//
//        Eigen::VectorXd R1_2   = -c * v;                               // PD×1
//        Eigen::VectorXd dR1_2_dc  = -v;                                 // PD×1
//        Eigen::MatrixXd dR1_2_dv  = -c * II;                            // PD×PD
//        Eigen::MatrixXd dR1_2_dGradc = Eigen::MatrixXd::Zero(PD, PD);   // PD×PD
//        std::vector<Eigen::MatrixXd> dR1_2_dGradv(                      // PD×PD×PD
//                PD, Eigen::MatrixXd::Zero(PD, PD));
//
//        Eigen::VectorXd R2_1   = xi * c * v;                            // PD×1
//        Eigen::VectorXd dR2_1_dc = xi * v;                              // PD×1
//        Eigen::MatrixXd dR2_1_dv = xi * c * II;                         // PD×PD
//        Eigen::MatrixXd dR2_1_dGradc = Eigen::MatrixXd::Zero(PD, PD);   // PD×PD
//        std::vector<Eigen::MatrixXd> dR2_1_dGradv(                      // PD×PD×PD
//                PD, Eigen::MatrixXd::Zero(PD, PD));
//
//        Eigen::MatrixXd R2_2   = sig;                                   // PD×PD
//        Eigen::MatrixXd dR2_2_dc = dsig_dc;                             // PD×PD
//        std::vector<Eigen::MatrixXd> dR2_2_dv(                          // PD×PD×PD
//                PD, Eigen::MatrixXd::Zero(PD, PD));
//        std::vector<Eigen::MatrixXd> dR2_2_dGradc(                      // PD×PD×PD
//                PD, Eigen::MatrixXd::Zero(PD, PD));
//        std::vector<std::vector<Eigen::MatrixXd>> dR2_2_dGradv(         // PD×PD×PD×PD
//                PD, std::vector<Eigen::MatrixXd>(PD, Eigen::MatrixXd::Zero(PD, PD)));
//
//        // Assemble scalar-field residual R1 from local contributions
//        for (int I = 0; I < NPE1; ++I) {
//            double R11 = (R1_1 * N1(I) + R1_2.dot(GradN1.row(I).transpose())) * JxW;
//            R1(I) += R11; // same as MATLAB’s (I-1)*1+1 slice
//        }
//
//        // Assemble vector-field residual R2 from local contributions
//        for (int I = 0; I < NPE2; ++I) {
//            Eigen::VectorXd R22 = (R2_1 * N2(I) + R2_2 * GradN2.row(I).transpose()) * JxW;
//            R2.segment(I * PD, PD) += R22;
//        }
//
//
//        // Assemble tangent blocks K11 (scalar-scalar) and K12 (scalar-vector)
//        for (int I = 0; I < NPE1; ++I) {
//            for (int J = 0; J < NPE1; ++J) {
//                double K11_1 = dR1_1_dc * N1(I) * N1(J) * JxW;
//
//                double K11_2 = dR1_1_dGradc.dot(N1(I) * GradN1.row(J).transpose()) * JxW;
//
//                double K11_3 = dR1_2_dc.dot(GradN1.row(I).transpose()) * N1(J) * JxW;
//
//                double K11_4 = 0.0;
//                for (int i = 0; i < PD; ++i)
//                    for (int j = 0; j < PD; ++j)
//                        K11_4 += dR1_2_dGradc(i, j) * GradN1(I, i) * GradN1(J, j) * JxW;
//
//                double K = K11_1 + K11_2 + K11_3 + K11_4;
//                K11(I, J) += K;
//            }
//
//            for (int J = 0; J < NPE2; ++J) {
//                Eigen::VectorXd K12_1 = dR1_1_dv * (N1(I) * N2(J) * JxW);
//
//                Eigen::VectorXd K12_2 = dR1_1_dGradv * (N1(I) * JxW) * GradN2.row(J).transpose();
//
//                Eigen::VectorXd K12_3 = dR1_2_dv.transpose()
//                                        * (GradN1.row(I).transpose() * (N2(J) * JxW));
//
//                Eigen::VectorXd K12_4 = Eigen::VectorXd::Zero(PD);
//                for (int i = 0; i < PD; ++i)
//                    for (int j = 0; j < PD; ++j)
//                        for (int l = 0; l < PD; ++l)
//                            K12_4(i) += dR1_2_dGradv[i](j, l) * GradN1(I, j) * GradN2(J, l) * JxW;
//
//                Eigen::RowVectorXd K = (K12_1 + K12_2 + K12_3 + K12_4).transpose(); // 1×PD
//                K12.block(I, J * PD, 1, PD) += K;
//            }
//        }
//
//        // Assemble tangent blocks K21 (vector–scalar) and K22 (vector–vector)
//        // Assemble tangent blocks K21 (vector–scalar) and K22 (vector–vector)
//        for (int I = 0; I < NPE2; ++I) {
//            for (int J = 0; J < NPE1; ++J) {
//                Eigen::VectorXd K21_1 = dR2_1_dc * (N2(I) * N1(J) * JxW);
//
//                Eigen::VectorXd K21_2 = dR2_1_dGradc * (N2(I) * JxW) * GradN1.row(J).transpose();
//
//                Eigen::VectorXd K21_3 = dR2_2_dc * (GradN2.row(I).transpose() * (N1(J) * JxW));
//
//                Eigen::VectorXd K21_4 = Eigen::VectorXd::Zero(PD);
//                for (int i = 0; i < PD; ++i)
//                    for (int j = 0; j < PD; ++j)
//                        for (int k = 0; k < PD; ++k)
//                            K21_4(i) += dR2_2_dGradc[i](j, k) * GradN2(I, j) * GradN1(J, k) * JxW;
//
//                Eigen::VectorXd K = K21_1 + K21_2 + K21_3 + K21_4; // PD×1
//                K21.block(I * PD, J, PD, 1) += K;
//            }
//
//            for (int J = 0; J < NPE2; ++J) {
//                Eigen::MatrixXd K22_1 = dR2_1_dv * (N2(I) * N2(J) * JxW);
//
//                Eigen::MatrixXd K22_2 = Eigen::MatrixXd::Zero(PD, PD);
//                for (int i = 0; i < PD; ++i)
//                    for (int j = 0; j < PD; ++j)
//                        for (int l = 0; l < PD; ++l)
//                            K22_2(i, j) += dR2_1_dGradv[i](j, l) * N2(I) * GradN2(J, l) * JxW;
//
//                Eigen::MatrixXd K22_3 = Eigen::MatrixXd::Zero(PD, PD);
//                for (int i = 0; i < PD; ++i)
//                    for (int k = 0; k < PD; ++k)
//                        for (int j = 0; j < PD; ++j)
//                            K22_3(i, k) += dR2_2_dv[i](j, k) * GradN2(I, j) * N2(J) * JxW;
//
//                Eigen::MatrixXd K22_4 = Eigen::MatrixXd::Zero(PD, PD);
//                for (int i = 0; i < PD; ++i)
//                    for (int k = 0; k < PD; ++k)
//                        for (int j = 0; j < PD; ++j)
//                            for (int m = 0; m < PD; ++m)
//                                K22_4(i, k) += dR2_2_dGradv[i][j](k, m) * GradN2(I, j) * GradN2(J, m) * JxW;
//
//                Eigen::MatrixXd K = K22_1 + K22_2 + K22_3 + K22_4; // PD×PD
//                K22.block(I * PD, J * PD, PD, PD) += K;
//            }
//        }
//
//
//    } // end gp loop

    for (int gp = 0; gp < NGP; ++gp){
        // Assuming: N1_, N2_, GradN1_, GradN2_, JJ_, GP already set as member matrices.
// Here inside the GP loop:
        const int col0 = gp * PD;

// views (no copies)
        const auto N1v     = N1.col(gp);                        // NPE1 x 1
        const auto N2v     = N2.col(gp);                        // NPE2 x 1
        const auto GradN1v = GradN1.block(0, col0, NPE1, PD);   // NPE1 x PD
        const auto GradN2v = GradN2.block(0, col0, NPE2, PD);   // NPE2 x PD
        const auto JJgp    = JJ.block(0, col0, PD, PD);         // PD   x PD

        const double w = GP(PD, gp);                             // last row are weights
        const double JxW = JJgp.determinant() * w;

// map GP values (no new allocs)
        const double c   = c_gp(gp);
        const double cn  = cn_gp(gp);
        Eigen::VectorXd  v   = v_gp.col(gp);
        Eigen::VectorXd  vn  = vn_gp.col(gp);

        Eigen::Matrix<double, Eigen::Dynamic, 1> Gradc(PD), Gradcn(PD);
        Gradc  = Gradc_gp.block(0, col0, 1, PD).transpose();
        Gradcn = Gradcn_gp.block(0, col0, 1, PD).transpose();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Gradv(PD,PD), Gradvn(PD,PD);
        Gradv  = Gradv_gp.block(0, col0, PD, PD);
        Gradvn = Gradvn_gp.block(0, col0, PD, PD);

        const double Divv = Gradv.trace();

// sig(c) & dsig_dc
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sig(PD,PD), dsig_dc(PD,PD);
        {
            double coeff;
            if (PD==1)        coeff = 2.0 * R;
            else if (PD==2)   coeff = M_PI * R * R;
            else              coeff = (4.0/3.0) * M_PI * R * R * R;

            const double den  = 1.0 - coeff * c;
            const double fac  = -(E * coeff * c) / den;
            const double dfac = -(E * coeff) / (den * den);

            sig.setZero();      sig.diagonal().array()     = fac;
            dsig_dc.setZero();  dsig_dc.diagonal().array() = dfac;
        }

// residuals
        const double    R1_1     = (c - cn) / dt;
        const auto      R1_2     = (-c) * v;           // PD x 1
        const auto      R2_1     = (xi * c) * v;       // PD x 1
        const auto&     R2_2     = sig;

// assemble R1
        for (int I=0; I<NPE1; ++I) {
            const double term = R1_1 * N1v(I) + R1_2.dot(GradN1v.row(I));
            R1(I) += term * JxW;
        }

// assemble R2
        for (int I=0; I<NPE2; ++I) {
            // PD x 1:  N2(I)*R2_1  +  sig * GradN2(I)^T
            Eigen::VectorXd acc = R2_1 * N2v(I) + R2_2 * GradN2v.row(I).transpose();
            R2.segment(I*PD, PD).noalias() += acc * JxW;
        }

// K11 (keep only nonzero terms)
        for (int I=0; I<NPE1; ++I) {
            const double aI = N1v(I);
            const double bI = GradN1v.row(I).dot(-v);   // = dR1_2_dc · GradN1(I)
            for (int J=0; J<NPE1; ++J) {
                const double K11_1 = (aI * N1v(J)) / dt;
                const double K11_3 = bI * N1v(J);
                K11(I,J) += (K11_1 + K11_3) * JxW;
            }
        }

// K12 (only K12_3 remains)
        for (int I=0; I<NPE1; ++I) {
            const Eigen::RowVectorXd gI = GradN1v.row(I); // 1xPD
            for (int J=0; J<NPE2; ++J) {
                // PD x 1: (-c I)^T * (gI^T * N2(J)) = -c * gI^T * N2(J)
                Eigen::RowVectorXd row = (-c) * gI * (N2v(J) * JxW);
                K12.block(I, J*PD, 1, PD).noalias() += row;
            }
        }

// K21 (K21_1 + K21_3)
        for (int I=0; I<NPE2; ++I) {
            const double aI = N2v(I);
            const Eigen::VectorXd gI = GradN2v.row(I).transpose(); // PDx1
            for (int J=0; J<NPE1; ++J) {
                // PD x 1
                Eigen::VectorXd K = (xi * v) * (aI * N1v(J)) + (dsig_dc * gI) * (N1v(J));
                K21.block(I*PD, J, PD, 1).noalias() += K * JxW;
            }
        }

// K22 (only K22_1 remains)
        for (int I=0; I<NPE2; ++I) {
            for (int J=0; J<NPE2; ++J) {
                // PDxPD: xi*c*I * (N2(I)*N2(J))
                const double s = xi * c * N2v(I) * N2v(J) * JxW;
                K22.block(I*PD, J*PD, PD, PD).diagonal().array() += s;
            }
        }



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




// --- helpers (local to this file) ---
static void printVecInts(const Eigen::VectorXd& v) {
    std::cout << "[";
    for (int i = 0; i < v.size(); ++i) {
        int val = static_cast<int>(std::llround(v(i)));
        std::cout << val << (i + 1 < v.size() ? "," : "");
    }
    std::cout << "]";
}

static void printVecDoubles(const Eigen::VectorXd& v) {
    std::cout << "[";
    for (int i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v(i)
                  << (i + 1 < v.size() ? "," : "");
    }
    std::cout << "]";
}

static void printMatrixRows(const Eigen::MatrixXd& M) {
    if (M.size() == 0) { std::cout << "[]"; return; }
    std::cout << "[";
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            std::cout << std::fixed << std::setprecision(4) << M(r, c);
            if (c + 1 < M.cols()) std::cout << ",";
        }
        if (r + 1 < M.rows()) std::cout << ";";
    }
    std::cout << "]";
}

static void printStdVec(const std::vector<double>& a) {
    std::cout << "[";
    for (size_t i = 0; i < a.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << a[i]
                  << (i + 1 < a.size() ? "," : "");
    }
    std::cout << "]";
}

// --- pretty “Property  Value” line ---
static void prop(const std::string& name, const std::function<void()>& printer) {
    std::cout << std::left << std::setw(10) << name << " ";
    printer();
    std::cout << "\n";
}

// --- instance method (so you can call EL[k].disp()) ---
void element::disp() const {
    std::cout << "EL(" << (Nr + 1) << ", 1)\n";  // mimic MATLAB 1-based display header

    prop("Nr",     [&]{ std::cout << (Nr + 1); });
    prop("NdL1",   [&]{ printVecInts(NdL1); });
    prop("NdL2",   [&]{ printVecInts(NdL2); });
    prop("NPE1",   [&]{ std::cout << NPE1; });
    prop("NPE2",   [&]{ std::cout << NPE2; });
    prop("deg1",   [&]{ std::cout << deg1; });
    prop("deg2",   [&]{ std::cout << deg2; });

    // Geometry and fields (printed compactly, row/semicolon format like MATLAB)
    prop("X",      [&]{ printMatrixRows(X); });
    prop("C",      [&]{ printVecDoubles(C); });
    prop("V",      [&]{ printMatrixRows(V); });
    prop("x",      [&]{ printMatrixRows(x); });
    prop("c",      [&]{ printVecDoubles(c); });
    prop("v",      [&]{ printMatrixRows(v); });
    prop("xn",      [&]{ printMatrixRows(xn); });
    prop("cn",      [&]{ printVecDoubles(cn); });
    prop("vn",      [&]{ printMatrixRows(vn); });


    // Optional extras (handy for quick checks)
    prop("parameters", [&]{
        if (parameters.empty()) std::cout << "[]";
        else printStdVec(parameters);
    });
    prop("NGP",    [&]{ std::cout << NGP; });
    prop("GP",     [&]{ printMatrixRows(GP); });
    prop("PD",     [&]{ std::cout << PD; });
}

// --- free function to print by index from a std::vector<element> ---
void dispEL(const std::vector<element>& EL, int index_one_based) {
    int idx = index_one_based - 1;
    if (idx < 0 || idx >= static_cast<int>(EL.size())) {
        std::cerr << "dispEL: index out of range (" << index_one_based << ")\n";
        return;
    }
    EL[idx].disp();
}
