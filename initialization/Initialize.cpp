#include "Initialize.hpp"
#include <random>

#include "Initialize.hpp"
#include <random>

std::pair<std::vector<Node>, std::vector<element>>
Initialize(int PD, const Eigen::MatrixXd &nl, const Eigen::MatrixXd &el_1, const Eigen::MatrixXd &el_2,
           double domain_size, double C_initial, double C_perturb, const Eigen::VectorXd &Density,
           const Eigen::VectorXd &Velocity, const std::string &Initial_density,
           const std::vector<int> &element_order, const Eigen::Vector2i &field_dim,
           const std::vector<double> &parameters) {

    const double tol = 1e-6;
    int NoN = nl.rows();
    int NoE = el_1.rows();
    int NGP = 0;

    switch (PD) {
        case 1: NGP = std::vector<int>{2, 3, 4, 5}[std::max(element_order[0], element_order[1]) - 1]; break;
        case 2: NGP = std::vector<int>{4, 9, 16, 25}[std::max(element_order[0], element_order[1]) - 1]; break;
        case 3: NGP = std::vector<int>{8, 27, 64, 125}[std::max(element_order[0], element_order[1]) - 1]; break;
    }

    std::vector<Node> NL;
    NL.reserve(NoN);

    int densityCounter = 0;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-C_perturb, C_perturb);

    for (int i = 0; i < NoN; ++i) {
        Eigen::VectorXd X = nl.row(i);

        Eigen::Vector2d field;
        field << (el_1.array() == (i + 1)).any(), (el_2.array() == (i + 1)).any();

        int node_id = i + 1;  // MATLAB nodes are 1-based

        std::vector<int> ElL_1_vec = find_linear_indices_eq(el_1, node_id);
        std::vector<int> ElL_2_vec = find_linear_indices_eq(el_2, node_id);

// Pack into Eigen column vectors (double) to match your Node ctor
        Eigen::MatrixXd ElL_1(ElL_1_vec.size(), 1);
        Eigen::MatrixXd ElL_2(ElL_2_vec.size(), 1);
        for (size_t k = 0; k < ElL_1_vec.size(); ++k) ElL_1(k, 0) = static_cast<double>(ElL_1_vec[k]);
        for (size_t k = 0; k < ElL_2_vec.size(); ++k) ElL_2(k, 0) = static_cast<double>(ElL_2_vec[k]);


        double C = 0.0;
        Eigen::VectorXd V = Eigen::VectorXd::Zero(PD);


        if (field(0)) {                 // MATLAB field(1) -> C++ field(0)
            densityCounter++;

            if (Initial_density == "Random") {
                // C = (C_initial - C_perturb) + (2*C_perturb) * rand
                C = C_initial + distribution(generator);

            } else if (Initial_density == "Sin") {
                // C = 3 + 0.1 * sin(2*pi*X)  (use x-coordinate X(0))
                C = 3.0 + 0.1 * std::sin(2.0 * M_PI * X(0));

            } else if (Initial_density == "Bubble") {
                // Center at domain_size/2 in each dimension; radius = domain_size/4
                Eigen::VectorXd center = Eigen::VectorXd::Constant(PD, domain_size / 2.0);
                double R = domain_size / 4.0;

                // Inside sphere (or interval in 1D) test
                if ((X - center).squaredNorm() - R * R < tol)
                    C = C_initial + C_perturb;
                else
                    C = C_initial;

            } else if (Initial_density == "Two-Bubble") {

                // Centers and radius
                const double R = 0.15 * domain_size;

                // Switch on PD just like in MATLAB
                switch (PD) {

                    case 1: {
                        const double center1 = 0.3 * domain_size;
                        const double center2 = 0.7 * domain_size;

                        const double t1 = (X(0) - center1) * (X(0) - center1) - R * R;
                        const double t2 = (X(0) - center2) * (X(0) - center2) - R * R;

                        if ( (t1 < tol) || (t2 < tol) )
                            C = C_initial + C_perturb;
                        else
                            C = C_initial;
                        break;
                    }

                    case 2: {
                        const Eigen::Vector2d center1(0.3 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector2d center2(0.7 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector2d center3(0.3 * domain_size, 0.7 * domain_size);
                        const Eigen::Vector2d center4(0.7 * domain_size, 0.7 * domain_size);

                        const double t1 = (X(0) - center1(0)) * (X(0) - center1(0))
                                          + (X(1) - center1(1)) * (X(1) - center1(1)) - R * R;
                        const double t2 = (X(0) - center2(0)) * (X(0) - center2(0))
                                          + (X(1) - center2(1)) * (X(1) - center2(1)) - R * R;
                        const double t3 = (X(0) - center3(0)) * (X(0) - center3(0))
                                          + (X(1) - center3(1)) * (X(1) - center3(1)) - R * R;
                        const double t4 = (X(0) - center4(0)) * (X(0) - center4(0))
                                          + (X(1) - center4(1)) * (X(1) - center4(1)) - R * R;

                        if ( (t1 < tol) || (t2 < tol) || (t3 < tol) || (t4 < tol) )
                            C = C_initial + C_perturb;
                        else
                            C = C_initial;
                        break;
                    }

                    case 3: {
                        const Eigen::Vector3d center1(0.3 * domain_size, 0.3 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector3d center2(0.7 * domain_size, 0.3 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector3d center3(0.3 * domain_size, 0.7 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector3d center4(0.7 * domain_size, 0.7 * domain_size, 0.3 * domain_size);
                        const Eigen::Vector3d center5(0.3 * domain_size, 0.3 * domain_size, 0.7 * domain_size);
                        const Eigen::Vector3d center6(0.7 * domain_size, 0.3 * domain_size, 0.7 * domain_size);
                        const Eigen::Vector3d center7(0.3 * domain_size, 0.7 * domain_size, 0.7 * domain_size);
                        const Eigen::Vector3d center8(0.7 * domain_size, 0.7 * domain_size, 0.7 * domain_size);

                        const auto t = [&](const Eigen::Vector3d& c){
                            return (X(0) - c(0)) * (X(0) - c(0))
                                   + (X(1) - c(1)) * (X(1) - c(1))
                                   + (X(2) - c(2)) * (X(2) - c(2)) - R * R;
                        };

                        if ( (t(center1) < tol) || (t(center2) < tol) || (t(center3) < tol) || (t(center4) < tol) ||
                             (t(center5) < tol) || (t(center6) < tol) || (t(center7) < tol) || (t(center8) < tol) )
                            C = C_initial + C_perturb;
                        else
                            C = C_initial;
                        break;
                    }

                    default:
                        // If PD is something else, keep the default
                        C = C_initial;
                        break;
                }
            }
            else if (Initial_density == "Preload") {
                // MATLAB uses 1-based indexing; here densityCounter is 1-based after ++
                C = Density(densityCounter - 1);
            }
        }
        if (field(1)) {
            V = Eigen::VectorXd::Zero(PD);
        }

        NL.emplace_back(i, PD, X, Eigen::VectorXd::Constant(1, C), V,
                        ElL_1, ElL_2, field, field_dim);
    }

    std::vector<element> EL;
    EL.reserve(NoE);

    for (int i = 0; i < NoE; ++i) {
        Eigen::VectorXd NdL_1 = el_1.row(i);
        Eigen::VectorXd NdL_2 = el_2.row(i);

        Eigen::MatrixXd X(PD, NdL_2.size());
        Eigen::VectorXd C(NdL_1.size());
        Eigen::MatrixXd V(PD, NdL_2.size());

        for (int j = 0; j < NdL_2.size(); ++j) {
            X.col(j) = NL[NdL_2(j) - 1].X;
            V.col(j) = NL[NdL_2(j) - 1].U.segment(1, PD);
        }

        for (int j = 0; j < NdL_1.size(); ++j)
            C(j) = NL[NdL_1(j) - 1].U(0);

        std::pair<int, int> temp_element_order={element_order[0], element_order[1]};
        EL.emplace_back(i, PD, NdL_1, NdL_2, X, C, V, NGP, temp_element_order, parameters);
    }

    return {NL, EL};
}

// Returns MATLAB-style linear indices (1-based, column-major) where M == value
static std::vector<int> find_linear_indices_eq(const Eigen::MatrixXd& M, int value) {
    const int rows = M.rows();
    const int cols = M.cols();
    std::vector<int> idx;
    idx.reserve(rows * cols);

    for (int c = 0; c < cols; ++c) {          // MATLAB scans columns first
        for (int r = 0; r < rows; ++r) {
            // Cast to int if your matrices store integer IDs in doubles
            if (static_cast<int>(M(r, c)) == value) {
                // MATLAB linear index: r + c*rows + 1  (1-based)
                idx.push_back(r + c * rows + 1);
            }
        }
    }
    return idx;
}
