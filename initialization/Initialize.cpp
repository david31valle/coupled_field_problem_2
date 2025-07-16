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

        std::vector<int> ElL_1_vec, ElL_2_vec;
        for (int el = 0; el < NoE; ++el) {
            if ((el_1.row(el).array() == i + 1).any()) ElL_1_vec.push_back(el);
            if ((el_2.row(el).array() == i + 1).any()) ElL_2_vec.push_back(el);
        }

        Eigen::MatrixXd ElL_1(ElL_1_vec.size(), 1);
        Eigen::MatrixXd ElL_2(ElL_2_vec.size(), 1);
        for (size_t k = 0; k < ElL_1_vec.size(); ++k) ElL_1(k, 0) = ElL_1_vec[k];
        for (size_t k = 0; k < ElL_2_vec.size(); ++k) ElL_2(k, 0) = ElL_2_vec[k];

        double C = 0.0;
        Eigen::VectorXd V = Eigen::VectorXd::Zero(PD);

        if (field(0)) {
            densityCounter++;
            if (Initial_density == "Random") {
                C = C_initial + distribution(generator);
            } else if (Initial_density == "Sin") {
                C = 3 + 0.1 * sin(2 * M_PI * X(0));
            } else if (Initial_density == "Bubble") {
                Eigen::VectorXd center = Eigen::VectorXd::Constant(PD, domain_size / 2.0);
                double R = domain_size / 4.0;
                if ((X - center).squaredNorm() - R * R < tol) C = C_initial + C_perturb;
                else C = C_initial;
            } else if (Initial_density == "Preload") {
                C = Density(densityCounter - 1);
            }
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
