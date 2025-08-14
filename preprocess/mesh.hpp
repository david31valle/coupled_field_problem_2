#pragma once

#include "../Eigen/Dense"
#include <vector>
#include <tuple>

class Mesh {
public:
    using MatrixD = Eigen::MatrixXd; // Nodes: N x PD
    using MatrixI = Eigen::MatrixXi; // Elements: Ne x NPE (1-based IDs)

    struct Result {
        MatrixD NL;                 // unified node list
        std::vector<MatrixI> ELs;   // one connectivity per requested order (size 1..3)
    };

    Mesh(int problem_dimension,
         double domain_size,
         int partition,
         std::vector<int> element_orders,
         bool plot_mesh = false);

    Result build() const; // picks 1D / 2D / 3D internally

private:
    int dim_;
    double L_;
    int p_;
    std::vector<int> orders_;
    bool plot_;

    static Result build1D(double L, int p, const std::vector<int>& orders, bool plot);
    static Result build2D(double L, int p, const std::vector<int>& orders, bool plot);
    static Result build3D(double L, int p, const std::vector<int>& orders, bool plot);
};

// This is supposed to be EIL1 and EIL2
std::tuple<Eigen::MatrixXd,  std::vector<Eigen::MatrixXd> >
generate_mesh(int /*PD*/,
              double domain_size,
              int partition,
              const std::vector<int>& element_orders,
              int problem_dimension);