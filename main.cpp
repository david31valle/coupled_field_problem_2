#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <tuple>            // for std::tie
#include "Eigen/Dense"
#include "preprocess/mesh.hpp"


int main() {
    // --- Problem setup ---
    int problem_dimension = 3;
    std::array<int,2> element_order = {1, 1};  // only element_order[0] used in 1D

    int domain_size = 1;
    int partition   = 10;

    // (Other simulation parameters can go here, if needed)
    std::string initial_density           = "Two-Bubble";
    double      initial_cell_density      = 1.0;
    double      cell_density_perturbation = 0.05;
    double      young_modulus             = 1.0;
    double      cell_radius               = 1.0;
    double      friction_coefficient      = 10.0;

    Eigen::VectorXd parameters(3);
    parameters << young_modulus,
                  cell_radius,
                  cell_density_perturbation;

    double T                       = 1e4;
    double dt                      = 0.02;
    std::string time_increment     = "Adaptive";
    double      time_factor        = 0.02;
    int         max_iter           = 10;
    double      tol                = 1e-9;
    std::string boundary_condition = "PBC";
    std::string corners            = "Free";
    std::string GP_vals            = "On";
    std::string plot_mesh          = "Off";

    auto [nl, el] = generate_mesh(/*PD=*/0, domain_size, partition,
                                                                element_order[0], problem_dimension);

    return 0;
}