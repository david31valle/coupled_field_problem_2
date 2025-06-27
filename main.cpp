#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include "preprocess/mesh.hpp"
#include "initialization/initialize.hpp"


int main() {
    // --- Problem setup ---
    int problem_dimension = 2;
    std::vector<int> element_order = { 2,1};  // {1} for 1D; {1, 1} for 2D, {1, 1, 1} for 3D

    int domain_size = 1;
    int partition   = 5;

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

    auto [nl, element_lists] = generate_mesh(/*PD=*/0, domain_size, partition,
                                                                element_order, problem_dimension);

    Initialize(problem_dimension, nl, element_lists[0], element_lists[1], domain_size, initial_cell_density, cell_density_perturbation, initial_density, )


    return 0;
}