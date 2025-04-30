#include <iostream>
#include <string>

#include <fstream>
#include <vector>
#include "Eigen/Dense"

Eigen::VectorXd readVectorFromFile(const std::string& filename);

int main() {
    int problem_dimension =1;
    Eigen::VectorXd density = readVectorFromFile("../data.txt");
    Eigen::VectorXd velocity = density;
    Eigen::VectorXd element_order(2);
    element_order << 1,1;

    Eigen::VectorXd field_dimension(2);
    field_dimension << 1, problem_dimension;

    int domain_size = 1;
    int partition = 50;

    std::string initial_desnity = "Two-Bubble";
    double initial_cell_density = 1.0;
    double cell_density_perturbation = 0.05;

    double young_modulus = 1.0;
    double cell_radius = 1.0;
    double friction_coefficient = 10.0;

    Eigen::VectorXd  parameters(3);
    parameters << young_modulus, cell_radius, cell_density_perturbation;

    double T = 1e4; //Final time
    double dt = 0.02; // Time increment

    std::string time_increment_method = "Adaptive";
    double time_factor = 0.02;

    int max_iter = 10;
    double tol = 1e-9;

    std::string boundary_condition = "PBC";
    std::string corners = "Free";

    std::string GP_vals ="On";

    std::string plot_mesh = "Off";


    return 0;
}



Eigen::VectorXd readVectorFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<double> values;
    double val;

    while (infile >> val) {
        values.push_back(val);
    }

    infile.close();

    Eigen::VectorXd vec(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        vec(i) = values[i];
    }

    return vec;
}