#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include "preprocess/mesh.hpp"
#include "initialization/initialize.hpp"
#include "utils/utils.hpp"
#include "problem/problem.hpp"
#include <chrono>

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

int main() {
    {
        Timer t("Program run");
        // --- Problem setup ---
        int problem_dimension = 1;
        std::vector<int> element_order = {2, 1};  // {degree_field_1, degree_field_2}
        Eigen::Vector2i field_dim = {1, problem_dimension};

        Eigen::VectorXd Density = readVectorFromFile(
                "C:\\Users\\drva_\\CLionProjects\\coupled_field_problem_2\\data.txt");
        Eigen::VectorXd Velocity = Density;

        int domain_size = 1;
        int partition = 5;

        // (Other simulation parameters can go here, if needed)
        std::string initial_density = "Two-Bubble";
        double initial_cell_density = 1.0;
        double cell_density_perturbation = 0.5;
        double young_modulus = 1.0;
        double cell_radius = 1.0;
        double friction_coefficient = 10.0;

        std::vector<double> parameters = {young_modulus, cell_radius, friction_coefficient};

        double T = 1e4;
        double dt = 0.02;
        std::string time_increment = "Adaptive";
        double time_factor = 2.0;
        int max_iter = 10;
        double tol = 1e-9;
        std::string boundary_condition = "PBC";
        std::string corners = "Free";
        std::string GP_vals = "On";
        std::string plot_mesh = "Off";

        auto [nl, element_lists] = generate_mesh(/*PD=*/0, domain_size, partition,
                                                        element_order, problem_dimension);
//    std::cout<<"nl"<<std::endl;
//    std::cout<<nl<<std::endl;
//    std::cout<<"el"<<std::endl;
//    std::cout<<element_lists[1] <<std::endl;

        auto [NL, EL] = Initialize(problem_dimension, nl, element_lists[0], element_lists[1], domain_size,
                                   initial_cell_density, cell_density_perturbation, Density, Velocity, initial_density,
                                   element_order, field_dim, parameters);
//    EL[0].disp();
//    EL[2].disp();
//    EL[4].disp();
        problem_coupled coupled_problem(problem_dimension, NL, EL, domain_size, boundary_condition, corners,
                                        initial_density, parameters, element_order, field_dim, GP_vals, time_increment,
                                        T, dt, time_factor, max_iter, tol);
    }
    return 0;
}

