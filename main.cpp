#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include "preprocess/mesh.hpp"
#include "initialization/initialize.hpp"
#include "utils/utils.hpp"
#include "utils/Config.hpp"
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

int main(int argc, char** argv) {
    Timer t("Program run");
    std::string cfgPath = "config.ini";
    for (int i=1; i<argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind("--config=", 0) == 0) { cfgPath = a.substr(9); }
    }

    try {
        Config cfg = Config::from_ini_required(cfgPath);    // <-- mandatory
        cfg = Config::override_from_argv(argc, argv, std::move(cfg));

        // --- unchanged below, just wire cfg into your existing calls ---
        Eigen::VectorXd Density  = readVectorFromFile(cfg.density_path);
        Eigen::VectorXd Velocity = Density;

        auto [nl, element_lists] = generate_mesh(
                /*PD=*/0, cfg.domain_size, cfg.partition, cfg.element_order, cfg.problem_dimension);

        Eigen::Vector2i field_dim(1, cfg.problem_dimension);

        auto [NL, EL] = Initialize(cfg.problem_dimension, nl,
                                   element_lists[0], element_lists[1],
                                   cfg.domain_size,
                                   cfg.initial_cell_density, cfg.cell_density_perturbation,
                                   Density, Velocity, cfg.initial_density,
                                   cfg.element_order, field_dim, cfg.parameters());

        problem_coupled coupled_problem(cfg.problem_dimension, NL, EL, cfg.domain_size,
                                        cfg.boundary_condition, cfg.corners,
                                        cfg.initial_density, cfg.parameters(),
                                        cfg.element_order, field_dim,
                                        cfg.GP_vals, cfg.time_increment,
                                        cfg.T, cfg.dt, cfg.time_factor, cfg.max_iter, cfg.tol);

    } catch (const std::exception& e) {
        std::cerr << "[Configuration error] " << e.what() << "\n"
                  << "Tip: pass a specific file with --config=path/to/config.ini\n";
        return EXIT_FAILURE;
    }

    return 0;
}
