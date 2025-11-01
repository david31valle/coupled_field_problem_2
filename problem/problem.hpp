#ifndef PROBLEM_COUPLED_HPP
#define PROBLEM_COUPLED_HPP

#include <string>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <filesystem>
#include <stdexcept>

#include "../Eigen/Dense"
#include "../Eigen/Sparse"
#include "../Eigen/SparseCholesky"
#include "../Eigen/SparseQR"
#include "../Eigen/SparseLU"
#include "../Eigen/IterativeLinearSolvers"
#include <Eigen/OrderingMethods>

#include "../node/node.hpp"
#include "../element/element.hpp"

class problem_coupled {
public:
    // === Input parameters ===
    int PD;                                     // Problem dimension
    int domain_size;                            // Domain width/height
    std::string BC_type;                        // Boundary condition type (e.g., "Dirichlet")
    std::string Initial_density;                // e.g., "Homogeneous", "Gaussian"
    std::string GP_vals;                        // "On" or "Off"
    std::string time_incr_method;               // "Adaptive" or "Fixed"
    std::vector<double> parameters;             // Physical or model parameters
    Eigen::Vector2i field_dim;                  // Number of fields per node: [scalar, vector]
    std::vector<int> element_order;             // Order of interpolation: [scalar field, vector field]

    // === Time-stepping ===
    double T;           // Final time
    double dt;          // Initial time step
    double time_factor; // Adaptivity factor
    int max_iter;       // Newton iterations per time step
    double tol;         // Convergence tolerance
    double t = 0.0;     // Current simulation time
    int counter = 1;    // Time step counter

    // === Output file ===
    std::string filename;

    // === Mesh and fields ===
    std::vector<Node> Node_List;
    std::vector<element> Element_List;

    // === DOF tracking ===
    int DOFs = 0;
    int GP_DOFs = 0;

    // === Global solution structures (standard DOFs) ===
    Eigen::VectorXd Rtot;
    Eigen::SparseMatrix<double> Ktot;

    // === GP solve: multi-RHS (columns = NGP_val) ===
    Eigen::MatrixXd Rtot_GP;                    // size: GP_DOFs × NGP_val
    Eigen::SparseMatrix<double> Ktot_GP;        // size: GP_DOFs × GP_DOFs

    // === Linear solver ===
    std::string Solver;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> slu_;
    bool slu_pattern_init_ = false;
    bool BCset;

    // === Constructor ===
    problem_coupled(int PD,
                    std::vector<Node>& NL,
                    std::vector<element>& EL,
                    int domain_size,
                    const std::string& BC_type,
                    const std::string Corners,
                    const std::string& Initial_density,
                    const std::vector<double>& parameters,
                    const std::vector<int>& element_order,
                    const Eigen::Vector2i& field_dim,
                    const std::string& GP_vals,
                    const std::string& time_incr_method,
                    double T,
                    double dt,
                    double time_factor,
                    int max_iter,
                    double tol);

private:
    // --- Setup & bookkeeping ---
    void Assign_BC(const std::string Corners);
    void Assign_GP_DOFs();
    void Assign_DOF_DBC();
    void Assign_DOF_PBC(const std::vector<int>& NLC,
                        const std::vector<int>& NLS,
                        const std::vector<int>& NLM,
                        const std::vector<int>& NLP);
    void problem_info();

    // --- Assembly ---
    void assemble(double dt);
    void assemble_GP(double dt);

    // --- Updates ---
    void update(const Eigen::VectorXd& dx);
    void update_GP(const Eigen::MatrixXd& dx_gp);   // <-- changed to MatrixXd
    void update_time();
    void downdate_time();

    // --- Reporting ---
    void post_process();
    void output_step_info();

    // --- Utilities ---
    Eigen::MatrixXd Get_all_velocity();
    Eigen::VectorXd Residual(double dt);
    std::pair<double, double> calculate_max_min_difference();
    double calculate_overall_density();

    // --- Solve loop ---
    void solve();

    // --- Linear solve helper ---
    Eigen::VectorXd solve_dx_(Eigen::SparseMatrix<double>& Ktot,
                              const Eigen::VectorXd& R,
                              bool verbose = true);

    Eigen::VectorXd dx_prev;
};

#endif // PROBLEM_COUPLED_HPP