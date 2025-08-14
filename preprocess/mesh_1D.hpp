//
// Created by Maitreya Limkar on 17-02-2025.
//

#pragma once

#include "../Eigen/Dense"
#include <vector>
#include <tuple>
#include <array>

class Mesh_1D {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXi;

    struct Result {
        Vector NL;                 // unified node coordinates (column vector)
        std::vector<Matrix> EL;    // each EL uses 1-based node IDs (MATLAB parity)
    };

    Mesh_1D() = default;

    Result generate(double domain_size,
                    int partition,
                    const std::vector<int>& element_orders) const;

private:
    static void individual(double domain_size,
                           int partition,
                           int degree,
                           Vector& NL,
                           Matrix& EL);
};

// Overloads to mirror MATLAB nargout behavior
std::pair<Mesh_1D::Vector, Mesh_1D::Matrix>
mesh_1D(int PD, double domain_size, int partition, int order, bool plot_mesh=false);

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int PD, double domain_size, int partition,
        const std::array<int,2>& order, bool plot_mesh=false);

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int PD, double domain_size, int partition,
        const std::array<int,3>& order, bool plot_mesh=false);

void printMesh1D(const Mesh_1D::Vector& NL,
                 const Mesh_1D::Matrix& EL);