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
    using Vector  = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXi;

    struct Result {
        Vector NL;
        std::vector<Matrix> EL;
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