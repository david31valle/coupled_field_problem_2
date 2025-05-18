//
// Created by Maitreya Limkar on 17-02-2025.
//

#pragma once

#include "../Eigen/Dense"
#include <vector>
#include <tuple>
#include <array>

class Mesh_2D {
public:
    using NodeMat = Eigen::MatrixXd;   // (#nodes) x 2 matrix of (x,y)
    using ElemMat = Eigen::MatrixXi;   // (#elements) x ((deg+1)^2) connectivity

    struct Result {
        NodeMat           NL;   // global node list
        std::vector<ElemMat> EL; // one connectivity matrix per requested order
    };

    // Build and merge all requested polynomial‐order meshes
    Result generate(double domain_size,
                    int partition,
                    const std::vector<int>& element_orders) const;

private:
    // Build a single degree-`degree` mesh (0-based indexing)
    void individual(double domain_size,
                    int partition,
                    int degree,
                    NodeMat& NL,
                    ElemMat& EL) const;
};

// MATLAB‐style free‐function overloads
std::pair<Mesh_2D::NodeMat,Mesh_2D::ElemMat>
mesh_2D(int PD, double domain_size, int partition, int order);

std::tuple<Mesh_2D::NodeMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat>
mesh_2D(int PD, double domain_size, int partition,
        const std::array<int,2>& orders);

std::tuple<Mesh_2D::NodeMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat>
mesh_2D(int PD, double domain_size, int partition,
        const std::array<int,3>& orders);

// Printer (for debugging)
void printMesh2D(const Mesh_2D::NodeMat& NL,
                 const Mesh_2D::ElemMat& EL);