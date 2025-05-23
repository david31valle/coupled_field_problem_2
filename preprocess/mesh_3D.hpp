//
// Created by Maitreya Limkar on 17-02-2025.
//

#pragma once

#include "../Eigen/Dense"
#include <vector>
#include <tuple>
#include <array>

class Mesh_3D {
public:
    using NodeMat = Eigen::MatrixXd;   // (#nodes) x 3
    using ElemMat = Eigen::MatrixXi;   // (#elements) x ((deg+1)^3)

    struct Result {
        NodeMat           NL;  // merged nodes
        std::vector<ElemMat> EL; // one entry per degree
    };

    // Build and merge meshes for each degree in element_orders
    Result generate(double domain_size,
                    int partition,
                    const std::vector<int>& element_orders) const;

private:
    // Build a single mesh with 0‐based indexing
    void individual(double domain_size,
                    int partition,
                    int degree,
                    NodeMat& NL,
                    ElemMat& EL) const;
};

// MATLAB‐style overloads
std::pair<Mesh_3D::NodeMat,Mesh_3D::ElemMat>
mesh_3D(int PD, double domain_size, int partition, int order);

std::tuple<Mesh_3D::NodeMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat>
mesh_3D(int PD, double domain_size, int partition,
        const std::array<int,2>& orders);

std::tuple<Mesh_3D::NodeMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat>
mesh_3D(int PD, double domain_size, int partition,
        const std::array<int,3>& orders);

// Debug printer
void printMesh3D(const Mesh_3D::NodeMat& NL,
                 const Mesh_3D::ElemMat& EL);