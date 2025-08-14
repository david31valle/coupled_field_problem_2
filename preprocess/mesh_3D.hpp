//
// Created by Maitreya Limkar on 17-02-2025.
//

#pragma once

#include "../Eigen/Dense"
#include <vector>

class Mesh_3D {
public:
    using MatrixD = Eigen::MatrixXd; // NL: N x 3  (x,y,z)
    using MatrixI = Eigen::MatrixXi; // EL: Ne x NPE (1-based node IDs)

    struct Result {
        MatrixD NL;                 // unified node list (x,y,z)
        std::vector<MatrixI> EL;    // one EL per requested element order
    };

    // element_orders.size() in [1..3]
    Result generate(double domain_size,
                    int partition,
                    const std::vector<int>& element_orders) const;

private:
    static void individual(double domain_size,
                           int partition,
                           int degree,
                           MatrixD& NL,
                           MatrixI& EL);
};

void printMesh3D(const Eigen::MatrixXd& NL,
                 const Eigen::MatrixXi& EL);