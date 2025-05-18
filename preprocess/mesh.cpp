#include "mesh.hpp"
#include "mesh_1D.hpp"
#include "mesh_2D.hpp"
#include "mesh_3D.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
generate_mesh(int /*PD*/,
              double domain_size,
              int partition,
              int element_order,
              int problem_dimension)
{
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generated_mesh;
    try {
        switch (problem_dimension) {
            case 1: {
                // 1D
                Mesh_1D mesh1d;
                auto res1 = mesh1d.generate(domain_size, partition, {element_order});
                printMesh1D(res1.NL, res1.EL[0]);
                generated_mesh.first  = res1.NL;                        // N×1
                generated_mesh.second = res1.EL[0].cast<double>();       // E×(order+1)
                break;
            }
            case 2: {
                // 2D
                Mesh_2D mesh2d;
                std::vector<int> orders2 = {element_order, element_order};
                auto res2 = mesh2d.generate(domain_size, partition, orders2);
                printMesh2D(res2.NL, res2.EL[0]);
                generated_mesh.first  = res2.NL;                         // N×2
                generated_mesh.second = res2.EL[0].cast<double>();
                break;
            }
            case 3: {
                    Mesh_3D mesh3d;
                    std::vector<int> orders3 = {element_order, element_order, element_order};
                    auto res3 = mesh3d.generate(domain_size, partition, orders3);
                    printMesh3D(res3.NL, res3.EL[0]);
                    generated_mesh.first  = res3.NL;                         // N×3
                    generated_mesh.second = res3.EL[0].cast<double>();
                break;
            }
            default:
                throw std::invalid_argument(
                    "Invalid problem dimension! Use 1 (1D), 2 (2D), or 3 (3D).");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Mesh generation error: " << e.what() << std::endl;
        // generated_mesh will be left empty if error
    }
    return generated_mesh;
}