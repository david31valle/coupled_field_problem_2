#include "mesh.hpp"

std::pair<Eigen::MatrixXd , Eigen::MatrixXd> generate_mesh(int domain_size, int partition, int element_order, int problem_dimension){
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generated_mesh;
    try {

        switch (problem_dimension) {
            case 1: {
                //------------ 1D Mesh ------------
                //std::cout << "Generating 1D Mesh...\n";
                std::vector<int> element_orders = {element_order};

                // Create and generate the 1D mesh
                Mesh_1D mesh(domain_size, partition, element_orders);
                mesh.generateMesh();
                //mesh.printMesh();

                generated_mesh.first=mesh.getNodeList();
                generated_mesh.second=mesh.getElementLists()[0];
                break;
            }

            case 2: {
                //------------ 2D Mesh ------------
                //std::cout << "Generating 2D Mesh...\n";
                std::vector<int> element_orders = {element_order, element_order};

                Mesh_2D mesh(domain_size, partition, element_orders);
                mesh.generateMesh();
                //mesh.printMesh();

                generated_mesh.first = mesh.getNodeList();
                generated_mesh.second = mesh.getElementLists()[0];
                break;
            }

            case 3: {
                //std::cout << "Generating 3D Mesh...\n";
                std::vector<int> element_orders = {element_order, element_order, element_order};

                Mesh_3D mesh(domain_size, partition, element_orders);
                mesh.generateMesh();
                //mesh.printMesh();

                Eigen::Matrix nl=mesh.getNodeList();
                nl.col(0).swap(nl.col(2));
                generated_mesh.first = nl;
                generated_mesh.second = mesh.getElementLists()[0];
                break;
            }

            default:
                throw std::invalid_argument("Invalid problem dimension! Use 1 (1D), 2 (2D), or 3 (3D).");
        }

        //std::cout << "Mesh generation complete.\n";
        return generated_mesh;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return generated_mesh;
    }

}

