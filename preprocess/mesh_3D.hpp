//
// Created by Maitreya Limkar on 17-02-2025.
//

#ifndef MESH_3D_HPP
#define MESH_3D_HPP

#include "../Eigen/Dense"
#include <vector>

// Type aliases for clarity
using NodeList_3D = Eigen::MatrixXd;
using ElementList = Eigen::MatrixXd;

// Mesh_3D class declaration
class Mesh_3D {
public:
    // Constructor
    Mesh_3D(double domain_size, int partition, const std::vector<int>& element_orders);

    // Main methods
    void generateMesh();
    void printMesh() const;

    // Accessors
    [[nodiscard]] NodeList_3D getNodeList() const;
    [[nodiscard]] std::vector<ElementList> getElementLists() const;

private:
    // Input parameters
    double domain_size;
    int partition;
    std::vector<int> element_orders;

    // Storage for mesh
    std::vector<NodeList_3D> node_lists;        // Nodes for each element order
    std::vector<ElementList> element_lists;     // Elements for each order
    NodeList_3D merged_node_list;               // Combined node list

    // Helper methods
    void generateIndividualMesh(int degree, NodeList_3D& nl, ElementList& el) const;
    void mergeNodeLists();
    void updateElementLists();
};

#endif //MESH_3D_HPP
