//
// Created by Maitreya Limkar on 17-02-2025.
//

#ifndef MESH_2D_HPP
#define MESH_2D_HPP

#include "../Eigen/Dense"
#include <vector>

// Type aliases for clarity
using NodeList_2D = Eigen::MatrixXd;
using ElementList = Eigen::MatrixXd;

// Mesh_2D class declaration
class Mesh_2D {
public:
    // Constructor
    Mesh_2D(double domain_size, int partition, const std::vector<int>& element_orders);

    // Main methods
    void generateMesh();
    void printMesh() const;

    // Accessors
    [[nodiscard]] NodeList_2D getNodeList() const;
    [[nodiscard]] std::vector<ElementList> getElementLists() const;

private:
    // Input parameters
    double domain_size;
    int partition;
    std::vector<int> element_orders;

    // Storage for mesh
    std::vector<NodeList_2D> node_lists;        // Nodes for each element order
    std::vector<ElementList> element_lists;  // Elements for each order
    NodeList_2D merged_node_list;               // Combined node list

    // Helper methods
    void generateIndividualMesh(int degree, NodeList_2D& nl, ElementList& el) const;
    void mergeNodeLists();
    void updateElementLists();
};

#endif //MESH_2D_HPP
