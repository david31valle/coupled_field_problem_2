//
// Created by Maitreya Limkar on 17-02-2025.
//

#ifndef MESH_1D_HPP
#define MESH_1D_HPP
#include "../Eigen/Dense"
#include <vector>

class Mesh_1D {
public:
    using NodeList_1D = Eigen::VectorXd;
    using ElementList = Eigen::MatrixXd;

    Mesh_1D(double domain_size, int partition, const std::vector<int>& element_orders);
    void generateMesh();
    void printMesh() const;
    [[nodiscard]] NodeList_1D getNodeList() const;
    [[nodiscard]] std::vector<ElementList> getElementLists() const;

private:
    double domain_size;
    int partition;
    std::vector<int> element_orders;
    std::vector<NodeList_1D> node_lists;
    std::vector<ElementList> element_lists;
    NodeList_1D merged_node_list;

    void generateIndividualMesh(int degree, NodeList_1D& nl, ElementList& el) const;
    void mergeNodeLists();
    void updateElementLists();
};

#endif //MESH_1D_HPP
