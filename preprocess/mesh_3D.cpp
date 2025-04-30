//
// Created by Maitreya Limkar on 18-02-2025.
//

#include "mesh_3D.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Custom hash function for std::tuple (for unique coordinate storage)
struct TupleHash {
    template <typename T1, typename T2, typename T3>
    std::size_t operator()(const std::tuple<T1, T2, T3>& t) const {
        return std::hash<T1>()(std::get<0>(t))
               ^ (std::hash<T2>()(std::get<1>(t)) << 1)
               ^ (std::hash<T3>()(std::get<2>(t)) << 2);
    }
};

// Constructor
Mesh_3D::Mesh_3D(double domain_size, int partition, const std::vector<int>& element_orders)
    : domain_size(domain_size), partition(partition), element_orders(element_orders) {}

// Main mesh generation method
void Mesh_3D::generateMesh() {
    for (const auto& degree : element_orders) {
        NodeList_3D nl;
        ElementList el;
        generateIndividualMesh(degree, nl, el);
        node_lists.push_back(nl);
        element_lists.push_back(el);
    }

    if (element_orders.size() > 1) {
        mergeNodeLists();
        updateElementLists();
    } else {
        merged_node_list = node_lists[0];
    }
}

// Generate nodes and elements for a given degree
void Mesh_3D::generateIndividualMesh(int degree, NodeList_3D& nl, ElementList& el) const {
    int num_nodes_per_dim = degree * partition + 1;  // Total nodes along each dimension
    int num_nodes = num_nodes_per_dim * num_nodes_per_dim * num_nodes_per_dim;
    int num_elements = partition * partition * partition;  // Total number of elements

    double dx = domain_size / (degree * partition);  // Node spacing

    // Generate the node list as an Eigen matrix (each row: [x, y, z])
    nl.resize(num_nodes, 3);
    int index = 0;
    for (int k = 0; k < num_nodes_per_dim; ++k) {
        for (int j = 0; j < num_nodes_per_dim; ++j) {
            for (int i = 0; i < num_nodes_per_dim; ++i) {
                nl(index, 0) = i * dx;  // X-coordinate
                nl(index, 1) = j * dx;  // Y-coordinate
                nl(index, 2) = k * dx;  // Z-coordinate
                ++index;
            }
        }
    }

    // Generate element connectivity for linear hexahedral elements (8 nodes per element)
    el.resize(num_elements, 8);
    int elem_idx = 0;
    for (int k = 0; k < partition; ++k) {
        for (int j = 0; j < partition; ++j) {
            for (int i = 0; i < partition; ++i) {
                // Using 1-based indexing for node numbering:
                int n1 = k * num_nodes_per_dim * num_nodes_per_dim + j * num_nodes_per_dim + i + 1; // Bottom-left-front
                int n2 = n1 + 1;                                                                    // Bottom-right-front
                int n4 = n1 + num_nodes_per_dim;                                                    // Top-left-front
                int n3 = n4 + 1;                                                                    // Top-right-front
                int n5 = n1 + num_nodes_per_dim * num_nodes_per_dim;                                // Bottom-left-back
                int n6 = n2 + num_nodes_per_dim * num_nodes_per_dim;                                // Bottom-right-back
                int n8 = n4 + num_nodes_per_dim * num_nodes_per_dim;                                // Top-left-back
                int n7 = n3 + num_nodes_per_dim * num_nodes_per_dim;                                // Top-right-back

                el(elem_idx, 0) = n5;
                el(elem_idx, 1) = n6;
                el(elem_idx, 2) = n2;
                el(elem_idx, 3) = n1;
                el(elem_idx, 4) = n8;
                el(elem_idx, 5) = n7;
                el(elem_idx, 6) = n3;
                el(elem_idx, 7) = n4;
                ++elem_idx;
            }
        }
    }
}

// Merge multiple node lists without duplication
void Mesh_3D::mergeNodeLists() {
    std::vector<std::tuple<double, double, double>> combined_nodes;
    for (const auto& nl : node_lists) {
        for (int i = 0; i < nl.rows(); ++i) {
            combined_nodes.emplace_back(nl(i, 0), nl(i, 1), nl(i, 2));
        }
    }

    std::sort(combined_nodes.begin(), combined_nodes.end());
    combined_nodes.erase(std::unique(combined_nodes.begin(), combined_nodes.end()), combined_nodes.end());

    merged_node_list.resize(combined_nodes.size(), 3);
    for (size_t i = 0; i < combined_nodes.size(); ++i) {
        merged_node_list(i, 0) = std::get<0>(combined_nodes[i]);
        merged_node_list(i, 1) = std::get<1>(combined_nodes[i]);
        merged_node_list(i, 2) = std::get<2>(combined_nodes[i]);
    }
}

// Update element indices based on merged node list
void Mesh_3D::updateElementLists() {
    std::unordered_map<std::tuple<double, double, double>, int, TupleHash> node_map;
    for (int i = 0; i < merged_node_list.rows(); ++i) {
        node_map[{merged_node_list(i, 0), merged_node_list(i, 1), merged_node_list(i, 2)}] = i + 1; // 1-based indexing
    }

    // For each element connectivity matrix (built as an Eigen matrix)
    for (size_t k = 0; k < element_lists.size(); ++k) {
        Eigen::MatrixXd &el = element_lists[k];
        int rows = el.rows();
        int cols = el.cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int old_index = static_cast<int>(el(i, j));
                double z = node_lists[k](old_index - 1, 0);
                double y = node_lists[k](old_index - 1, 1);
                double x = node_lists[k](old_index - 1, 2);
                int new_index = node_map[{x, y, z}];
                el(i, j) = new_index;
            }
        }
    }
}

// Print mesh (for debugging)
void Mesh_3D::printMesh() const {
    //std::cout << "Nodes (x, y, z):\n" << merged_node_list << "\n";
    for (size_t i = 0; i < element_lists.size(); ++i) {
        std::cout << "Elements (Order " << element_orders[i] << "):\n";
        std::cout << element_lists[i] << "\n";
    }
}

// Accessors
NodeList_3D Mesh_3D::getNodeList() const { return merged_node_list; }
std::vector<ElementList> Mesh_3D::getElementLists() const { return element_lists; }