//
// Created by Maitreya Limkar on 17-02-2025.
//

#include "mesh_2D.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Custom hash function for std::pair (for unique coordinate storage)
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ (std::hash<T2>()(p.second) << 1);
    }
};

// Constructor
Mesh_2D::Mesh_2D(double domain_size, int partition, const std::vector<int>& element_orders)
    : domain_size(domain_size), partition(partition), element_orders(element_orders) {}

// Main mesh generation method
void Mesh_2D::generateMesh() {
    for (const auto& degree : element_orders) {
        NodeList_2D nl;
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
void Mesh_2D::generateIndividualMesh(int degree, NodeList_2D& nl, ElementList& el) const {
    int num_nodes_per_dim = degree * partition + 1;  // Total nodes per dimension
    int num_elements = partition * partition;        // Total number of elements
    double dx = domain_size / (degree * partition);    // Spacing between nodes

    // Build the node list: one row per node, with two coordinates (x, y)
    nl.resize(num_nodes_per_dim * num_nodes_per_dim, 2);
    int index = 0;
    for (int j = 0; j < num_nodes_per_dim; ++j) {
        for (int i = 0; i < num_nodes_per_dim; ++i) {
            nl(index, 0) = j * dx;  // x-coordinate
            nl(index, 1) = i * dx;  // y-coordinate
            ++index;
        }
    }

    // Build the element connectivity matrix.
    // For a linear quadrilateral, each element has 4 nodes.
    el.resize(num_elements, 4);
    int elem_idx = 0;
    for (int i = 0; i < partition; ++i) {
        for (int j = 0; j < partition; ++j) {
            // 1-based indexing assumed for node numbering:
            int n1 = j * num_nodes_per_dim + i + 1;       // Bottom-left
            int n2 = n1 + 1;                              // Bottom-right
            int n4 = n1 + num_nodes_per_dim;              // Top-left
            int n3 = n4 + 1;                              // Top-right

            el(elem_idx, 0) = n1;
            el(elem_idx, 1) = n4;
            el(elem_idx, 2) = n3;
            el(elem_idx, 3) = n2;
            ++elem_idx;
        }
    }
}

// Merge multiple node lists without duplication
void Mesh_2D::mergeNodeLists() {
    std::vector<std::pair<double, double>> combined_nodes;
    for (const auto& nl : node_lists) {
        for (int i = 0; i < nl.rows(); ++i) {
            combined_nodes.emplace_back(nl(i, 0), nl(i, 1));
        }
    }

    std::sort(combined_nodes.begin(), combined_nodes.end());
    combined_nodes.erase(std::unique(combined_nodes.begin(), combined_nodes.end()), combined_nodes.end());

    merged_node_list.resize(combined_nodes.size(), 2);
    for (size_t i = 0; i < combined_nodes.size(); ++i) {
        merged_node_list(i, 1) = combined_nodes[i].first;
        merged_node_list(i, 0) = combined_nodes[i].second;
    }
}

// Update element indices based on merged node list
void Mesh_2D::updateElementLists() {
    std::unordered_map<std::pair<double, double>, int, PairHash> node_map;
    for (int i = 0; i < merged_node_list.rows(); ++i) {
        node_map[{merged_node_list(i, 0), merged_node_list(i, 1)}] = i + 1; // 1-based indexing
    }

    // Update each element connectivity matrix (which is now an Eigen matrix)
    for (size_t k = 0; k < element_lists.size(); ++k) {
        Eigen::MatrixXd &el = element_lists[k];
        int rows = el.rows();
        int cols = el.cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int old_index = static_cast<int>(el(i, j));
                // Retrieve the corresponding coordinate from the original node list
                double x = node_lists[k](old_index - 1, 0);
                double y = node_lists[k](old_index - 1, 1);
                int new_index = node_map[{x, y}];
                el(i, j) = new_index;
            }
        }
    }
}

// Print mesh (for debugging)
void Mesh_2D::printMesh() const {
    std::cout << "Nodes (x, y):\n" << merged_node_list << "\n";
    for (size_t i = 0; i < element_lists.size(); ++i) {
        std::cout << "Elements (Order " << element_orders[i] << "):\n";
        std::cout << element_lists[i] << "\n";
    }
}

// Accessors
NodeList_2D Mesh_2D::getNodeList() const { return merged_node_list; }
std::vector<ElementList> Mesh_2D::getElementLists() const { return element_lists; }