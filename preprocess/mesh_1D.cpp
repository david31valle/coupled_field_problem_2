#include "mesh_1D.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "../Eigen/Dense"

// Constructor
Mesh_1D::Mesh_1D(const double domain_size, const int partition, const std::vector<int>& element_orders)
    : domain_size(domain_size), partition(partition), element_orders(element_orders) {}

// Main function to generate the mesh
void Mesh_1D::generateMesh() {
    for (const auto& degree : element_orders) {
        NodeList_1D nl;
        Eigen::MatrixXd el;  // Use an Eigen matrix directly for element connectivity
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
void Mesh_1D::generateIndividualMesh(const int degree, NodeList_1D& nl, Eigen::MatrixXd& el) const {
    const int num_nodes = degree * partition + 1;
    const int num_elements = partition;
    const int nodes_per_element = degree + 1;

    // Generate node coordinates using Eigen’s LinSpaced
    nl = NodeList_1D::LinSpaced(num_nodes, 0.0, domain_size);

    // Allocate the element connectivity matrix
    el.resize(num_elements, nodes_per_element);

    // Fill the connectivity matrix directly
    for (int i = 0; i < num_elements; ++i) {
        if (i == 0) {
            for (int j = 0; j < nodes_per_element; ++j) {
                el(i, j) = j + 1;
            }
        } else {
            el(i, 0) = el(i - 1, nodes_per_element - 1);  // Start with the last node of the previous element
            for (int j = 1; j < nodes_per_element; ++j) {
                el(i, j) = el(i, j - 1) + 1;
            }
        }
    }
}

// Combine node lists without duplicates
void Mesh_1D::mergeNodeLists() {
    std::vector<double> combined_nodes;
    for (const auto& nl : node_lists) {
        for (int i = 0; i < nl.size(); ++i) {
            combined_nodes.push_back(nl(i));
        }
    }

    std::sort(combined_nodes.begin(), combined_nodes.end());
    combined_nodes.erase(std::unique(combined_nodes.begin(), combined_nodes.end()), combined_nodes.end());
    merged_node_list = Eigen::Map<NodeList_1D>(combined_nodes.data(), combined_nodes.size());
}

// Update element indices based on merged node list
void Mesh_1D::updateElementLists() {
    std::unordered_map<double, int> node_map;
    for (int i = 0; i < merged_node_list.size(); ++i) {
        node_map[merged_node_list(i)] = i + 1;  // 1-based indexing
    }

    // For each individual element list (now an Eigen matrix), update the connectivity
    for (size_t k = 0; k < element_lists.size(); k++) {
        Eigen::MatrixXd &el = element_lists[k];
        int rows = el.rows();
        int cols = el.cols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int old_index = static_cast<int>(el(i, j));
                // Get the corresponding coordinate from the original node list for this element set
                double coord = node_lists[k](old_index - 1);
                // Update using the new global indexing
                int new_index = node_map[coord];
                el(i, j) = new_index;
            }
        }
    }
}

// Print the mesh
void Mesh_1D::printMesh() const {
    std::cout << "Nodes (x):\n" << merged_node_list << "\n\n";

    for (size_t i = 0; i < element_lists.size(); ++i) {
        std::cout << "Element Connectivity (Order " << element_orders[i] << "):\n";
        std::cout << element_lists[i] << "\n\n";
    }
}

// Accessor Methods
Mesh_1D::NodeList_1D Mesh_1D::getNodeList() const {
    return merged_node_list;
}

std::vector<Eigen::MatrixXd> Mesh_1D::getElementLists() const {
    return element_lists;
}