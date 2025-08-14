#include "mesh_1D.hpp"
#include <algorithm>
#include <iostream>

// Implementation of Mesh_1D::generate
Mesh_1D::Result Mesh_1D::generate(double domain_size,
                                  int partition,
                                  const std::vector<int>& element_orders) const
{
    // Compute individual NL and EL for each degree
    std::vector<Vector> NL_list;
    std::vector<Matrix> EL_list;
    NL_list.reserve(element_orders.size());
    EL_list.reserve(element_orders.size());

    for (int deg : element_orders) {
        Vector NL_i;
        Matrix EL_i;
        individual(domain_size, partition, deg, NL_i, EL_i);
        NL_list.push_back(std::move(NL_i));
        EL_list.push_back(std::move(EL_i));
    }

    // Union (sorted unique) of all node coordinates
    std::vector<double> all_nodes;
    size_t total = 0;
    for (const auto& NL_i : NL_list) total += static_cast<size_t>(NL_i.size());
    all_nodes.reserve(total);

    for (const auto& NL_i : NL_list) {
        for (int i = 0; i < NL_i.size(); ++i) all_nodes.push_back(NL_i(i));
    }
    std::sort(all_nodes.begin(), all_nodes.end());
    all_nodes.erase(std::unique(all_nodes.begin(), all_nodes.end()), all_nodes.end());

    // Build Result: NL
    Mesh_1D::Result res;
    res.NL = Eigen::Map<const Vector>(all_nodes.data(), static_cast<int>(all_nodes.size()));

    // Remap each element connectivity to unified, keeping 1-based IDs
    for (size_t m = 0; m < EL_list.size(); ++m) {
        const auto& NL_i = NL_list[m];
        const auto& E_old = EL_list[m];
        Matrix E_new = E_old;

        // Precompute old(0-based) -> new(1-based) node ID map
        std::vector<int> map_old_to_new(static_cast<size_t>(NL_i.size()));
        for (int j = 0; j < NL_i.size(); ++j) {
            double coord = NL_i(j);
            auto it = std::lower_bound(all_nodes.begin(), all_nodes.end(), coord);
            // since coord came from all_nodes, it must exist
            int pos = static_cast<int>(std::distance(all_nodes.begin(), it));
            map_old_to_new[static_cast<size_t>(j)] = pos + 1; // 1-based
        }

        for (int e = 0; e < E_old.rows(); ++e) {
            for (int n = 0; n < E_old.cols(); ++n) {
                int old1 = E_old(e, n);     // 1-based index in the per-degree NL
                int old0 = old1 - 1;        // convert to 0-based to look up the map
                E_new(e, n) = map_old_to_new[static_cast<size_t>(old0)];
            }
        }
        res.EL.push_back(std::move(E_new));
    }

    return res;
}

// Implementation of Mesh_1D::individual
void Mesh_1D::individual(double domain_size,
                         int partition,
                         int degree,
                         Vector& NL,
                         Matrix& EL)
{
    const int p   = partition;
    const int NoN = degree * p + 1;
    const int NoE = p;
    const int NPE = degree + 1;

    NL.resize(NoN);
    EL.resize(NoE, NPE);

    const double dx = domain_size / static_cast<double>(degree * p);
    for (int i = 0; i < NoN; ++i)
        NL(i) = i * dx;

    // Build 1-based connectivity (MATLAB parity)
    for (int e = 0; e < NoE; ++e) {
        for (int n = 0; n < NPE; ++n) {
            if (e == 0) {
                if (n == 0)      EL(e, n) = 1;
                else             EL(e, n) = EL(e, n - 1) + 1;
            } else {
                if (n == 0)      EL(e, n) = EL(e - 1, NPE - 1);
                else             EL(e, n) = EL(e, n - 1) + 1;
            }
        }
    }
}

void printMesh1D(const Mesh_1D::Vector& NL,
                 const Mesh_1D::Matrix& EL)
{
    std::cout << "=== 1D Mesh ===\n";
    std::cout << "Nodes (" << NL.size() << "):\n  ";
    for (int i = 0; i < NL.size(); ++i) {
        std::cout << NL(i) << (i + 1 < NL.size() ? ", " : "\n\n");
    }

    std::cout << "Elements (" << EL.rows() << "):\n";
    for (int e = 0; e < EL.rows(); ++e) {
        std::cout << "  [ ";
        for (int j = 0; j < EL.cols(); ++j) {
            std::cout << EL(e, j)
                      << (j + 1 < EL.cols() ? " , " : " ]\n");
        }
    }
    std::cout << "===============\n";
}

// Free-function definitions (unchanged API)
std::pair<Mesh_1D::Vector, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition, int order, bool /*plot_mesh*/)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order});
    return {res.NL, res.EL[0]};
}

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition,
        const std::array<int,2>& order, bool /*plot_mesh*/)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order[0], order[1]});
    return {res.NL, res.EL[0], res.EL[1]};
}

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition,
        const std::array<int,3>& order, bool /*plot_mesh*/)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order[0], order[1], order[2]});
    return {res.NL, res.EL[0], res.EL[1], res.EL[2]};
}