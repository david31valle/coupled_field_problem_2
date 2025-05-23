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

    // Union of all node coordinates
    std::vector<double> all_nodes;
    for (const auto& NL_i : NL_list)
        for (double i : NL_i)
            all_nodes.push_back(i);
    std::sort(all_nodes.begin(), all_nodes.end());
    all_nodes.erase(std::unique(all_nodes.begin(), all_nodes.end()), all_nodes.end());

    // Build Result
    Result res;
    res.NL = Eigen::Map<Vector>(all_nodes.data(), all_nodes.size());

    // Remap each element connectivity
    for (size_t m = 0; m < EL_list.size(); ++m) {
        const auto& NL_i = NL_list[m];
        const auto& E_old = EL_list[m];
        Matrix E_new = E_old;
        for (int e = 0; e < E_old.rows(); ++e) {
            for (int n = 0; n < E_old.cols(); ++n) {
                int old_idx = E_old(e,n);
                double coord = NL_i[old_idx];
                auto it = std::lower_bound(all_nodes.begin(), all_nodes.end(), coord);
                E_new(e,n) = static_cast<int>(std::distance(all_nodes.begin(), it));
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
    int p = partition;
    int NoN = degree*p + 1;
    int NoE = p;
    int NPE = degree + 1;

    NL.resize(NoN);
    EL.resize(NoE, NPE);

    double dx = domain_size / static_cast<double>(degree * p);
    for (int i = 0; i < NoN; ++i)
        NL[i] = i * dx;

    for (int e = 0; e < NoE; ++e) {
        for (int n = 0; n < NPE; ++n) {
            if (e == 0) {
                EL(e,n) = n;
            } else {
                if (n == 0)
                    EL(e,n) = EL(e-1, NPE-1);
                else
                    EL(e,n) = EL(e,n-1) + 1;
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
        std::cout << NL[i] << (i+1<NL.size()? ", " : "\n\n");
    }

    std::cout << "Elements (" << EL.rows() << "):\n";
    for (int e = 0; e < EL.rows(); ++e) {
        std::cout << "  [ ";
        for (int j = 0; j < EL.cols(); ++j) {
            std::cout << EL(e,j)
                      << (j+1<EL.cols()? " , " : " ]\n");
        }
    }
    std::cout << "===============\n";
}

// Free-function definitions
std::pair<Mesh_1D::Vector, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition, int order, bool plot_mesh)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order});
    return {res.NL, res.EL[0]};
}

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition,
        const std::array<int,2>& order, bool plot_mesh)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order[0], order[1]});
    return {res.NL, res.EL[0], res.EL[1]};
}

std::tuple<Mesh_1D::Vector, Mesh_1D::Matrix, Mesh_1D::Matrix, Mesh_1D::Matrix>
mesh_1D(int /*PD*/, double domain_size, int partition,
        const std::array<int,3>& order, bool plot_mesh)
{
    Mesh_1D m;
    auto res = m.generate(domain_size, partition, {order[0], order[1], order[2]});
    return {res.NL, res.EL[0], res.EL[1], res.EL[2]};
}