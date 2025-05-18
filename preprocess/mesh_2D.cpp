#include "mesh_2D.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>

Mesh_2D::Result Mesh_2D::generate(double domain_size, int partition,
                                 const std::vector<int>& element_orders) const
{
    // 1) build each individual mesh
    std::vector<NodeMat> NL_list;
    std::vector<ElemMat> EL_list;
    NL_list.reserve(element_orders.size());
    EL_list.reserve(element_orders.size());

    for (int deg : element_orders) {
        NodeMat NL_i;
        ElemMat EL_i;
        individual(domain_size, partition, deg, NL_i, EL_i);
        NL_list.push_back(std::move(NL_i));
        EL_list.push_back(std::move(EL_i));
    }

    // 2) union of all node rows
    std::vector<std::array<double,2>> all_nodes;
    for (auto& NL_i : NL_list) {
        for (int r = 0; r < NL_i.rows(); ++r) {
            all_nodes.push_back({ NL_i(r,0), NL_i(r,1) });
        }
    }
    auto cmp = [](auto const&a, auto const&b){
        return a[0]<b[0] || (a[0]==b[0] && a[1]<b[1]);
    };
    std::sort(all_nodes.begin(), all_nodes.end(), cmp);
    all_nodes.erase(std::unique(all_nodes.begin(), all_nodes.end()), all_nodes.end());

    // 3) pack Result.NL
    Result R;
    R.NL.resize(all_nodes.size(), 2);
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        R.NL(i,0) = all_nodes[i][0];
        R.NL(i,1) = all_nodes[i][1];
    }

    // 4) remap each EL_i into global indices
    for (size_t m = 0; m < EL_list.size(); ++m) {
        const auto& NL_i = NL_list[m];
        const auto& Eold = EL_list[m];
        ElemMat Enew = Eold;

        for (int e = 0; e < Eold.rows(); ++e) {
            for (int k = 0; k < Eold.cols(); ++k) {
                int old_idx = Eold(e,k);  // already 0-based
                double x = NL_i(old_idx,0);
                double y = NL_i(old_idx,1);
                auto it = std::lower_bound(all_nodes.begin(),
                                           all_nodes.end(),
                                           std::array<double,2>{x,y},
                                           cmp);
                Enew(e,k) = static_cast<int>(std::distance(all_nodes.begin(), it));
            }
        }
        R.EL.push_back(std::move(Enew));
    }
    for (auto &E : R.EL) {
        E.array() += 1;
    }
    return R;
}

// ─── individual ──────────────────────────────────────────────────────────────
void Mesh_2D::individual(double domain_size, int partition, int degree,
                        NodeMat& NL, ElemMat& EL) const
 {
    int p = partition;
    int NoN = (degree * p + 1) * (degree * p + 1);
    int NoE = p * p;
    int NPE = (degree + 1) * (degree + 1);

    // Initialize NL and EL with appropriate sizes
    NL.resize(NoN, 2);
    EL.resize(NoE, NPE);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%    Nodes   %%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    double dx = domain_size / (degree * p);
    double dy = domain_size / (degree * p);

    // Generate coordinates
    std::vector<double> X, Y;
    for (double x = 0; x <= domain_size + 1e-10; x += dx) {
        X.push_back(x);
    }
    for (double y = 0; y <= domain_size + 1e-10; y += dy) {
        Y.push_back(y);
    }

    // Create node list
    int node_idx = 0;
    for (double y : Y) {
        for (double x : X) {
            NL(node_idx, 0) = x;
            NL(node_idx, 1) = y;
            node_idx++;
        }
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%  Elements  %%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for (int i = 1; i <= p; i++) {
        for (int j = 1; j <= p; j++) {
            if (j == 1) {
                switch (degree) {
                    case 1: {
                        int base = (i-1)*(p+1)+j;
                        EL((i-1)*p+j-1, 0) = base;
                        EL((i-1)*p+j-1, 1) = base + 1;
                        EL((i-1)*p+j-1, 3) = base + p+1;
                        EL((i-1)*p+j-1, 2) = base + p+2;
                        break;
                    }
                    case 2: {
                        int base = 2*(i-1)*(2*p+1)+j;
                        EL((i-1)*p+j-1, 0) = base;
                        EL((i-1)*p+j-1, 4) = base + 1;
                        EL((i-1)*p+j-1, 1) = base + 2;

                        EL((i-1)*p+j-1, 7) = base + 2*p+1;
                        EL((i-1)*p+j-1, 8) = base + 2*p+2;
                        EL((i-1)*p+j-1, 5) = base + 2*p+3;

                        EL((i-1)*p+j-1, 3) = base + 4*p+2;
                        EL((i-1)*p+j-1, 6) = base + 4*p+3;
                        EL((i-1)*p+j-1, 2) = base + 4*p+4;
                        break;
                    }
                    case 3: {
                        int base = 3*(i-1)*(3*p+1)+j;
                        EL((i-1)*p+j-1, 0) = base;
                        EL((i-1)*p+j-1, 4) = base + 1;
                        EL((i-1)*p+j-1, 5) = base + 2;
                        EL((i-1)*p+j-1, 1) = base + 3;

                        EL((i-1)*p+j-1, 11) = base + 3*p+1;
                        EL((i-1)*p+j-1, 12) = base + 3*p+2;
                        EL((i-1)*p+j-1, 13) = base + 3*p+3;
                        EL((i-1)*p+j-1, 6) = base + 3*p+4;

                        EL((i-1)*p+j-1, 10) = base + 6*p+2;
                        EL((i-1)*p+j-1, 15) = base + 6*p+3;
                        EL((i-1)*p+j-1, 14) = base + 6*p+4;
                        EL((i-1)*p+j-1, 7) = base + 6*p+5;

                        EL((i-1)*p+j-1, 3) = base + 9*p+3;
                        EL((i-1)*p+j-1, 9) = base + 9*p+4;
                        EL((i-1)*p+j-1, 8) = base + 9*p+5;
                        EL((i-1)*p+j-1, 2) = base + 9*p+6;
                        break;
                    }
                    case 4: {
                        int base = 4*(i-1)*(4*p+1)+j;
                        EL((i-1)*p+j-1, 0) = base;
                        EL((i-1)*p+j-1, 4) = base + 1;
                        EL((i-1)*p+j-1, 5) = base + 2;
                        EL((i-1)*p+j-1, 6) = base + 3;
                        EL((i-1)*p+j-1, 1) = base + 4;

                        EL((i-1)*p+j-1, 15) = base + 4*p+1;
                        EL((i-1)*p+j-1, 16) = base + 4*p+2;
                        EL((i-1)*p+j-1, 17) = base + 4*p+3;
                        EL((i-1)*p+j-1, 18) = base + 4*p+4;
                        EL((i-1)*p+j-1, 7) = base + 4*p+5;

                        EL((i-1)*p+j-1, 14) = base + 8*p+2;
                        EL((i-1)*p+j-1, 23) = base + 8*p+3;
                        EL((i-1)*p+j-1, 24) = base + 8*p+4;
                        EL((i-1)*p+j-1, 19) = base + 8*p+5;
                        EL((i-1)*p+j-1, 8) = base + 8*p+6;

                        EL((i-1)*p+j-1, 13) = base + 12*p+3;
                        EL((i-1)*p+j-1, 22) = base + 12*p+4;
                        EL((i-1)*p+j-1, 21) = base + 12*p+5;
                        EL((i-1)*p+j-1, 20) = base + 12*p+6;
                        EL((i-1)*p+j-1, 9) = base + 12*p+7;

                        EL((i-1)*p+j-1, 3) = base + 16*p+4;
                        EL((i-1)*p+j-1, 12) = base + 16*p+5;
                        EL((i-1)*p+j-1, 11) = base + 16*p+6;
                        EL((i-1)*p+j-1, 10) = base + 16*p+7;
                        EL((i-1)*p+j-1, 2) = base + 16*p+8;
                        break;
                    }
                }
            } else {
                for (int k = 0; k < NPE; k++) {
                    EL((i-1)*p+j-1, k) = EL((i-1)*p+j-2, k) + degree;
                }
            }
        }
    }

    // Convert to 0-based indexing if needed
    EL.array() -= 1;
}

// ─── free-function overloads ─────────────────────────────────────────────────
std::pair<Mesh_2D::NodeMat,Mesh_2D::ElemMat>
mesh_2D(int /*PD*/, double domain_size, int partition, int order)
{
    Mesh_2D m;
    auto R = m.generate(domain_size, partition, {order});
    return { R.NL, R.EL[0] };
}

std::tuple<Mesh_2D::NodeMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat>
mesh_2D(int /*PD*/, double domain_size, int partition,
        const std::array<int,2>& orders)
{
    Mesh_2D m;
    auto R = m.generate(domain_size, partition, {orders[0], orders[1]});
    return { R.NL, R.EL[0], R.EL[1] };
}

std::tuple<Mesh_2D::NodeMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat,Mesh_2D::ElemMat>
mesh_2D(int /*PD*/, double domain_size, int partition,
        const std::array<int,3>& orders)
{
    Mesh_2D m;
    auto R = m.generate(domain_size, partition,
                        {orders[0], orders[1], orders[2]});
    return { R.NL, R.EL[0], R.EL[1], R.EL[2] };
}

// ─── debug printer ───────────────────────────────────────────────────────────
void printMesh2D(const Mesh_2D::NodeMat& NL,
                 const Mesh_2D::ElemMat& EL)
{
    std::cout << "\n=== 2D Mesh ===\n";
    std::cout << "Nodes (" << NL.rows() << "):\n";
    for (int i = 0; i < NL.rows(); ++i) {
        std::cout << "  " << std::setw(4) << "["
                  << std::setw(8) << NL(i,0) << ", "
                  << std::setw(8) << NL(i,1) << "]\n";
    }

    // Print Elements
    std::cout << "\nElements (" << EL.rows() << "):\n";
    for (int e = 0; e < EL.rows(); ++e) {
        std::cout << "  " << std::setw(4) << "[ ";
        for (int k = 0; k < EL.cols(); ++k) {
            std::cout << EL(e,k);
            if (k+1 < EL.cols()) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << "===============\n\n";
}