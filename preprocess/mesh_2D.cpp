#include "mesh_2D.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <map>

// ---------- individual (MATLAB "individaul") ----------
void Mesh_2D::individual(double L,
                         int p,
                         int degree,
                         MatrixD& NL,
                         MatrixI& EL)
{
    if (p <= 0 || degree <= 0)
        throw std::invalid_argument("partition and degree must be positive.");
    if (degree > 4)
        throw std::invalid_argument("Only degrees 1..4 are implemented.");

    const int NoN = (degree * p + 1) * (degree * p + 1);
    const int NoE = p * p;
    const int NPE = (degree + 1) * (degree + 1);
    const int NX  = degree * p + 1;    // nodes per row/col

    NL.resize(NoN, 2);
    EL.resize(NoE, NPE);

    const double dx = L / static_cast<double>(degree * p);
    const double dy = L / static_cast<double>(degree * p);

    // Build NL with MATLAB's ndgrid(X,Y); NL = [xx(:) yy(:)];
    // (column-major flatten: outer Y, inner X)
    int idx = 0;
    for (int j = 0; j < NX; ++j) {           // Y index (0..NX-1)
        const double y = j * dy;
        for (int i = 0; i < NX; ++i) {       // X index (0..NX-1)
            NL(idx, 0) = i * dx;
            NL(idx, 1) = y;
            ++idx;
        }
    }

    // Elements: follow MATLAB exactly (1-based IDs)
    for (int i = 1; i <= p; ++i) {           // element row (Y)
        for (int j = 1; j <= p; ++j) {       // element col (X)
            const int e = (i - 1) * p + j;   // 1-based element row in EL
            const int r = e - 1;             // 0-based row

            if (j == 1) {
                switch (degree)
                {
                case 1: {
                    // base and vertical stride
                    const int base = (i - 1) * (p + 1) + j;
                    const int vs   = (p + 1);
                    EL(r, 0) = base;
                    EL(r, 1) = EL(r, 0) + 1;
                    EL(r, 3) = EL(r, 0) + vs;
                    EL(r, 2) = EL(r, 3) + 1;
                    break;
                }
                case 2: {
                    const int vs   = 2 * p + 1;
                    const int base = 2 * (i - 1) * vs + j;
                    EL(r, 0) = base;
                    EL(r, 4) = EL(r, 0) + 1;
                    EL(r, 1) = EL(r, 0) + 2;

                    EL(r, 7) = EL(r, 0) + vs;
                    EL(r, 8) = EL(r, 7) + 1;
                    EL(r, 5) = EL(r, 7) + 2;

                    EL(r, 3) = EL(r, 7) + vs;
                    EL(r, 6) = EL(r, 3) + 1;
                    EL(r, 2) = EL(r, 3) + 2;
                    break;
                }
                case 3: {
                    const int vs   = 3 * p + 1;
                    const int base = 3 * (i - 1) * vs + j;

                    EL(r, 0)  = base;
                    EL(r, 4)  = EL(r, 0)  + 1;
                    EL(r, 5)  = EL(r, 0)  + 2;
                    EL(r, 1)  = EL(r, 0)  + 3;

                    EL(r, 11) = EL(r, 0)  + vs;
                    EL(r, 12) = EL(r, 11) + 1;
                    EL(r, 13) = EL(r, 11) + 2;
                    EL(r, 6)  = EL(r, 11) + 3;

                    EL(r, 10) = EL(r, 11) + vs;
                    EL(r, 15) = EL(r, 10) + 1;
                    EL(r, 14) = EL(r, 10) + 2;
                    EL(r, 7)  = EL(r, 10) + 3;

                    EL(r, 3)  = EL(r, 10) + vs;
                    EL(r, 9)  = EL(r, 3)  + 1;
                    EL(r, 8)  = EL(r, 3)  + 2;
                    EL(r, 2)  = EL(r, 3)  + 3;
                    break;
                }
                case 4: {
                    const int vs   = 4 * p + 1;
                    const int base = 4 * (i - 1) * vs + j;

                    EL(r, 0)  = base;
                    EL(r, 4)  = EL(r, 0)  + 1;
                    EL(r, 5)  = EL(r, 0)  + 2;
                    EL(r, 6)  = EL(r, 0)  + 3;
                    EL(r, 1)  = EL(r, 0)  + 4;

                    EL(r, 15) = EL(r, 0)  + vs;
                    EL(r, 16) = EL(r, 15) + 1;
                    EL(r, 17) = EL(r, 15) + 2;
                    EL(r, 18) = EL(r, 15) + 3;
                    EL(r, 7)  = EL(r, 15) + 4;

                    EL(r, 14) = EL(r, 15) + vs;
                    EL(r, 23) = EL(r, 14) + 1;
                    EL(r, 24) = EL(r, 14) + 2;
                    EL(r, 19) = EL(r, 14) + 3;
                    EL(r, 8)  = EL(r, 14) + 4;

                    EL(r, 13) = EL(r, 14) + vs;
                    EL(r, 22) = EL(r, 13) + 1;
                    EL(r, 21) = EL(r, 13) + 2;
                    EL(r, 20) = EL(r, 13) + 3;
                    EL(r,10)  = EL(r, 13) + 4;

                    EL(r, 3)  = EL(r, 13) + vs;
                    EL(r, 12) = EL(r, 3)  + 1;
                    EL(r, 11) = EL(r, 3)  + 2;
                    EL(r, 9)  = EL(r, 3)  + 3;
                    EL(r, 2)  = EL(r, 3)  + 4;
                    break;
                }
                default: break;
                }
            } else {
                // Shift from previous element in the same row by 'degree'
                for (int c = 0; c < NPE; ++c)
                    EL(r, c) = EL(r - 1, c) + degree;
            }
        }
    }
}

// ---------- generate (MATLAB union + remap) ----------
Mesh_2D::Result Mesh_2D::generate(double L,
                                  int p,
                                  const std::vector<int>& orders) const
{
    if (orders.empty() || orders.size() > 3)
        throw std::invalid_argument("element_orders must have 1..3 entries.");

    // Per-order meshes
    std::vector<MatrixD> NLs;
    std::vector<MatrixI> ELs;
    NLs.reserve(orders.size());
    ELs.reserve(orders.size());

    for (int d : orders) {
        MatrixD NL_i; MatrixI EL_i;
        individual(L, p, d, NL_i, EL_i);
        NLs.emplace_back(std::move(NL_i));
        ELs.emplace_back(std::move(EL_i));
    }

    // Union of rows (exact), lexicographic on (x,y)
    std::vector<std::array<double,2>> all;
    size_t total_rows = 0;
    for (const auto& NL_i : NLs) total_rows += static_cast<size_t>(NL_i.rows());
    all.reserve(total_rows);

    for (const auto& NL_i : NLs) {
        for (int r = 0; r < NL_i.rows(); ++r)
            all.push_back({NL_i(r,0), NL_i(r,1)});
    }

    std::sort(all.begin(), all.end()); // std::array has lexicographic operator<
    all.erase(std::unique(all.begin(), all.end()),
              all.end());

    // Build unified NL
    Result res;
    res.NL.resize(static_cast<int>(all.size()), 2);
    for (int i = 0; i < res.NL.rows(); ++i) {
        res.NL(i,0) = all[static_cast<size_t>(i)][0];
        res.NL(i,1) = all[static_cast<size_t>(i)][1];
    }

    // Map (x,y) -> 1-based index in unified NL
    std::map<std::array<double,2>, int> idx_map; // lexicographic
    for (int i = 0; i < res.NL.rows(); ++i) {
        idx_map[{res.NL(i,0), res.NL(i,1)}] = i + 1; // 1-based
    }

    // Remap each EL via old->new node ids
    for (size_t k = 0; k < NLs.size(); ++k) {
        const MatrixD& NL_i = NLs[k];
        const MatrixI& Eold = ELs[k];
        Mesh_2D::MatrixI Enew = Eold;

        // build map from local node index (0-based) to new (1-based)
        std::vector<int> map_old_to_new(static_cast<size_t>(NL_i.rows()));
        for (int r = 0; r < NL_i.rows(); ++r) {
            auto it = idx_map.find({NL_i(r,0), NL_i(r,1)});
            if (it == idx_map.end())
                throw std::runtime_error("2D remap failed: node not found in unified NL.");
            map_old_to_new[static_cast<size_t>(r)] = it->second; // 1-based
        }

        for (int e = 0; e < Eold.rows(); ++e)
            for (int c = 0; c < Eold.cols(); ++c) {
                const int old1 = Eold(e,c);   // 1-based in local NL
                const int old0 = old1 - 1;
                Enew(e,c) = map_old_to_new[static_cast<size_t>(old0)];
            }

        res.EL.push_back(std::move(Enew));
    }

    return res;
}

// ---------- small console printer ----------
void printMesh2D(const Eigen::MatrixXd& NL,
                 const Eigen::MatrixXi& EL)
{
    std::cout << "=== 2D Mesh ===\n";
    std::cout << "Nodes (N=" << NL.rows() << "): [x y]\n";
    const int showN = std::min<int>(NL.rows(), 10);
    for (int i = 0; i < showN; ++i)
        std::cout << "  " << i+1 << ": " << NL(i,0) << "  " << NL(i,1) << "\n";
    if (NL.rows() > showN) std::cout << "  ...\n";

    std::cout << "\nElements (E=" << EL.rows()
              << ", NPE=" << (EL.cols()) << "): node IDs (1-based)\n";
    const int showE = std::min<int>(EL.rows(), 10);
    for (int e = 0; e < showE; ++e) {
        std::cout << "  e" << e+1 << ": [";
        for (int j = 0; j < EL.cols(); ++j) {
            std::cout << EL(e,j) << (j+1<EL.cols()? ", ":"");
        }
        std::cout << "]\n";
    }
    if (EL.rows() > showE) std::cout << "  ...\n";
    std::cout << "================\n";
}