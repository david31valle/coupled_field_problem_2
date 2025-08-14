#include "mesh_3D.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>

// -------------------- individual (MATLAB "individaul") --------------------
void Mesh_3D::individual(double L,
                         int p,
                         int degree,
                         MatrixD& NL,
                         MatrixI& EL)
{
    if (p <= 0 || degree <= 0)
        throw std::invalid_argument("partition and degree must be positive.");
    if (degree > 4)
        throw std::invalid_argument("Only degrees 1..4 are implemented.");

    const int NX  = degree * p + 1;                 // nodes per axis
    const int NoN = NX * NX * NX;
    const int NoE = p * p * p;
    const int NPE = (degree + 1) * (degree + 1) * (degree + 1);

    NL.resize(NoN, 3);
    EL.resize(NoE, NPE);

    const double dx = L / static_cast<double>(degree * p);
    const double dy = L / static_cast<double>(degree * p);
    const double dz = L / static_cast<double>(degree * p);

    // NL = [xx(:) yy(:) zz(:)] from ndgrid(X,Y,Z) with linear indexing
    // i (X) fastest, then j (Y), k (Z)
    int idx = 0;
    for (int k = 0; k < NX; ++k) {
        const double z = k * dz;
        for (int j = 0; j < NX; ++j) {
            const double y = j * dy;
            for (int i = 0; i < NX; ++i) {
                NL(idx, 0) = i * dx;
                NL(idx, 1) = y;
                NL(idx, 2) = z;
                ++idx;
            }
        }
    }

    // Helper: 1-based column setter
    auto set1 = [&](int r, int col1, int val) {
        EL(r, col1 - 1) = val;
    };

    // Precomputed strides
    const int vs = NX;            // "vertical" stride across Y
    const int plane = NX * NX;    // Z-plane stride

    // Elements
    for (int ii = 1; ii <= p; ++ii) {               // layer in Z (MATLAB i)
        for (int jj = 1; jj <= p; ++jj) {           // row in Y (MATLAB j)
            for (int kk = 1; kk <= p; ++kk) {       // col in X (MATLAB k)
                const int e1 = (ii - 1) * p * p + (jj - 1) * p + kk; // 1-based
                const int r  = e1 - 1; // row in EL

                if (ii == 1) {
                    if (kk == 1) {
                        switch (degree)
                        {
                        case 1: {
                            // MATLAB:
                            // EL(...,4)=(j-1)*(p+1)+k; EL(...,3)=...+1;
                            // EL(...,8)=...+(p+1);     EL(...,7)=...+1;
                            // EL(...,1)=...+(p+1)^2;   EL(...,2)=...+(p+1)^2; ...
                            const int base = (jj - 1) * (p + 1) + kk;
                            set1(r, 4, base);
                            set1(r, 3, base + 1);
                            set1(r, 8, base + (p + 1));
                            set1(r, 7, base + (p + 1) + 1);

                            set1(r, 1, base + (p + 1) * (p + 1));
                            set1(r, 2, (base + 1) + (p + 1) * (p + 1));
                            set1(r, 5, (base + (p + 1)) + (p + 1) * (p + 1));
                            set1(r, 6, (base + (p + 1) + 1) + (p + 1) * (p + 1));
                            break;
                        }
                        case 2: {
                            // vs = 2*p+1
                            const int v2  = 2 * p + 1;
                            const int pl2 = v2 * v2;
                            const int base = 2 * (jj - 1) * v2 + kk;

                            set1(r,  4, base);
                            set1(r, 11, base + 1);
                            set1(r,  3, base + 2);

                            set1(r, 16, base + v2);
                            set1(r, 24, base + v2 + 1);
                            set1(r, 15, base + v2 + 2);

                            set1(r,  8, base + 2 * v2);
                            set1(r, 19, base + 2 * v2 + 1);
                            set1(r,  7, base + 2 * v2 + 2);

                            set1(r, 12, base + pl2);
                            set1(r, 21, base + 1 + pl2);
                            set1(r, 10, base + 2 + pl2);
                            set1(r, 25, base + v2 + pl2);
                            set1(r, 27, base + v2 + 1 + pl2);
                            set1(r, 23, base + v2 + 2 + pl2);
                            set1(r, 20, base + 2 * v2 + pl2);
                            set1(r, 26, base + 2 * v2 + 1 + pl2);
                            set1(r, 18, base + 2 * v2 + 2 + pl2);

                            set1(r,  1, base + 2 * pl2);
                            set1(r,  9, base + 1 + 2 * pl2);
                            set1(r,  2, base + 2 + 2 * pl2);
                            set1(r, 13, base + v2 + 2 * pl2);
                            set1(r, 22, base + v2 + 1 + 2 * pl2);
                            set1(r, 14, base + v2 + 2 + 2 * pl2);
                            set1(r,  5, base + 2 * v2 + 2 * pl2);
                            set1(r, 17, base + 2 * v2 + 1 + 2 * pl2);
                            set1(r,  6, base + 2 * v2 + 2 + 2 * pl2);
                            break;
                        }
                        case 3: {
                            const int v3  = 3 * p + 1;
                            const int pl3 = v3 * v3;
                            const int base = 3 * (jj - 1) * v3 + kk;

                            set1(r,  4, base);
                            set1(r, 14, base + 1);
                            set1(r, 13, base + 2);
                            set1(r,  3, base + 3);

                            set1(r, 23, base + v3);
                            set1(r, 46, base + v3 + 1);
                            set1(r, 45, base + v3 + 2);
                            set1(r, 21, base + v3 + 3);

                            set1(r, 24, base + 2 * v3);
                            set1(r, 47, base + 2 * v3 + 1);
                            set1(r, 48, base + 2 * v3 + 2);
                            set1(r, 22, base + 2 * v3 + 3);

                            set1(r,  8, base + 3 * v3);
                            set1(r, 30, base + 3 * v3 + 1);
                            set1(r, 29, base + 3 * v3 + 2);
                            set1(r,  7, base + 3 * v3 + 3);

                            set1(r, 15, base + pl3);
                            set1(r, 36, base + 1 + pl3);
                            set1(r, 35, base + 2 + pl3);
                            set1(r, 12, base + 3 + pl3);
                            set1(r, 49, base + v3 + pl3);
                            set1(r, 60, base + v3 + 1 + pl3);
                            set1(r, 59, base + v3 + 2 + pl3);
                            set1(r, 42, base + v3 + 3 + pl3);
                            set1(r, 52, base + 2 * v3 + pl3);
                            set1(r, 64, base + 2 * v3 + 1 + pl3);
                            set1(r, 63, base + 2 * v3 + 2 + pl3);
                            set1(r, 43, base + 2 * v3 + 3 + pl3);
                            set1(r, 31, base + 3 * v3 + pl3);
                            set1(r, 56, base + 3 * v3 + 1 + pl3);
                            set1(r, 55, base + 3 * v3 + 2 + pl3);
                            set1(r, 28, base + 3 * v3 + 3 + pl3);

                            set1(r, 16, base + 2 * pl3);
                            set1(r, 33, base + 1 + 2 * pl3);
                            set1(r, 34, base + 2 + 2 * pl3);
                            set1(r, 11, base + 3 + 2 * pl3);
                            set1(r, 50, base + v3 + 2 * pl3);
                            set1(r, 57, base + v3 + 1 + 2 * pl3);
                            set1(r, 58, base + v3 + 2 + 2 * pl3);
                            set1(r, 41, base + v3 + 3 + 2 * pl3);
                            set1(r, 51, base + 2 * v3 + 2 * pl3);
                            set1(r, 61, base + 2 * v3 + 1 + 2 * pl3);
                            set1(r, 62, base + 2 * v3 + 2 + 2 * pl3);
                            set1(r, 44, base + 2 * v3 + 3 + 2 * pl3);
                            set1(r, 32, base + 3 * v3 + 2 * pl3);
                            set1(r, 53, base + 3 * v3 + 1 + 2 * pl3);
                            set1(r, 54, base + 3 * v3 + 2 + 2 * pl3);
                            set1(r, 27, base + 3 * v3 + 3 + 2 * pl3);

                            set1(r,  1, base + 3 * pl3);
                            set1(r,  9, base + 1 + 3 * pl3);
                            set1(r, 10, base + 2 + 3 * pl3);
                            set1(r,  2, base + 3 + 3 * pl3);
                            set1(r, 17, base + v3 + 3 * pl3);
                            set1(r, 37, base + v3 + 1 + 3 * pl3);
                            set1(r, 38, base + v3 + 2 + 3 * pl3);
                            set1(r, 19, base + v3 + 3 + 3 * pl3);
                            set1(r, 18, base + 2 * v3 + 3 * pl3);
                            set1(r, 40, base + 2 * v3 + 1 + 3 * pl3);
                            set1(r, 39, base + 2 * v3 + 2 + 3 * pl3);
                            set1(r, 20, base + 2 * v3 + 3 + 3 * pl3);
                            set1(r,  5, base + 3 * v3 + 3 * pl3);
                            set1(r, 25, base + 3 * v3 + 1 + 3 * pl3);
                            set1(r, 26, base + 3 * v3 + 2 + 3 * pl3);
                            set1(r,  6, base + 3 * v3 + 3 + 3 * pl3);
                            break;
                        }
                        case 4: {
                            const int v4  = 4 * p + 1;
                            const int pl4 = v4 * v4;
                            const int base = 4 * (jj - 1) * v4 + kk;

                            set1(r,  4, base);
                            set1(r, 17, base + 1);
                            set1(r, 16, base + 2);
                            set1(r, 15, base + 3);
                            set1(r,  3, base + 4);

                            set1(r, 30, base + v4);
                            set1(r, 74, base + v4 + 1);
                            set1(r, 73, base + v4 + 2);
                            set1(r, 72, base + v4 + 3);
                            set1(r, 27, base + v4 + 4);

                            set1(r, 31, base + 2 * v4);
                            set1(r, 75, base + 2 * v4 + 1);
                            set1(r, 80, base + 2 * v4 + 2);
                            set1(r, 79, base + 2 * v4 + 3);
                            set1(r, 28, base + 2 * v4 + 4);

                            set1(r, 32, base + 3 * v4);
                            set1(r, 76, base + 3 * v4 + 1);
                            set1(r, 77, base + 3 * v4 + 2);
                            set1(r, 78, base + 3 * v4 + 3);
                            set1(r, 29, base + 3 * v4 + 4);

                            set1(r,  8, base + 4 * v4);
                            set1(r, 41, base + 4 * v4 + 1);
                            set1(r, 40, base + 4 * v4 + 2);
                            set1(r, 39, base + 4 * v4 + 3);
                            set1(r,  7, base + 4 * v4 + 4);

                            // pl4
                            set1(r, 18, base + pl4);
                            set1(r, 51, base + 1 + pl4);
                            set1(r, 50, base + 2 + pl4);
                            set1(r, 49, base + 3 + pl4);
                            set1(r, 14, base + 4 + pl4);
                            set1(r, 81, base + v4 + pl4);
                            set1(r,105, base + v4 + 1 + pl4);
                            set1(r,104, base + v4 + 2 + pl4);
                            set1(r,103, base + v4 + 3 + pl4);
                            set1(r, 65, base + v4 + 4 + pl4);
                            set1(r, 88, base + 2 * v4 + pl4);
                            set1(r,113, base + 2 * v4 + 1 + pl4);
                            set1(r,112, base + 2 * v4 + 2 + pl4);
                            set1(r,111, base + 2 * v4 + 3 + pl4);
                            set1(r, 66, base + 2 * v4 + 4 + pl4);
                            set1(r, 87, base + 3 * v4 + pl4);
                            set1(r,121, base + 3 * v4 + 1 + pl4);
                            set1(r,120, base + 3 * v4 + 2 + pl4);
                            set1(r,119, base + 3 * v4 + 3 + pl4);
                            set1(r, 69, base + 3 * v4 + 4 + pl4);
                            set1(r, 42, base + 4 * v4 + pl4);
                            set1(r, 96, base + 4 * v4 + 1 + pl4);
                            set1(r, 95, base + 4 * v4 + 2 + pl4);
                            set1(r, 94, base + 4 * v4 + 3 + pl4);
                            set1(r, 38, base + 4 * v4 + 4 + pl4);

                            // 2*pl4
                            set1(r, 19, base + 2 * pl4);
                            set1(r, 52, base + 1 + 2 * pl4);
                            set1(r, 53, base + 2 + 2 * pl4);
                            set1(r, 48, base + 3 + 2 * pl4);
                            set1(r, 13, base + 4 + 2 * pl4);
                            set1(r, 82, base + v4 + 2 * pl4);
                            set1(r,106, base + v4 + 1 + 2 * pl4);
                            set1(r,123, base + v4 + 2 + 2 * pl4);
                            set1(r,102, base + v4 + 3 + 2 * pl4);
                            set1(r, 64, base + v4 + 4 + 2 * pl4);
                            set1(r, 89, base + 2 * v4 + 2 * pl4);
                            set1(r,114, base + 2 * v4 + 1 + 2 * pl4);
                            set1(r,125, base + 2 * v4 + 2 + 2 * pl4);
                            set1(r,110, base + 2 * v4 + 3 + 2 * pl4);
                            set1(r, 71, base + 2 * v4 + 4 + 2 * pl4);
                            set1(r, 86, base + 3 * v4 + 2 * pl4);
                            set1(r,122, base + 3 * v4 + 1 + 2 * pl4);
                            set1(r,124, base + 3 * v4 + 2 + 2 * pl4);
                            set1(r,118, base + 3 * v4 + 3 + 2 * pl4);
                            set1(r, 68, base + 3 * v4 + 4 + 2 * pl4);
                            set1(r, 43, base + 4 * v4 + 2 * pl4);
                            set1(r, 97, base + 4 * v4 + 1 + 2 * pl4);
                            set1(r, 98, base + 4 * v4 + 2 + 2 * pl4);
                            set1(r, 93, base + 4 * v4 + 3 + 2 * pl4);
                            set1(r, 37, base + 4 * v4 + 4 + 2 * pl4);

                            // 3*pl4
                            set1(r, 20, base + 3 * pl4);
                            set1(r, 45, base + 1 + 3 * pl4);
                            set1(r, 46, base + 2 + 3 * pl4);
                            set1(r, 47, base + 3 + 3 * pl4);
                            set1(r, 12, base + 4 + 3 * pl4);
                            set1(r, 83, base + v4 + 3 * pl4);
                            set1(r, 99, base + v4 + 1 + 3 * pl4);
                            set1(r,100, base + v4 + 2 + 3 * pl4);
                            set1(r,101, base + v4 + 3 + 3 * pl4);
                            set1(r, 63, base + v4 + 4 + 3 * pl4);
                            set1(r, 84, base + 2 * v4 + 3 * pl4);
                            set1(r,107, base + 2 * v4 + 1 + 3 * pl4);
                            set1(r,108, base + 2 * v4 + 2 + 3 * pl4);
                            set1(r,109, base + 2 * v4 + 3 + 3 * pl4);
                            set1(r, 70, base + 2 * v4 + 4 + 3 * pl4);
                            set1(r, 85, base + 3 * v4 + 3 * pl4);
                            set1(r,115, base + 3 * v4 + 1 + 3 * pl4);
                            set1(r,116, base + 3 * v4 + 2 + 3 * pl4);
                            set1(r,117, base + 3 * v4 + 3 + 3 * pl4);
                            set1(r, 69, base + 3 * v4 + 4 + 3 * pl4);
                            set1(r, 44, base + 4 * v4 + 3 * pl4);
                            set1(r, 90, base + 4 * v4 + 1 + 3 * pl4);
                            set1(r, 91, base + 4 * v4 + 2 + 3 * pl4);
                            set1(r, 92, base + 4 * v4 + 3 + 3 * pl4);
                            set1(r, 36, base + 4 * v4 + 4 + 3 * pl4);

                            // 4*pl4 (top plane)
                            set1(r,  1, base + 4 * pl4);
                            set1(r,  9, base + 1 + 4 * pl4);
                            set1(r, 10, base + 2 + 4 * pl4);
                            set1(r, 11, base + 3 + 4 * pl4);
                            set1(r,  2, base + 4 + 4 * pl4);
                            set1(r, 21, base + v4 + 4 * pl4);
                            set1(r, 54, base + v4 + 1 + 4 * pl4);
                            set1(r, 55, base + v4 + 2 + 4 * pl4);
                            set1(r, 56, base + v4 + 3 + 4 * pl4);
                            set1(r, 24, base + v4 + 4 + 4 * pl4);
                            set1(r, 22, base + 2 * v4 + 4 * pl4);
                            set1(r, 61, base + 2 * v4 + 1 + 4 * pl4);
                            set1(r, 62, base + 2 * v4 + 2 + 4 * pl4);
                            set1(r, 57, base + 2 * v4 + 3 + 4 * pl4);
                            set1(r, 25, base + 2 * v4 + 4 + 4 * pl4);
                            set1(r, 23, base + 3 * v4 + 4 * pl4);
                            set1(r, 60, base + 3 * v4 + 1 + 4 * pl4);
                            set1(r, 59, base + 3 * v4 + 2 + 4 * pl4);
                            set1(r, 58, base + 3 * v4 + 3 + 4 * pl4);
                            set1(r, 26, base + 3 * v4 + 4 + 4 * pl4);
                            set1(r,  5, base + 4 * v4 + 4 * pl4);
                            set1(r, 33, base + 4 * v4 + 1 + 4 * pl4);
                            set1(r, 34, base + 4 * v4 + 2 + 4 * pl4);
                            set1(r, 35, base + 4 * v4 + 3 + 4 * pl4);
                            set1(r,  6, base + 4 * v4 + 4 + 4 * pl4);
                            break;
                        }
                        default: break;
                        }
                    } else {
                        // Same Z-layer & row, shift from previous element in X by 'degree'
                        for (int c = 0; c < NPE; ++c)
                            EL(r, c) = EL(r - 1, c) + degree;
                    }
                } else {
                    // Copy from element one layer below (p^2 elements back) + stride across Z
                    const int prev = r - p * p; // previous layer's same (j,k)
                    for (int c = 0; c < NPE; ++c)
                        EL(r, c) = EL(prev, c) + degree * plane;
                }
            }
        }
    }
}

// -------------------- generate (MATLAB union + remap) --------------------
Mesh_3D::Result Mesh_3D::generate(double L,
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

    // Union rows (exact), lexicographic (x,y,z)
    std::vector<std::array<double,3>> all;
    size_t total = 0;
    for (const auto& NL_i : NLs) total += static_cast<size_t>(NL_i.rows());
    all.reserve(total);

    for (const auto& NL_i : NLs)
        for (int r = 0; r < NL_i.rows(); ++r)
            all.push_back({NL_i(r,0), NL_i(r,1), NL_i(r,2)});

    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    Result res;
    res.NL.resize(static_cast<int>(all.size()), 3);
    for (int i = 0; i < res.NL.rows(); ++i) {
        res.NL(i,0) = all[static_cast<size_t>(i)][0];
        res.NL(i,1) = all[static_cast<size_t>(i)][1];
        res.NL(i,2) = all[static_cast<size_t>(i)][2];
    }

    // Map (x,y,z) -> 1-based node id
    std::map<std::array<double,3>, int> idx_map;
    for (int i = 0; i < res.NL.rows(); ++i)
        idx_map[{res.NL(i,0), res.NL(i,1), res.NL(i,2)}] = i + 1;

    // Remap ELs to unified NL (1-based preserved)
    for (size_t k = 0; k < NLs.size(); ++k) {
        const MatrixD& NL_i = NLs[k];
        const MatrixI& Eold = ELs[k];
        MatrixI Enew = Eold;

        std::vector<int> map_old_to_new(static_cast<size_t>(NL_i.rows()));
        for (int r = 0; r < NL_i.rows(); ++r) {
            auto it = idx_map.find({NL_i(r,0), NL_i(r,1), NL_i(r,2)});
            if (it == idx_map.end())
                throw std::runtime_error("3D remap failed: node not found in unified NL.");
            map_old_to_new[static_cast<size_t>(r)] = it->second; // 1-based
        }

        for (int e = 0; e < Eold.rows(); ++e)
            for (int c = 0; c < Eold.cols(); ++c) {
                const int old1 = Eold(e,c);   // 1-based local id
                const int old0 = old1 - 1;
                Enew(e,c) = map_old_to_new[static_cast<size_t>(old0)];
            }

        res.EL.push_back(std::move(Enew));
    }

    return res;
}

// -------------------- small console printer --------------------
void printMesh3D(const Eigen::MatrixXd& NL,
                 const Eigen::MatrixXi& EL)
{
    std::cout << "=== 3D Mesh ===\n";
    std::cout << "Nodes (N=" << NL.rows() << "): [x y z]\n";
    const int showN = std::min<int>(NL.rows(), 10);
    for (int i = 0; i < showN; ++i)
        std::cout << "  " << i+1 << ": " << NL(i,0) << "  " << NL(i,1) << "  " << NL(i,2) << "\n";
    if (NL.rows() > showN) std::cout << "  ...\n";

    std::cout << "\nElements (E=" << EL.rows()
              << ", NPE=" << (EL.cols()) << "): node IDs (1-based)\n";
    const int showE = std::min<int>(EL.rows(), 6);
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