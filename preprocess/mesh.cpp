#include "mesh.hpp"
#include "mesh_1D.hpp"
#include "mesh_2D.hpp"
#include "mesh_3D.hpp"
#include <iostream>
#include <stdexcept>

// ----------------- Mesh -----------------
Mesh::Mesh(int problem_dimension,
           double domain_size,
           int partition,
           std::vector<int> element_orders,
           bool plot_mesh)
: dim_(problem_dimension), L_(domain_size), p_(partition), orders_(std::move(element_orders)), plot_(plot_mesh) {}

Mesh::Result Mesh::build() const {
    switch (dim_) {
        case 1: return build1D(L_, p_, orders_, plot_);
        case 2: return build2D(L_, p_, orders_, plot_);
        case 3: return build3D(L_, p_, orders_, plot_);
        default: throw std::invalid_argument("problem_dimension must be 1, 2, or 3");
    }
}

Mesh::Result Mesh::build1D(double L, int p, const std::vector<int>& orders, bool plot) {
    Mesh_1D m;
    auto res = m.generate(L, p, orders); // res.NL: VectorXd, res.EL[k]: MatrixXi

    // Convert NL (VectorXd) to MatrixXd (N x 1) for uniform interface
    Mesh::Result out;
    out.NL.resize(res.NL.size(), 1);
    out.NL.col(0) = res.NL;

    out.ELs = std::move(res.EL);

    if (plot && !out.ELs.empty()) {
        // Optional: 1D print helper
        printMesh1D(res.NL, out.ELs[0]);
    }
    return out;
}

Mesh::Result Mesh::build2D(double L, int p, const std::vector<int>& orders, bool plot) {
    Mesh_2D m;
    auto res = m.generate(L, p, orders); // res.NL: (N x 2)

    Mesh::Result out;
    out.NL  = res.NL;
    out.ELs = std::move(res.EL);

    if (plot && !out.ELs.empty()) {
        printMesh2D(out.NL, out.ELs[0]);
    }
    return out;
}

Mesh::Result Mesh::build3D(double L, int p, const std::vector<int>& orders, bool plot) {
    Mesh_3D m;
    auto res = m.generate(L, p, orders); // res.NL: (N x 3)

    Mesh::Result out;
    out.NL  = res.NL;
    out.ELs = std::move(res.EL);

    if (plot && !out.ELs.empty()) {
        printMesh3D(out.NL, out.ELs[0]);
    }
    return out;
}

// ----------------- Existing facade API (unchanged for main) -----------------
std::tuple<Eigen::MatrixXd,  std::vector<Eigen::MatrixXd> >
generate_mesh(int /*PD*/,
              double domain_size,
              int partition,
              const std::vector<int>& element_orders,
              int problem_dimension)
{
    Eigen::MatrixXd nl, el1, el2;
    std::vector<Eigen::MatrixXd> Element_lists;

    try {
        if (element_orders.empty())
            throw std::invalid_argument("element_orders must have at least one entry.");

        // Reuse computation if first two orders are equal
        std::vector<int> orders_to_build;
        if (element_orders.size() == 1) {
            orders_to_build = { element_orders[0] };
        } else {
            const int o1 = element_orders[0];
            const int o2 = element_orders[1];
            orders_to_build = (o1 == o2) ? std::vector<int>{ o1 }
                                         : std::vector<int>{ o1, o2 };
        }

        Mesh mesh(problem_dimension, domain_size, partition, orders_to_build, /*plot=*/false);
        auto out = mesh.build();

        // --- Print nodes + first EL based on problem dimension ---
        switch (problem_dimension) {
        case 1: {
                // printMesh1D expects a VectorXd; out.NL is (N x 1)
                if (!out.ELs.empty()) {
                    Eigen::VectorXd nl1d = out.NL.col(0);
                    printMesh1D(nl1d, out.ELs[0]);
                }
                break;
        }
        case 2: {
                if (!out.ELs.empty()) {
                    printMesh2D(out.NL, out.ELs[0]);
                }
                break;
        }
        case 3: {
                if (!out.ELs.empty()) {
                    printMesh3D(out.NL, out.ELs[0]);
                }
                break;
        }
        default: break;
        }

        // Nodes
        nl = out.NL; // (N x PD)

        // Element lists (cast to double to match your current return type)
        if (!out.ELs.empty())
            el1 = out.ELs[0].cast<double>();

        if (out.ELs.size() > 1) {
            el2 = out.ELs[1].cast<double>();
        } else if (element_orders.size() >= 2 && element_orders[0] == element_orders[1] && !out.ELs.empty()) {
            // If two orders were requested but equal, duplicate EL1
            el2 = el1;
        } else {
            el2.resize(0,0);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Mesh generation error: " << e.what() << "\n";
        nl.resize(0,0);
        el1.resize(0,0);
        el2.resize(0,0);
    }

    Element_lists.push_back(el1);
    Element_lists.push_back(el2);
    return { nl, Element_lists };
}