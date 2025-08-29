
#include "problem.hpp"


#include "problem.hpp"
#include <fstream>
#include <iomanip>

#include <set>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

problem_coupled::problem_coupled(
        int PD,
        std::vector<Node>& NL,
        std::vector<element>& EL,
        int domain_size,
        const std::string& BC_type,
        const std::string Corners,
        const std::string& Initial_density,
        const std::vector<double>& parameters,
        const std::vector<int>& element_order,
        const Eigen::Vector2i& field_dim,
        const std::string& GP_vals,
        const std::string& time_incr_method,
        double T,
        double dt,
        double time_factor,
        int max_iter,
        double tol
) :
        PD(PD), Node_List(NL), Element_List(EL), domain_size(domain_size),
        BC_type(BC_type), Initial_density(Initial_density), parameters(parameters),
        element_order(element_order), field_dim(field_dim),
        GP_vals(GP_vals), time_incr_method(time_incr_method),
        T(T), dt(dt), time_factor(time_factor), max_iter(max_iter), tol(tol)
{
    // === Construct output filename ===
    std::ostringstream oss;
    oss << PD << "D_Normal_" << EL.size()
        << "_EL=[";
    for (int eo : element_order) oss << eo;
    oss << "]_" << Initial_density << "_" << time_incr_method << "_" << BC_type << "_[";
    for (size_t i = 0; i < parameters.size(); ++i) {
        oss << parameters[i];
        if (i != parameters.size() - 1) oss << ",";
    }
    oss << "].txt";

    filename = oss.str();

    // === Assign boundary conditions ===
    Assign_BC(Corners);  // fills BC and DOFs
    //Assign_DOF_DBC();    // computes global DOF numbering

    // === Assign Gauss Point DOFs if requested ===
    if (GP_vals == "On") {
        Assign_GP_DOFs();
    }

    // === Output problem setup info ===
    problem_info();  // displays PD, mesh, DOFs, etc.
    solve();
}



void problem_coupled::Assign_BC(const std::string Corners) {
    int NoNs = Node_List.size();
    double tol = 1e-6;

    double W = static_cast<double>(domain_size);
    double H = W;
    double D = W;

    if (BC_type == "DBC") {
        // (DBC branch from before...)
        Assign_DOF_DBC();
        return;
    }

    // === PBC ===
    std::vector<int> NLC;  // Corner nodes
    std::vector<int> NLS;  // Star nodes
    std::vector<int> NLM;  // Minus edge/face
    std::vector<int> NLP;  // Plus edge/face

    if (PD == 1) {
        for (int i = 0; i < NoNs; ++i) {
            auto& node = Node_List[i];
            double x = node.X(0);

            if (std::abs(x - 0.0) < tol || std::abs(x - W) < tol)
                NLC.push_back(i);
            else
                NLS.push_back(i);
        }
    }

    else if (PD == 2) {
        std::vector<int> MEX, PEX, MEY, PEY;

        for (int i = 0; i < NoNs; ++i) {
            const auto& X = Node_List[i].X;
            double x = X(0), y = X(1);

            bool onLeft = std::abs(x - 0.0) < tol;
            bool onRight = std::abs(x - W) < tol;
            bool onBottom = std::abs(y - 0.0) < tol;
            bool onTop = std::abs(y - H) < tol;

            bool isCorner =
                    (onLeft && onBottom) || (onRight && onBottom) ||
                    (onRight && onTop) || (onLeft && onTop);

            if (isCorner) {
                NLC.push_back(i);
            } else if (onLeft) {
                MEY.push_back(i);
            } else if (onRight) {
                PEY.push_back(i);
            } else if (onBottom) {
                MEX.push_back(i);
            } else if (onTop) {
                PEX.push_back(i);
            } else {
                NLS.push_back(i);
            }
        }

        // Match edges to construct NLM/NLP
        for (int pi : PEX) {
            for (int mi : MEX) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PEY) {
            for (int mi : MEY) {
                if (std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }
    }

    else if (PD == 3) {
        std::vector<int> MEX, PEX, MEY, PEY, MEZ, PEZ;
        std::vector<int> MFX, PFX, MFY, PFY, MFZ, PFZ;

        for (int i = 0; i < NoNs; ++i) {
            const auto& X = Node_List[i].X;
            double x = X(0), y = X(1), z = X(2);

            bool onX0 = std::abs(x - 0.0) < tol;
            bool onXW = std::abs(x - W) < tol;
            bool onY0 = std::abs(y - 0.0) < tol;
            bool onYH = std::abs(y - H) < tol;
            bool onZ0 = std::abs(z - 0.0) < tol;
            bool onZD = std::abs(z - D) < tol;

            if ((onX0 && onY0 && onZ0) || (onX0 && onY0 && onZD) ||
                (onX0 && onYH && onZ0) || (onX0 && onYH && onZD) ||
                (onXW && onY0 && onZ0) || (onXW && onY0 && onZD) ||
                (onXW && onYH && onZ0) || (onXW && onYH && onZD)) {
                NLC.push_back(i);
            } else if (onX0 && onY0) {
                MEZ.push_back(i);
            } else if (onX0 && onYH) {
                PEZ.push_back(i);
            } else if (onX0 && onZ0) {
                MEY.push_back(i);
            } else if (onX0 && onZD) {
                PEY.push_back(i);
            } else if (onY0 && onXW) {
                PEZ.push_back(i);
            } else if (onY0 && onZ0) {
                MEX.push_back(i);
            } else if (onY0 && onZD) {
                PEX.push_back(i);
            } else if (onX0) {
                MFX.push_back(i);
            } else if (onXW) {
                PFX.push_back(i);
            } else if (onY0) {
                MFY.push_back(i);
            } else if (onYH) {
                PFY.push_back(i);
            } else if (onZ0) {
                MFZ.push_back(i);
            } else if (onZD) {
                PFZ.push_back(i);
            } else {
                NLS.push_back(i);
            }
        }

        // Match each pair of periodic faces
        auto match_nodes = [&](const std::vector<int>& plus, const std::vector<int>& minus, int comp) {
            for (int pi : plus) {
                for (int mi : minus) {
                    if (std::abs(Node_List[pi].X(comp) - Node_List[mi].X(comp)) < tol) {
                        NLP.push_back(pi);
                        NLM.push_back(mi);
                    }
                }
            }
        };

        match_nodes(PEX, MEX, 0);
        match_nodes(PEY, MEY, 1);
        match_nodes(PEZ, MEZ, 2);

        for (int pi : PFX) {
            for (int mi : MFX) {
                if (std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol &&
                    std::abs(Node_List[pi].X(2) - Node_List[mi].X(2)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PFY) {
            for (int mi : MFY) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol &&
                    std::abs(Node_List[pi].X(2) - Node_List[mi].X(2)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PFZ) {
            for (int mi : MFZ) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol &&
                    std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }
    }

    // === Assign values for corner and periodic nodes ===

    for (int i : NLC) {
        Node_List[i].U = Node_List[NLC[0]].U;
        Node_List[i].u = Node_List[NLC[0]].u;
        Node_List[i].un = Node_List[NLC[0]].un;
    }

    for (size_t i = 0; i < NLP.size(); ++i) {
        Node_List[NLP[i]].U = Node_List[NLM[i]].U;
        Node_List[NLP[i]].u = Node_List[NLM[i]].u;
        Node_List[NLP[i]].un = Node_List[NLM[i]].un;
    }

    // === Corner constraint to avoid rigid body motion ===
    if (Corners == "Fixed") {
        for (int i : NLC) {
            Node_List[i].U.segment(1, PD).setZero();
            Node_List[i].u.segment(1, PD).setZero();
            Node_List[i].un.segment(1, PD).setZero();
            Node_List[i].BC.segment(1, PD).setZero();
        }
    }

    // === Assign DOFs for PBC ===
    Assign_DOF_PBC(NLC, NLS, NLM, NLP);
}

void problem_coupled::Assign_DOF_DBC() {
    int NoNs = static_cast<int>(Node_List.size());
    int dofs = 0;

    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd BC  = Node_List[i].BC;   // copy, like MATLAB
        Eigen::VectorXd DOF = Node_List[i].DOF;  // copy

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1.0) {
                std::cout<< "node: "<< i<<std::endl;
                dofs = dofs + 1;
                DOF(p) = dofs;
            }
        }

        Node_List[i].DOF = DOF;  // write back
    }

    DOFs = dofs;  // total
}



// Assign DOFs with periodic BC (node index lists are already 0-based)
void problem_coupled::Assign_DOF_PBC(const std::vector<int>& NLC,
                                     const std::vector<int>& NLS,
                                     const std::vector<int>& NLM,
                                     const std::vector<int>& NLP)
{
    DOFs = 0;

    const int NoCNs = static_cast<int>(NLC.size());
    const int NoSNs = static_cast<int>(NLS.size());
    const int NoMNs = static_cast<int>(NLM.size());
    const int NoPNs = static_cast<int>(NLP.size());

    // Corner nodes
    if (NoCNs != 0) {
        const int firstCorner = NLC[0];
        auto& DOF0 = Node_List[firstCorner].DOF;
        const auto& BC0 = Node_List[firstCorner].BC;
        for (Eigen::Index p = 0; p < BC0.size(); ++p) {
            if (BC0(p) == 1) {
                ++DOFs;
                DOF0(p) = static_cast<double>(DOFs);
            }
        }
        for (int i = 0; i < NoCNs; ++i) {
            int idx = NLC[i];
            Node_List[idx].DOF = DOF0;
        }
    }

    // Star nodes
    for (int i = 0; i < NoSNs; ++i) {
        int idx = NLS[i];
        auto& DOF = Node_List[idx].DOF;
        const auto& BC = Node_List[idx].BC;
        for (Eigen::Index p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++DOFs;
                DOF(p) = static_cast<double>(DOFs);
            }
        }
    }

    // Minus nodes
    for (int i = 0; i < NoMNs; ++i) {
        int idx = NLM[i];
        auto& DOF = Node_List[idx].DOF;
        const auto& BC = Node_List[idx].BC;
        for (Eigen::Index p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++DOFs;
                DOF(p) = static_cast<double>(DOFs);
            }
        }
    }

    // Plus nodes (copy from paired minus)
    const int nCopy = std::min(NoPNs, NoMNs);
    for (int i = 0; i < nCopy; ++i) {
        int idxP = NLP[i];
        int idxM = NLM[i];
        Node_List[idxP].BC  = Node_List[idxM].BC;
        Node_List[idxP].DOF = Node_List[idxM].DOF;
    }
}

void problem_coupled::Assign_GP_DOFs() {
    int GP_dofs = 0;
    int NoNs = Node_List.size();

    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC = Node_List[i].GP_BC;
        Eigen::VectorXd& DOF = Node_List[i].GP_DOF;

        // If any entry in GP_BC is 1, assign a new DOF index to all GP_DOF entries
        if ((BC.array() == 1).any()) {
            ++GP_dofs;
            DOF = Eigen::VectorXd::Constant(BC.size(), GP_dofs);
            Node_List[i].GP_DOF = DOF;
        }
    }

    GP_DOFs = GP_dofs;  // store total number of Gauss point DOFs
}



void problem_coupled::update(const Eigen::VectorXd& dx) {
    int NoNs = Node_List.size();
    int NoEs = Element_List.size();
    int PD   = this->PD;
//    std::cout<<"dx" << std::endl;
//    std::cout<<dx<<std::endl;
    // === Update node unknowns u based on DOFs and dx ===
    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC  = Node_List[i].BC;
        Eigen::VectorXd& DOF = Node_List[i].DOF;
        Eigen::VectorXd& u   = Node_List[i].u;

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                int dofIdx = static_cast<int>(DOF(p)) - 1;  // convert to 0-based index
                if (dofIdx >= 0 && dofIdx < dx.size()) {
                    u(p) += dx(dofIdx);
                }
            }
        }
//        std::cout<<'u' <<std::endl;
//        std::cout<<u<<std::endl;
        Node_List[i].u = u;
    }

    // === Update element field values from updated nodal u ===
    for (int e = 0; e < NoEs; ++e) {
        auto& elem = Element_List[e];

        const int NPE1 = elem.NPE1;
        const int NPE2 = elem.NPE2;

        // === Scalar field c (keep EXACTLY as before) ===
        Eigen::VectorXd c(NPE1);
        for (int i = 0; i < NPE1; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL1(i)) - 1; // 1-based -> 0-based
            c(i) = Node_List[nodeIdx].u(0);                  // c is the first entry
        }
//        std::cout<<'c' <<std::endl;
//        std::cout<<c<<std::endl;
        elem.c = c;

        // === Vector field v (use MATLAB first/last) ===
        // MATLAB: dim = PD; first = last + 1; last = last + dim, with c: first=1,last=1
        const int dim   = PD;
        int last0  = 0;          // after c at index 0 in C++
        int first0 = last0 + 1;  // v starts right after c -> index 1
        last0     += dim;        // end index of v (unused here, kept for clarity)

        Eigen::MatrixXd v(PD, NPE2);
        for (int i = 0; i < NPE2; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL2(i)) - 1; // 1-based -> 0-based
            v.col(i) = Node_List[nodeIdx].u.segment(first0, dim); // u(first0 : first0+PD-1)
        }
//        std::cout<<'v' <<std::endl;
//        std::cout<<v<<std::endl;
        elem.v = v;

    }


    // === Update element field values from updated nodal u ===
//    for (int e = 0; e < NoEs; ++e) {
//        auto& elem = Element_List[e];
//
//        int NPE1 = elem.NPE1;
//        int NPE2 = elem.NPE2;
//
//        // === Scalar field c (dimension ) ===
//// === Scalar field c (Vector of length NPE1) ===
//        Eigen::VectorXd c(NPE1);
//        for (int i = 0; i < NPE1; ++i) {
//            int nodeIdx = static_cast<int>(elem.NdL1(i)) - 1; // 1-based -> 0-based
//            // scalar field is the first entry in Node.u
//            c(i) = Node_List[nodeIdx].u(0);
//        }
//        elem.c = c;  // OK: VectorXd -> VectorXd
//
//
//        // === Vector field v (dimension PD) ===
//        const int dim   = PD;
//        const int first = last + 1;   // 1-based
//        last += dim;
//
//        Eigen::MatrixXd v(PD, NPE2);
//        for (int i = 0; i < NPE2; ++i) {
//            int nodeIdx = static_cast<int>(elem.NdL2(i)) - 1;      // 1-based -> 0-based
//            v.col(i) = Node_List[nodeIdx].u.segment(first - 1, dim); // u(first:last)
//        }
//        elem.v = v;
//    }
}

void problem_coupled::update_GP(const Eigen::VectorXd& dx_gp) {
    int NoNs = Node_List.size();

    for (int i = 0; i < NoNs; ++i) {
        const Eigen::VectorXd& BC  = Node_List[i].GP_BC;
        const Eigen::VectorXd& DOF = Node_List[i].GP_DOF;

        if ((BC.array() == 1).any()) {
            int dof_idx = static_cast<int>(DOF(0)) - 1;  // 0-based index
            if (dof_idx >= 0 && dof_idx < dx_gp.size()) {
                Node_List[i].GP_vals = Eigen::VectorXd::Constant(BC.size(), dx_gp(dof_idx));
            }
        }
    }
}

void problem_coupled::update_time() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    for (int e = 0; e < NoEs; ++e) {
        Element_List[e].cn = Element_List[e].c;
        Element_List[e].vn = Element_List[e].v;
    }

    for (int i = 0; i < NoNs; ++i) {
        Node_List[i].un = Node_List[i].u;
    }
}


void problem_coupled::downdate_time() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    for (int e = 0; e < NoEs; ++e) {
        Element_List[e].c = Element_List[e].cn;
        Element_List[e].v = Element_List[e].vn;
    }

    for (int i = 0; i < NoNs; ++i) {
        Node_List[i].u = Node_List[i].un;
    }
}


// Global residual assembly (R1 and R2)
Eigen::VectorXd problem_coupled::Residual(double dt)
{
    const int NoEs  = static_cast<int>(Element_List.size());
    if (NoEs == 0) return Eigen::VectorXd::Zero(DOFs);

    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    Eigen::VectorXd Rtot = Eigen::VectorXd::Zero(DOFs);

    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2] = Element_List[e].Residual(dt);
        const auto& NdL1 = Element_List[e].NdL1; // 1-based ids
        const auto& NdL2 = Element_List[e].NdL2;

        // R1 (scalar)
        {
            const int dim    = 1;          // cell
            const int first0 = 0;          // MATLAB first=1  ->  C++ 0
            const int last0  = first0 + dim - 1; // = 0

            for (int i = 0; i < NPE1; ++i) {
                const int nodeIdx = static_cast<int>(NdL1(i)) - 1;   // 1-based -> 0-based
                const auto& BC  = Node_List[nodeIdx].BC;
                const auto& DOF = Node_List[nodeIdx].DOF;

                for (int p0 = first0; p0 <= last0; ++p0) {          // only p0 = 0
                    if (BC(p0) == 1) {
                        const int g = static_cast<int>(DOF(p0)) - 1;    // global DOF 1-based -> 0-based
                        const int r = i * dim + (p0 - first0);          // = i
                        if (g >= 0 && g < Rtot.size() && r >= 0 && r < R1.size()) {
                            Rtot(g) += R1(r);
                        }
                    }
                }
            }
        }


        // R2 (vector)
        {
            const int dim   = this->PD;       // velocity block size
            const int first0 = 1;             // zero-based: v starts right after c at index 0
            const int last0  = first0 + dim - 1;

            for (int i = 0; i < NPE2; ++i) {
                const int nodeIdx = static_cast<int>(NdL2(i)) - 1;   // 1-based -> 0-based
                const auto& BC  = Node_List[nodeIdx].BC;
                const auto& DOF = Node_List[nodeIdx].DOF;

                for (int p0 = first0; p0 <= last0; ++p0) {          // p0 in [1, PD]
                    if (BC(p0) == 1) {
                        const int g = static_cast<int>(DOF(p0)) - 1;    // global DOF: 1-based -> 0-based
                        const int r = i * dim + (p0 - first0);          // R2 index: 0..(NPE2*PD-1)
                        if (g >= 0 && g < Rtot.size() && r >= 0 && r < R2.size()) {
                            Rtot(g) += R2(r);
                        }
                    }
                }
            }
        }

    }

    return Rtot;
}



void problem_coupled::assemble(double dt) {
    // === MATLAB: NoEs = size(EL,2); NoNs = size(NL,2); ===
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());

    // Guard for empty meshes (optional, but safer)
    if (NoEs == 0 || NoNs == 0) {
        Rtot = Eigen::VectorXd::Zero(DOFs);
        Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
        return;
    }

    // === MATLAB: NPE1 = EL(1).NPE1; NPE2 = EL(1).NPE2; ===
    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    // === MATLAB: Rtot = zeros(DOFs,1); ===
    Rtot = Eigen::VectorXd::Zero(DOFs);

    // === MATLAB: sprC = 0; % sparse counter ===
    // We'll track triplets instead of a manual counter.
    std::vector<Eigen::Triplet<double>> triplets;

    // === MATLAB: for e = 1:NoEs ... end  (empty body for now) ===
    for (int e = 0; e < NoEs; ++e) {
        // Assemble contribution of R1

        auto [R1, R2, K11, K12, K21, K22] = Element_List[e].RK(dt);
        const auto& NdL1 = Element_List[e].NdL1;
        const auto& NdL2 = Element_List[e].NdL2;

        const int dim   = 1;
        const int first = 0;
        const int last  = dim - 1;

            for (int i = 0; i < NPE1; ++i)
            {
                int nodeIdx = static_cast<int>(NdL1(i)) - 1;
                const auto& BC  = Node_List[nodeIdx].BC;
                const auto& DOF = Node_List[nodeIdx].DOF;

                for (int p = first; p <= last; ++p)
                {
                    if (BC(p) == 1)
                    {
                        int g = static_cast<int>(DOF(p)) - 1;
                        int r = i * dim + p;
                        Rtot(g) += R1(r);
                    }
                }

            }

        // Assemble contribution of R2
        {
            int dim = PD;
            int first = 1;         // corresponds to MATLAB last+1 (last=0 above)
            int last = first + dim - 1;

            for (int i = 0; i < NPE2; ++i) {
                int nodeIdx = static_cast<int>(NdL2(i)) - 1;
                const auto &BC = Node_List[nodeIdx].BC;
                const auto &DOF = Node_List[nodeIdx].DOF;

                for (int p = first; p <= last; ++p) {
                    if (BC(p) == 1) {
                        int g = static_cast<int>(DOF(p)) - 1;
                        int r = i * dim + (p - first);
                        Rtot(g) += R2(r);
                    }
                }

            }
        }
        // Assemble contributions of K11 and K12 into triplets
        {
            const int dim_i = 1;
            const int first_i = 0, last_i = 0;

            for (int i = 0; i < NPE1; ++i) {
                int node_i = static_cast<int>(NdL1(i)) - 1;
                const auto& BC_i  = Node_List[node_i].BC;
                const auto& DOF_i = Node_List[node_i].DOF;

                for (int p = first_i; p <= last_i; ++p) {
                    if (BC_i(p) != 1) continue;

                    const int row = static_cast<int>(DOF_i(p)) - 1;

                    // --- K11 ---
                    {
                        const int dim_j = 1;
                        const int first_j = 0, last_j = 0;

                        for (int j = 0; j < NPE1; ++j) {
                            int node_j = static_cast<int>(NdL1(j)) - 1;
                            const auto& BC_j  = Node_List[node_j].BC;
                            const auto& DOF_j = Node_List[node_j].DOF;

                            for (int q = first_j; q <= last_j; ++q) {
                                if (BC_j(q) != 1) continue;

                                const int col = static_cast<int>(DOF_j(q)) - 1;
                                const int rloc = i * dim_i + (p - first_i);          // = i
                                const int cloc = j * dim_j + (q - first_j);          // = j
                                triplets.emplace_back(row, col, K11(rloc, cloc));
                            }
                        }
                    }

                    // --- K12 ---
                    {
                        const int dim_j = PD;
                        const int first_j = 1, last_j = first_j + dim_j - 1;        // 1..PD

                        for (int j = 0; j < NPE2; ++j) {
                            int node_j = static_cast<int>(NdL2(j)) - 1;
                            const auto& BC_j  = Node_List[node_j].BC;
                            const auto& DOF_j = Node_List[node_j].DOF;

                            for (int q = first_j; q <= last_j; ++q) {
                                if (BC_j(q) != 1) continue;

                                const int col  = static_cast<int>(DOF_j(q)) - 1;
                                const int rloc = i * dim_i + (p - first_i);          // = i
                                const int cloc = j * dim_j + (q - first_j);          // j*PD + (q-1)
                                triplets.emplace_back(row, col, K12(rloc, cloc));
                            }
                        }
                    }
                }
            }
        }

        // Assemble contributions of K21 and K22 into triplets
        {
            const int dim_i = PD;
            const int first_i = 1, last_i = first_i + dim_i - 1;   // velocity block: 1..PD

            for (int i = 0; i < NPE2; ++i) {
                int node_i = static_cast<int>(NdL2(i)) - 1;
                const auto& BC_i  = Node_List[node_i].BC;
                const auto& DOF_i = Node_List[node_i].DOF;

                for (int p = first_i; p <= last_i; ++p) {
                    if (BC_i(p) != 1) continue;
                    const int row = static_cast<int>(DOF_i(p)) - 1;
                    const int rloc = i * dim_i + (p - first_i);     // i*PD + (p-1)

                    // --- K21 ---
                    {
                        const int dim_j = 1;
                        const int first_j = 0, last_j = 0;          // scalar block: 0..0

                        for (int j = 0; j < NPE1; ++j) {
                            int node_j = static_cast<int>(NdL1(j)) - 1;
                            const auto& BC_j  = Node_List[node_j].BC;
                            const auto& DOF_j = Node_List[node_j].DOF;

                            for (int q = first_j; q <= last_j; ++q) {
                                if (BC_j(q) != 1) continue;
                                const int col  = static_cast<int>(DOF_j(q)) - 1;
                                const int cloc = j * dim_j + (q - first_j); // = j
                                triplets.emplace_back(row, col, K21(rloc, cloc));
                            }
                        }
                    }

                    // --- K22 ---
                    {
                        const int dim_j = PD;
                        const int first_j = 1, last_j = first_j + dim_j - 1; // velocity block: 1..PD

                        for (int j = 0; j < NPE2; ++j) {
                            int node_j = static_cast<int>(NdL2(j)) - 1;
                            const auto& BC_j  = Node_List[node_j].BC;
                            const auto& DOF_j = Node_List[node_j].DOF;

                            for (int q = first_j; q <= last_j; ++q) {
                                if (BC_j(q) != 1) continue;
                                const int col  = static_cast<int>(DOF_j(q)) - 1;
                                const int cloc = j * dim_j + (q - first_j); // j*PD + (q-1)
                                triplets.emplace_back(row, col, K22(rloc, cloc));
                            }
                        }
                    }
                }
            }
        }

    }

    // Build an (empty-for-now) global K to keep the function self-contained
    Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
    Ktot.setFromTriplets(triplets.begin(), triplets.end());

}

void problem_coupled::assemble_GP(double dt) {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();
    int NPE1 = Element_List[0].NPE1;
    int NGP_val = Node_List[0].GP_BC.size();  // Number of Gauss point values

    Rtot_GP = Eigen::VectorXd::Zero(GP_DOFs * NGP_val);
    std::vector<Eigen::Triplet<double>> triplets;

    for (int e = 0; e < NoEs; ++e) {
        auto [R, K] = Element_List[e].RK_GP(dt, NGP_val);
        const Eigen::VectorXd& NdL1 = Element_List[e].NdL1;

        // === Assemble Rtot_GP ===
        for (int s = 0; s < NGP_val; ++s) {
            for (int i = 0; i < NPE1; ++i) {
                int node_i = static_cast<int>(NdL1(i)) - 1;
                const auto& BC = Node_List[node_i].GP_BC;
                const auto& DOF = Node_List[node_i].GP_DOF;

                for (int p = 0; p < 1; ++p) {  // scalar field only
                    if (BC(p) == 1) {
                        int dof_idx = static_cast<int>(DOF(p)) - 1;
                        if (dof_idx >= 0) {
                            int global_row = dof_idx * NGP_val + s;
                            int local_row = i * 1 + (p - 0);  // equivalent to i in this case
                            Rtot_GP(global_row) += R(local_row, s);
                        }
                    }
                }
            }
        }

        // === Assemble Ktot_GP ===
        for (int i = 0; i < NPE1; ++i) {
            int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BC_i = Node_List[node_i].GP_BC;
            const auto& DOF_i = Node_List[node_i].GP_DOF;

            for (int p = 0; p < 1; ++p) {
                if (BC_i(p) == 1) {
                    int row = static_cast<int>(DOF_i(p)) - 1;

                    for (int j = 0; j < NPE1; ++j) {
                        int node_j = static_cast<int>(NdL1(j)) - 1;
                        const auto& BC_j = Node_List[node_j].GP_BC;
                        const auto& DOF_j = Node_List[node_j].GP_DOF;

                        for (int q = 0; q < 1; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * 1 + (p - 0);
                                int k_idx_col = j * 1 + (q - 0);
                                triplets.emplace_back(row, col, K(k_idx_row, k_idx_col));
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize Ktot_GP sparse matrix
    Ktot_GP = Eigen::SparseMatrix<double>(GP_DOFs, GP_DOFs);
    Ktot_GP.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::MatrixXd problem_coupled::Get_all_velocity() {
    int NoNs = Node_List.size();
    int PD = this->PD;

    Eigen::MatrixXd v(PD, NoNs);

    for (int i = 0; i < NoNs; ++i) {
        v.col(i) = Node_List[i].u.segment(1, PD);  // Skip scalar field at index 0
    }

    return v;
}

#include <fstream>
#include <iomanip>
#include <iostream>

void problem_coupled::problem_info() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    std::cout << "======================================================" << std::endl;
    std::cout << "================  Problem information  ===============" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Problem dimension                   : " << PD << std::endl;
    std::cout << "Time increment                      : " << dt << std::endl;
    std::cout << "Final time                          : " << T << std::endl;
    std::cout << "Time increment method               : " << time_incr_method << std::endl;
    std::cout << "Number of nodes                     : " << NoNs << std::endl;
    std::cout << "Number of bulk elements             : " << NoEs << std::endl;
    std::cout << "Number of DOFs                      : " << DOFs << std::endl;

    std::cout << "Element order                       : [";
    for (size_t i = 0; i < element_order.size(); ++i) {
        std::cout << element_order[i];
        if (i != element_order.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "E  R  xi                            : [";
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << parameters[i];
        if (i != parameters.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "======================================================" << std::endl;

    std::ofstream file(filename);
    file << "======================================================\n";
    file << "================  Problem information  ===============\n";
    file << "======================================================\n";
    file << "Problem dimension                   : " << PD << "\n";
    file << "Time increment                      : " << dt << "\n";
    file << "Final time                          : " << T << "\n";
    file << "Time increment method               : " << time_incr_method << "\n";
    file << "Number of nodes                     : " << NoNs << "\n";
    file << "Number of bulk elements             : " << NoEs << "\n";
    file << "Number of DOFs                      : " << DOFs << "\n";

    file << "Element order                       : ";
    for (int val : element_order)
        file << val << " ";
    file << "\n";

    file << "E  R  xi                            : ";
    for (double val : parameters)
        file << val << " ";
    file << "\n";

    file << "======================================================\n\n\n";
    file.close();
}

std::pair<double, double> problem_coupled::calculate_max_min_difference() {
    int NoNs = Node_List.size();
    int PD = this->PD;

    std::vector<double> c_values, p_values;
    c_values.reserve(NoNs);
    p_values.reserve(NoNs);

    for (int i = 0; i < NoNs; ++i) {
        const auto& u = Node_List[i].u;
        c_values.push_back(u(0));              // c is the first entry
        p_values.push_back(u(1 + PD));         // p is after c and velocity
    }

    auto [c_min_it, c_max_it] = std::minmax_element(c_values.begin(), c_values.end());
    auto [p_min_it, p_max_it] = std::minmax_element(p_values.begin(), p_values.end());

    double c_range = *c_max_it - *c_min_it;
    double p_range = *p_max_it - *p_min_it;

    return {c_range, p_range};
}

//
//double problem_coupled::calculate_overall_density() {
//    double out = 0.0;
//    int NoE = Element_List.size();
//
//    for (int i = 0; i < NoE; ++i) {
//        out += Element_List[i].density;
//    }
//
//    return out;
//}

// Optional tiny helpers
static inline Eigen::VectorXd solve_sparse_linear_system(
        const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b)
{
    // Try a fast direct SPD solver first
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.compute(A);
    if (ldlt.info() == Eigen::Success) {
        std::cout<<"solved using simplicial"<<std::endl;
        return ldlt.solve(b);
    }
    // Fall back to general sparse LU
    Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
    slu.analyzePattern(A);
    slu.factorize(A);
    if (slu.info() == Eigen::Success) {
        std::cout<< "solved using slu"<<std::endl;
        return slu.solve(b);
    }
    // Last resort: iterative
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bicg;
    bicg.compute(A);
    return bicg.solve(b);
}

void problem_coupled::post_process() {
    // Minimal stub so you can compile/run. Hook your 1D/2D/3D exporters here.
    // e.g., write nodal c and v to file if you want.
}

void problem_coupled::output_step_info() {
    std::ofstream file(filename, std::ios::app);
    file << "\nStep number: " << counter
         << ",   Time increment: " << dt
         << ",   Current time: " << t << "\n\n";
}

void problem_coupled::solve() {
    // Drive the full Newton–Raphson time integration, mirroring your MATLAB flow.
    t = 0.0;
    counter = 1;

    while (t < T) {
        std::ofstream file(filename, std::ios::app);

        int  error_counter = 1;
        bool isNotAccurate = true;
        bool try_again     = false;

        while (isNotAccurate) {
            // ---------- Residual + Tangent ----------
            assemble(dt);  // fills: Ktot (Sparse), Rtot (Vector)
            double Norm_R0 = 1.0;
//            std::cout<<"Ktot at the end of first assemble"<<std::endl;
//            std::cout<<Eigen::MatrixXd (Ktot).rows() << " x "<< Eigen::MatrixXd (Ktot).cols() <<std::endl;
//            Eigen::MatrixXd dense = Eigen::MatrixXd(Ktot);
//            std::cout<<dense<<std::endl;
//            std::cout << " column (0):\n"
//                      << dense.col(0) << "\n";
//            std::cout << " column (1):\n"
//                      << dense.col(1) << "\n";
//            std::cout << " column (10):\n"
//                      << dense.col(10) << "\n";
//            std::cout<<"Rtot"<<std::endl;
//            std::cout<<Rtot<<std::endl;

            // ---------- Initial output & predictor residual ----------

            if (error_counter == 1) {
                if (counter == 1 && GP_vals == "On") {
                    assemble_GP(dt);
                    Eigen::VectorXd dx_GP = solve_sparse_linear_system(Ktot_GP, Rtot_GP);
                    update_GP(dx_GP);
                }

                // initial PostProcess (optional; mirrors MATLAB)
               //post_process();

                double Norm_R0 = Rtot.norm();
                std::cout << "Residual Norm at Predictor               : "
                          << std::scientific << Rtot.norm()
                          << " , normalized : 1\n";
                file << "Residual Norm at Predictor               : "
                     << std::scientific << Rtot.norm()
                     << " , normalized : 1\n";
            }

            // ---------- Solve & update (Corrector) ----------
            // MATLAB has: dx = -K \ R
            // We solve K y = R then take dx = -y
//            auto testing=Ktot;
//            auto b=Rtot;
//            std::cout<<"Frist try"<< "\n";
//            Eigen::SimplicialCholesky<SpMat> chol(testing);  // performs a Cholesky factorization of A
//            Eigen::VectorXd x = chol.solve(b);
//            std::cout<<"x"<<std::endl;
//            std::cout <<x<<std::endl;
//


            std::cout<<"second try" <<std::endl;
            // Fall back to general sparse LU
            auto testing2=Ktot;
            auto b2=Rtot;

            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            slu.analyzePattern(testing2);
            slu.factorize(testing2);
            //std::cout<< "solved using slu"<<std::endl;
            Eigen::VectorXd  dx = slu.solve(-b2);
//            std::cout<<"dx"<<std::endl;
//            std::cout <<dx<<std::endl;

//            std::cout<<"third try" <<std::endl;
//            // Fall back to general sparse LU
//            auto testing3=Ktot;
//            auto b3=Rtot;
//            // Last resort: iterative
//            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bicg;
//            bicg.compute(testing3);
//            Eigen::VectorXd x3= bicg.solve(b);
//            std::cout<<"x3"<<std::endl;
//            std::cout <<x3<<std::endl;



//            Eigen::VectorXd dx = solve_sparse_linear_system(Ktot, Rtot);

//
//            std::cout<< "dx" <<std::endl;
//            std::cout<<dx<<std::endl;
            Eigen::VectorXd rcheck = Ktot * dx + Rtot;   // should be ~ zero
            std::cout << "||K dx + R|| = " << rcheck.norm() << "\n";



//            std::cout<<"y"<<std::endl;
//            std::cout<<y<<std::endl;
            update(dx);   // updates node and element unknowns from DOFs

            // ---------- Recompute residual with updated values ----------
            Rtot = Residual(dt);

            const double Rn = Rtot.norm();
            std::cout << "Residual Norm @ Increment " << error_counter
                      << " at Corrector : " << std::scientific << Rn
                      << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";
            file << "Residual Norm @ Increment " << error_counter
                 << " at Corrector : " << std::scientific << Rn
                 << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";

            // ---------- Convergence / failure checks ----------
            if (Rn < tol) {
                isNotAccurate = false;
            } else if (error_counter > max_iter || Rn > 1e6) {
                isNotAccurate = false;
                std::cout << "Convergence is not obtained!\n";
                file << "Convergence is not obtained!\n";
                try_again = true;
            }

            ++error_counter;
        } // end Newton iterations

        // ---------- Early stop if velocities ~ 0 ----------
        {
            Eigen::MatrixXd velocity = Get_all_velocity();
            if (velocity.cwiseAbs().maxCoeff() < 1e-6) {
                post_process();
                break;
            }
        }

        // ---------- Time-step control ----------
        if (try_again) {
            if (dt < 1e-7) {
                std::cout << "\nSimulation terminated due to the very small time increment\n";
                file      << "\nSimulation terminated due to the very small time increment\n";
                break;
            } else {
                dt = dt / time_factor;
                std::cout << "\nReducing time step to: " << std::scientific << dt << "\n";
                file      << "\nReducing time step to: " << std::scientific << dt << "\n";
                downdate_time();
            }
        } else {
            // Accept the step
            t += dt;
            ++counter;

            if (GP_vals == "On") {
                assemble_GP(dt);
                Eigen::VectorXd dx_GP = solve_sparse_linear_system(Ktot_GP, Rtot_GP);
                update_GP(dx_GP);
            }

            post_process();
            update_time(); // commit c→cn, v→vn, u→un

            if (time_incr_method == "Adaptive") {
                if (error_counter < 5) {
                    dt = dt * time_factor;
                    std::ofstream f2(filename, std::ios::app);
                    std::cout << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                    f2        << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                } else if (error_counter > 8) {
                    dt = dt / time_factor;
                    std::ofstream f2(filename, std::ios::app);
                    std::cout << "\nReducing time step to: " << std::scientific << dt << "\n\n";
                    f2        << "\nReducing time step to: " << std::scientific << dt << "\n\n";
                }
            }

            output_step_info();
        }

        file.close();
    } // while t < T
}

